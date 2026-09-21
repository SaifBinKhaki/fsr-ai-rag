import json
import logging
import os
import secrets
import time

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, BackgroundTasks, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from scrape_btu import (
    scrape,
    scrape_structured,
    clear_scraped_data,
    clear_structured_data,
)
from scrape_module_links import (
    LINKS_FILE,
    ModuleLinksError,
    read_module_links,
    rebuild_module_links,
)
from scrape_professors import (
    PROFESSORS_META_FILE,
    PROFESSORS_OUTPUT_FILE,
    clear_professors_data,
    scrape_professors,
)
from ingest_to_qdrant import ingest, clear_qdrant_collection

# Load .env so API_BEARER_TOKEN (and the Qdrant vars) are available locally too.
load_dotenv()

# Configure the logging format and level
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- AUTH ---
# Every route below is gated behind a static bearer token read from the
# environment. Set API_BEARER_TOKEN in .env (or the container env) to a long
# random secret; requests must send `Authorization: Bearer <token>`.
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN")

# auto_error=False so we can return our own 401 (with WWW-Authenticate) instead
# of FastAPI's default 403 when the header is missing.
_bearer_scheme = HTTPBearer(auto_error=False)


def require_token(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> None:
    """Reject any request without a valid bearer token.

    Fails closed: if API_BEARER_TOKEN isn't configured the API is unusable
    rather than silently open. compare_digest avoids timing side-channels.
    """
    if not API_BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server auth is not configured (API_BEARER_TOKEN unset).",
        )

    if credentials is None or not secrets.compare_digest(
        credentials.credentials, API_BEARER_TOKEN
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Applying the dependency at the app level protects every route in one place.
app = FastAPI(dependencies=[Depends(require_token)])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    # Process the request
    response = await call_next(request)

    # Calculate latency
    process_time = time.time() - start_time
    logger.info(
        f"Method={request.method} Path={request.url.path} Latency={process_time:.4f}s"
    )

    return response


@app.post("/run-etl")
async def run_etl(background_tasks: BackgroundTasks):
    # This runs the process in the background so the API
    # doesn't time out while scraping
    background_tasks.add_task(execute_pipeline)
    return {"status": "Pipeline started"}


@app.post("/scrape-module-links")
async def scrape_module_links_route(dry_run: bool = False, min_links: int | None = None):
    """Rebuild `btu_subject_links.txt` from BTU's live module catalogue.

    Scrapes the qisserver3 module search, asked for in one page via
    `P_anzahl=9999`, and rewrites the link file to match it exactly. The file is
    **replaced**, not merged, so a module BTU has retired leaves the list here
    too; the file always reflects the real catalogue.

    This is the *currently offered* catalogue (~3,200 modules). The old
    `b-tu.de/modul` index this replaced also listed modules marked "no longer
    offered", which is why the list got shorter when the source changed.

    This is the input every other crawler reads, so it is worth running before a
    full `/run-etl` or `/scrape` to pick up the semester's new modules.

    One page fetch, so it runs inline rather than in the background and the
    response carries the diff: `added` and `removed` list the actual URLs.

    `dry_run=true` reports that diff without touching the file. `min_links`
    overrides the safety floor that rejects an implausibly short scrape — set it
    to 0 to write whatever was parsed.
    """
    try:
        report = await rebuild_module_links(dry_run=dry_run, min_links=min_links)
    except ModuleLinksError as exc:
        # The catalogue was unreachable or came back without its table. The old
        # link file is still intact, so this is a failed refresh, not data loss.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        )

    logger.info(
        f"Module links rebuilt: {report['total']} total "
        f"(+{report['added_count']}, -{report['removed_count']}), "
        f"written={report['written']}"
    )
    return {
        "status": "Dry run complete, file unchanged"
        if dry_run
        else "Module links rebuilt",
        **report,
    }


@app.get("/module-links")
async def get_module_links():
    """Return the module URLs currently in `btu_subject_links.txt`."""
    if not os.path.isfile(LINKS_FILE):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No link file yet. POST /scrape-module-links first.",
        )

    links = read_module_links(LINKS_FILE)
    return {"file": LINKS_FILE, "count": len(links), "links": links}


@app.post("/scrape")
async def scrape_only(background_tasks: BackgroundTasks):
    """Run only the scraper (no clear, no ingest). Existing files are resumed.

    Crawls every module in `btu_subject_links.txt` into the markdown corpus that
    feeds Qdrant, following each module's timetable events one level deep.
    """
    background_tasks.add_task(_run_scrape)
    return {"status": "Scrape started"}


@app.post("/scrape-json")
async def scrape_json(background_tasks: BackgroundTasks, fresh: bool = False):
    """Crawl every module page into structured JSON, event sub-pages nested in.

    Writes one file per module to `structured_data/`, separate from the markdown
    corpus that feeds Qdrant — this route touches neither `scraped_data/` nor the
    vector store.

    `fresh=false` (default) resumes: modules already written are skipped, so a
    crawl interrupted partway through picks up where it left off. `fresh=true`
    wipes `structured_data/` first to force a full re-crawl.
    """
    removed = clear_structured_data() if fresh else 0
    background_tasks.add_task(_run_scrape_structured)
    return {
        "status": "Structured JSON scrape started",
        "fresh": fresh,
        "cleared": removed,
    }


@app.delete("/structured-data")
async def delete_structured_files():
    """Delete every structured JSON file on disk (blocking, fast)."""
    removed = clear_structured_data()
    return {"status": "Structured files deleted", "removed": removed}


@app.post("/scrape-professors")
async def scrape_professors_route(background_tasks: BackgroundTasks, fresh: bool = False):
    """Crawl every faculty -> chair -> team page into one professors JSON.

    Walks the six faculty overviews, follows each chair to its "Team" menu, and
    reads every designation page under it — professors, academic staff,
    secretariat, student assistants and alumni alike. Each person carries their
    contact block, portrait, CV and office hours where the chair publishes them,
    and everyone is de-duplicated across the pages that list them.

    `structured-professors/data.json` is written as one flat dictionary — person
    key -> that person's details, every value a scalar, faculty/institute/chair
    carried as plain fields. Run counters land in `metadata.json` beside it.

    Independent of both other corpora: this route touches neither `scraped_data/`
    nor `structured_data/` nor the vector store.

    The crawl always runs in full (roughly 1,700 pages, a couple of minutes) and
    overwrites the export at the end, so `fresh=true` only matters if you want the
    stale file gone while the new one is still being built.
    """
    removed = clear_professors_data() if fresh else 0
    background_tasks.add_task(_run_scrape_professors)
    return {
        "status": "Professor scrape started",
        "fresh": fresh,
        "cleared": removed,
        "output_file": PROFESSORS_OUTPUT_FILE,
    }


@app.get("/professors")
async def get_professors(faculty: int | None = None, q: str | None = None):
    """Return the scraped professor data, optionally filtered.

    The stored file is one flat dictionary keyed by person, and that is what this
    returns under `people`. `faculty=3` narrows it to that faculty and `q=hauer`
    matches name, email or chair; with neither, the whole roster comes back.
    """
    if not os.path.isfile(PROFESSORS_OUTPUT_FILE):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No professor data yet. POST /scrape-professors first.",
        )

    with open(PROFESSORS_OUTPUT_FILE, encoding="utf-8") as handle:
        roster = json.load(handle)

    if faculty is not None:
        needle = f"fakultät {faculty}".casefold()
        roster = {
            key: person
            for key, person in roster.items()
            if needle in (person["faculty"] or "").casefold()
        }
    if q:
        needle = q.casefold()
        roster = {
            key: person
            for key, person in roster.items()
            if needle in person["name"].casefold()
            or needle in (person["email"] or "").casefold()
            or needle in (person["chair"] or "").casefold()
        }

    metadata = None
    if os.path.isfile(PROFESSORS_META_FILE):
        with open(PROFESSORS_META_FILE, encoding="utf-8") as handle:
            metadata = json.load(handle)["metadata"]

    return {
        "metadata": metadata,
        "filters": {"faculty": faculty, "q": q},
        "count": len(roster),
        "people": roster,
    }


@app.delete("/professors")
async def delete_professors_data():
    """Delete the exported professors JSON (blocking, fast)."""
    removed = clear_professors_data()
    return {"status": "Professor data deleted", "removed": removed}


@app.post("/ingest")
async def ingest_only(background_tasks: BackgroundTasks):
    """Run only the Qdrant ingestion over whatever is currently in scraped_data."""
    background_tasks.add_task(_run_ingest)
    return {"status": "Ingestion started"}


@app.post("/qdrant/clear")
async def qdrant_clear():
    """Empty the Qdrant cluster: drop and recreate the collection (blocking, fast)."""
    result = clear_qdrant_collection()
    return {"status": "Qdrant collection emptied", **result}


@app.delete("/scraped-data")
async def delete_scraped_files():
    """Delete every scraped JSON file on disk (blocking, fast)."""
    removed = clear_scraped_data()
    return {"status": "Scraped files deleted", "removed": removed}


def _run_scrape():
    import asyncio

    asyncio.run(scrape())
    logger.info("Scrape-only run complete.")


def _run_scrape_structured():
    import asyncio

    asyncio.run(scrape_structured())
    logger.info("Structured JSON scrape complete.")


def _run_scrape_professors():
    import asyncio

    meta = asyncio.run(scrape_professors())
    logger.info(
        f"Professor scrape complete: {meta['people']} people across "
        f"{meta['chairs']} chairs."
    )


def _run_ingest():
    ingest()
    logger.info("Ingest-only run complete.")


async def execute_pipeline():
    # 0. Wipe the previous scrape so every run crawls the latest content.
    clear_scraped_data()

    # 1. Run your scraper (re-fetches every URL since the dir is now empty)
    await scrape()

    # 2. Run your ingestion (change detection decides what to update in Qdrant)
    ingest()

    logger.info("ETL cycle complete.")

import logging
import os
import secrets
import time

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, BackgroundTasks, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from scrape_btu import scrape, clear_scraped_data
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


@app.post("/scrape")
async def scrape_only(background_tasks: BackgroundTasks):
    """Run only the scraper (no clear, no ingest). Existing files are resumed."""
    background_tasks.add_task(_run_scrape)
    return {"status": "Scrape started"}


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

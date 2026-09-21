"""Crawl BTU's qisserver3 module catalogue into two corpora.

Both start from ``btu_subject_links.txt`` (written by ``scrape_module_links``),
which holds one qisserver3 module-description URL per line:

* ``scrape()`` writes ``scraped_data/`` — one markdown-in-JSON file per module,
  the corpus ``ingest_to_qdrant`` embeds.
* ``scrape_structured()`` writes ``structured_data/`` — the same crawl as typed
  fields instead of prose, for consumers that want to query rather than read.

Both follow each module's "Veranstaltungen im aktuellen Semester" links one
level deep into the qisserver3 timetable and fold those sub-pages into the
parent module, because lecturer, room and schedule live only there.

Two things changed when the source moved from the old ``b-tu.de/modul`` index to
qisserver3, and they are worth knowing before reading on:

**No browser.** These pages are server-rendered with no JavaScript, and
qisserver3 serves them to a plain HTTP client without a session — measured at 8
concurrent pages in ~1s. The crawl used to drive headless Chromium through
crawl4ai at ~100MB per tab, which on the 1GB Railway box capped concurrency at
3. httpx costs kilobytes per request, so ``SCRAPE_CONCURRENCY`` can be raised
instead of rationed. This mirrors what ``scrape_professors`` already does.

**Markdown is rendered, not converted.** ``btu_markdown`` builds the markdown
from the parsed structure rather than converting the page's HTML. qisserver3
pages carry navigation chrome and embed each event's QR code as a base64
``data:`` URI, all of which used to land in the corpus. See that module for why
rendering also makes change detection honest.

The two corpora name their files differently, on purpose:

* ``structured_data/modul_11101.json`` is keyed on the human-facing module
  number, so the file you open is the module you were looking for.
* ``scraped_data/modul_p6951.json`` is keyed on the ``pord.pordnr`` from the
  URL, with a ``p`` prefix so the two id kinds can never be confused.

The module number is only in the page *body*, not the URL, so it cannot be known
before fetching — which is exactly what the resume check has to decide. The
structured crawl therefore rebuilds its "already done" index by reading
``source.pordnr`` back out of the files it wrote (see ``_structured_pordnrs``),
while the markdown crawl, whose names come straight from the URL, just lists the
directory. Both records carry the module number and the pordnr either way.
"""

import asyncio
import hashlib
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from btu_markdown import render_module_markdown
from btu_parser import extract_veranstid, parse_event_page, parse_module_page

logger = logging.getLogger(__name__)

# Configuration
INPUT_FILE = os.getenv("MODULE_LINKS_FILE", "btu_subject_links.txt")
OUTPUT_DIR = "scraped_data"

# Where scrape_structured() writes. Kept separate from OUTPUT_DIR so the
# markdown corpus that feeds Qdrant and the structured corpus can be refreshed
# independently — neither run can clobber the other.
STRUCTURED_OUTPUT_DIR = "structured_data"

# Concurrent HTTP requests. Well above the old browser-tab budget of 3: an httpx
# request costs kilobytes rather than ~100MB, so the 1GB box is no longer the
# binding constraint. Still low enough to stay a polite guest on a university
# server. (The former SCRAPE_BATCH_SIZE is gone with the browser; this replaces
# it.)
CONCURRENCY = int(os.getenv("SCRAPE_CONCURRENCY", "8"))

# How many times to re-attempt a URL that timed out or was blocked. b-tu.de
# stalls flagged requests from datacenter IPs instead of refusing them, so a
# page that fails once often succeeds on a later, more spaced-out attempt.
MAX_RETRIES = int(os.getenv("SCRAPE_MAX_RETRIES", "3"))

REQUEST_TIMEOUT = float(os.getenv("SCRAPE_TIMEOUT", "45"))

# Jitter between requests so a burst doesn't arrive as a synchronized pattern,
# which is what most anti-bot heuristics key on.
MIN_DELAY = float(os.getenv("SCRAPE_MIN_DELAY", "0.05"))
MAX_DELAY = float(os.getenv("SCRAPE_MAX_DELAY", "0.25"))

# A realistic desktop UA is the single cheapest anti-bot mitigation: a default
# client UA is flagged instantly.
USER_AGENT = os.getenv(
    "SCRAPE_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
)

# Shared with scrape_professors and scrape_module_links so one setting steers
# every crawler. Routing through a rotating residential proxy is the real fix
# once UA and retry tuning aren't enough.
SCRAPE_PROXY = os.getenv("SCRAPE_PROXY", "").strip()
SCRAPE_PROXY_USER = os.getenv("SCRAPE_PROXY_USER", "").strip()
SCRAPE_PROXY_PASS = os.getenv("SCRAPE_PROXY_PASS", "").strip()

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STRUCTURED_OUTPUT_DIR, exist_ok=True)


def _proxy_url() -> str | None:
    """The configured gateway proxy with credentials folded in, or None."""
    if not SCRAPE_PROXY:
        return None
    if SCRAPE_PROXY_USER and "@" not in SCRAPE_PROXY:
        scheme, _, rest = SCRAPE_PROXY.partition("://")
        return f"{scheme}://{SCRAPE_PROXY_USER}:{SCRAPE_PROXY_PASS}@{rest}"
    return SCRAPE_PROXY


# --- FILENAMES ---

# The module's identity in its own URL. qisserver3 keys module descriptions on
# `pord.pordnr`, which is unrelated to the module number printed on the page.
PORDNR_RE = re.compile(r"[?&]pord\.pordnr=(\d+)")


def module_pordnr(url: str) -> str | None:
    """The `pord.pordnr` a module URL is keyed by, or None if it carries none."""
    match = PORDNR_RE.search(url)
    return match.group(1) if match else None


def module_filename(url: str) -> str:
    """Filename for a module keyed on its URL: ``modul_p<pordnr>.json``.

    The ``p`` prefix is there to stop a pordnr being misread as a module number —
    they look alike and mean different things, and a file called
    ``modul_6951.json`` would be a standing invitation to confuse them.

    A URL carrying no pordnr falls back to a hash of the URL, which keeps this
    total if the link list ever grows another kind of page.
    """
    pordnr = module_pordnr(url)
    if pordnr:
        return f"modul_p{pordnr}.json"

    return f"modul_x{hashlib.md5(url.encode('utf-8')).hexdigest()[:12]}.json"


# The module number as it should appear in a filename. The page prints a
# phase-out marker inside the same cell ("12566 - Auslaufmodul", "11154 -
# Phase-out Module"), so the number is taken as the leading digits rather than
# the whole string — otherwise the marker ends up in the filename, spaces and
# all. Verified unique across the catalogue: 3,237 modules, 3,237 distinct
# numbers after this trim.
MODULE_NUMBER_RE = re.compile(r"\s*(\d+)")


def structured_filename(url: str, parsed: dict | None = None) -> str:
    """Filename for the structured corpus: ``modul_<module number>.json``.

    Keyed on the human-facing module number rather than the ``pordnr`` in the
    URL, so ``structured_data/modul_11101.json`` is the file you open when you
    want module 11101 instead of something you have to grep the directory for.

    The catch is that the number lives in the *page body*, not in the URL, so
    this needs `parsed` and a caller can only name the file after fetching it.
    That is why the structured crawl's resume check reads
    ``_structured_pordnrs()`` instead of just listing filenames. Without a parse
    — or on the odd record that carries no number — this falls back to the
    URL-derived name so every module still gets a file.
    """
    if parsed:
        number = (parsed.get("fields") or {}).get("module_number")
        match = MODULE_NUMBER_RE.match(str(number or ""))
        if match:
            return f"modul_{match.group(1)}.json"

    return module_filename(url)


def _structured_pordnrs(directory: str) -> set[str]:
    """The pordnrs already written to the structured corpus.

    The structured files are named after the module number, which the URL does
    not carry, so the resume check cannot work by testing filenames the way the
    markdown crawl does. Reading `source.pordnr` back out of each file derives
    that index from the corpus itself — which costs ~0.6s for 3,200 files, and
    unlike a sidecar index cannot drift out of step with what is on disk.
    """
    pordnrs: set[str] = set()
    if not os.path.isdir(directory):
        return pordnrs

    for entry in os.scandir(directory):
        if not (entry.is_file() and entry.name.endswith(".json")):
            continue
        try:
            with open(entry.path, encoding="utf-8") as handle:
                pordnr = json.load(handle).get("source", {}).get("pordnr")
        except (json.JSONDecodeError, OSError):
            # A truncated file from an interrupted write: leave it out of the
            # index so the module is simply crawled again.
            continue
        if pordnr:
            pordnrs.add(str(pordnr))
    return pordnrs


# The markdown corpus names files straight from the URL, so listing the
# directory is all its resume check needs.
url_to_safe_filename = module_filename


# --- HOUSEKEEPING ---


def _clear_json_dir(directory: str) -> int:
    """Delete every .json file in `directory`, returning how many were removed."""
    if not os.path.isdir(directory):
        return 0

    removed = 0
    for entry in os.scandir(directory):
        if entry.is_file() and entry.name.endswith(".json"):
            os.remove(entry.path)
            removed += 1
    return removed


def clear_structured_data() -> int:
    """Delete previously written structured JSON so the next run starts fresh.

    Only touches STRUCTURED_OUTPUT_DIR — the markdown corpus behind Qdrant is
    left alone, so wiping structured output can never cost you the vector store.
    """
    removed = _clear_json_dir(STRUCTURED_OUTPUT_DIR)
    logger.info(f"🧹 Cleared {removed} previously structured files for a fresh crawl.")
    return removed


def clear_scraped_data() -> int:
    """Delete all previously scraped JSON files so the next crawl starts fresh.

    Run before scrape() for the weekly full refresh: with no existing files the
    resume/skip logic re-fetches every URL, guaranteeing the latest content.
    Qdrant still holds the prior vectors, so a mid-run failure only delays the
    refresh — it doesn't wipe the knowledge base.

    Returns the number of files removed so callers (e.g. the API) can report it.
    """
    removed = _clear_json_dir(OUTPUT_DIR)
    logger.info(f"🧹 Cleared {removed} previously scraped files for a fresh crawl.")
    return removed


def load_and_clean_urls(filepath: str) -> list[str]:
    """Reads links from txt file, removes duplicates and whitespace."""
    if not os.path.exists(filepath):
        logger.error(f"❌ Input file not found: {filepath}")
        raise FileNotFoundError()

    with open(filepath, "r", encoding="utf-8") as f:
        # Strip whitespace, ignore empty lines and comment lines
        urls = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    # dict.fromkeys rather than set(): de-duplicates while keeping catalogue
    # order, so a crawl's progress log follows the module numbering.
    unique_urls = list(dict.fromkeys(urls))
    logger.info(
        f"📄 Loaded {len(urls)} links from file "
        f"({len(unique_urls)} unique URLs after deduplication)."
    )
    return unique_urls


# --- FETCHING ---


@dataclass
class FetchResult:
    """One page fetch, with the parse that decides whether it really worked."""

    url: str
    status_code: int | None
    html: str
    success: bool
    error_message: str | None
    parsed: dict | None = None


class SessionPool:
    """A pool of independent HTTP sessions that fetch a list of URLs.

    Each worker owns its own ``httpx.AsyncClient``, which is the whole point:
    **qisserver3 keys navigation state to the session cookie**, so two requests
    sharing one cookie jar overwrite each other's state and the server answers
    with a page that is HTTP 200 but carries no module table. Measured over 24
    modules: one shared client parsed 12/24 at 8-way concurrency and 7/24 at
    2-way, while one client *per worker* parsed 24/24 at every width tried.

    So concurrency here is a count of sessions, not of requests in flight on one
    session — and raising it is safe in a way that a plain semaphore was not.
    """

    def __init__(self, workers: int = CONCURRENCY):
        self.workers = max(1, workers)
        self.stats = {"requests": 0, "failures": 0, "retries": 0}

    def _new_client(self) -> httpx.AsyncClient:
        proxy = _proxy_url()
        return httpx.AsyncClient(
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
            },
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
            proxy=proxy,
        )

    async def _fetch_one(self, client: httpx.AsyncClient, url: str, parse) -> FetchResult:
        """Fetch and parse one URL, retrying a page that came back unusable.

        A 200 whose body holds no module/event table counts as a failure, not as
        an empty record: qisserver3 reports an exhausted or clashing session that
        way rather than with a status code, and writing those would quietly fill
        the corpus with contentless documents.

        The cookie jar is dropped between attempts so a retry starts from a fresh
        session instead of the one that just failed.
        """
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            if attempt > 1:
                self.stats["retries"] += 1
                client.cookies.clear()
                await asyncio.sleep(2 * attempt)

            await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
            try:
                self.stats["requests"] += 1
                response = await client.get(url)
            except Exception as exc:  # network error, timeout, proxy failure
                last_error = f"{type(exc).__name__}: {exc}"
                continue

            if not response.is_success:
                last_error = f"HTTP {response.status_code}"
                # A 4xx that isn't rate limiting won't fix itself on a retry.
                if response.status_code < 500 and response.status_code != 429:
                    break
                continue

            # The *returned* URL: qisserver3 rewrites session params on redirect,
            # and the page we parsed is the one worth recording.
            final_url = str(response.url)
            parsed = parse(response.text, final_url)
            if parsed.get("parsed"):
                return FetchResult(
                    url=final_url,
                    status_code=response.status_code,
                    html=response.text,
                    success=True,
                    error_message=None,
                    parsed=parsed,
                )

            last_error = parsed.get("parse_error") or "page carried no content table"

        self.stats["failures"] += 1
        return FetchResult(
            url=url,
            status_code=None,
            html="",
            success=False,
            error_message=last_error or "unknown error",
            parsed=None,
        )

    async def map(self, urls: list[str], parse) -> dict[str, FetchResult]:
        """Fetch every URL across the pool. Keyed by the URL as passed in.

        Keyed on the requested URL rather than the returned one because a
        redirect rewrites the latter, and callers only know what they asked for.
        """
        if not urls:
            return {}

        queue: asyncio.Queue = asyncio.Queue()
        for url in urls:
            queue.put_nowait(url)

        results: dict[str, FetchResult] = {}

        async def worker() -> None:
            async with self._new_client() as client:
                while True:
                    try:
                        url = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    results[url] = await self._fetch_one(client, url, parse)

        await asyncio.gather(*(worker() for _ in range(min(self.workers, len(urls)))))
        return results


# --- EVENT SUB-PAGES ---


def _event_key(url: str) -> str:
    """Stable identity for an event sub-page.

    Keyed on the `veranstid` rather than the full URL because qisserver3 rewrites
    session/tracking params on redirect: matching the returned URL against the
    requested one string-for-string silently loses pages that were fetched fine.
    """
    return extract_veranstid(url) or url


async def _fetch_event_pages(
    pool: SessionPool, urls: list[str], event_cache: dict[str, dict]
) -> None:
    """Fetch and parse event sub-pages, filling `event_cache` in place.

    The cache is run-level, so a course event linked from twenty modules is
    fetched once for the whole crawl rather than once per chunk. Failures are
    cached too — a page that never came back is recorded as unavailable so a dead
    link can't be re-attempted by every module that references it.
    """
    pending = [u for u in urls if _event_key(u) not in event_cache]
    if not pending:
        return

    logger.info(f"  🔗 Fetching {len(pending)} new event sub-page(s)...")
    results = await pool.map(pending, parse_event_page)

    failures = 0
    for requested in pending:
        result = results[requested]
        key = _event_key(requested)
        if not result.success:
            failures += 1
            event_cache[key] = {
                "url": requested,
                "veranstid": extract_veranstid(requested),
                "fetched": False,
                "fetch_error": result.error_message,
                "status_code": result.status_code,
                "parsed": False,
            }
            continue

        parsed = result.parsed
        parsed["url"] = result.url
        parsed["status_code"] = result.status_code
        parsed["fetched"] = True
        parsed["fetch_error"] = None
        event_cache[key] = parsed

    if failures:
        logger.warning(f"  ⚠️ {failures} event sub-page(s) unreachable this run.")


def _module_events(parsed: dict, event_cache: dict[str, dict]) -> list[dict]:
    """This module's event records, in the order the module lists them.

    Each carries `listed_as` — how the parent module labelled the event
    ("430912 Vorlesung ... 2 SWS") — which is context the sub-page never repeats.
    """
    events = []
    for link in parsed["event_links"]:
        event = dict(event_cache.get(_event_key(link["url"]), {}))
        event["listed_as"] = link["title"]
        event.setdefault("url", link["url"])
        event.setdefault("veranstid", link["veranstid"])
        event.setdefault("fetched", False)
        event.setdefault("fetch_error", "Sub-page was never fetched.")
        events.append(event)
    return events


# --- PAYLOADS ---


def _build_markdown_payload(
    result: FetchResult, parsed: dict, events: list[dict], filename: str
) -> dict:
    """One module as markdown-in-JSON — the shape `ingest_to_qdrant` reads.

    `doc_id`, `content_markdown` and `metadata.url` are that module's contract
    with the ingester; everything else here is for humans reading the file.
    """
    markdown = render_module_markdown(parsed, result.url, events)

    return {
        "doc_id": filename.replace(".json", ""),
        "metadata": {
            "url": result.url,
            "title": (parsed["fields"].get("module_title") or parsed["page_title"]),
            "module_number": parsed["fields"].get("module_number"),
            "pordnr": module_pordnr(result.url),
            "status_code": result.status_code,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "word_count": len(markdown.split()),
            "nested_event_pages": [e.get("url") for e in events if e.get("fetched")],
        },
        "content_markdown": markdown,
    }


def _build_structured_payload(
    result: FetchResult, parsed: dict, events: list[dict], filename: str
) -> dict:
    """One module's structured record, sub-pages included.

    Everything the two page types carry ends up under a single record, so a
    consumer never has to join two files to see a course's schedule.
    """
    return {
        "doc_id": filename.replace(".json", ""),
        "source": {
            "url": result.url,
            "pordnr": module_pordnr(result.url),
            "status_code": result.status_code,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "page_title": parsed["page_title"],
            "heading": parsed["heading"],
            # Which label set the page used; the site publishes modules in either
            # German or English and both map onto the same field names.
            "language": parsed["language"],
            "parsed": parsed["parsed"],
            "parse_error": parsed["parse_error"],
        },
        "module": dict(parsed["fields"]),
        # Non-empty only if qisserver3 adds a row we have no mapping for, which
        # keeps a layout change visible instead of silently dropping data.
        "unmapped_fields": parsed["unmapped_fields"],
        "events": events,
        "event_summary": {
            "linked": len(parsed["event_links"]),
            "fetched": sum(1 for e in events if e.get("fetched")),
            "failed": sum(1 for e in events if not e.get("fetched")),
        },
    }


# --- CRAWL DRIVER ---
#
# The two corpora resume differently because they name files differently. The
# markdown corpus derives its name from the URL, so listing the directory
# answers "already done?". The structured corpus names files after the module
# number, which is only in the page body, so it has to read the pordnr back out
# of the files it already wrote.


def _markdown_pending(all_urls: list[str], output_dir: str) -> list[str]:
    """Modules with no markdown file yet."""
    existing = set(os.listdir(output_dir))
    return [url for url in all_urls if module_filename(url) not in existing]


def _structured_pending(all_urls: list[str], output_dir: str) -> list[str]:
    """Modules with no structured file yet, matched on pordnr."""
    done = _structured_pordnrs(output_dir)
    existing = set(os.listdir(output_dir))

    pending = []
    for url in all_urls:
        pordnr = module_pordnr(url)
        if pordnr:
            if pordnr not in done:
                pending.append(url)
        # No pordnr means the fallback name, which the URL does give us.
        elif module_filename(url) not in existing:
            pending.append(url)
    return pending


def _markdown_name(result: FetchResult, parsed: dict) -> str:
    return module_filename(result.url)


def _structured_name(result: FetchResult, parsed: dict) -> str:
    return structured_filename(result.url, parsed)


async def _crawl(output_dir: str, build_payload, pending_for, name_for, label: str) -> None:
    """Crawl every module in the link list into `output_dir`.

    Shared by both entry points: the only difference between the markdown and
    structured corpora is `build_payload`, since both parse the same pages with
    the same parser and differ only in how they write the result.

    Resumes: a module whose file already exists is skipped, so a crawl
    interrupted partway through does not start over. A module that could not be
    fetched *or parsed* is not written at all, which means the next run retries
    it rather than leaving a contentless file behind.
    """
    all_urls = load_and_clean_urls(INPUT_FILE)

    urls_to_scrape = pending_for(all_urls, output_dir)

    skipped_count = len(all_urls) - len(urls_to_scrape)
    if skipped_count:
        logger.info(
            f"⏩ Resuming job: skipped {skipped_count} already-written module(s). "
            f"{len(urls_to_scrape)} remaining."
        )

    if not urls_to_scrape:
        logger.info(f"🎉 Every module is already in {output_dir}!")
        return

    logger.info(f"🚀 Starting {label} crawl across {CONCURRENCY} session(s)...")
    start_time = asyncio.get_event_loop().time()

    pool = SessionPool(CONCURRENCY)
    # Shared across the whole run so each event sub-page is fetched exactly once,
    # even when several modules link to the same course.
    event_cache: dict[str, dict] = {}
    written = 0
    failed: list[str] = []

    # Chunked only to bound peak memory — a chunk's HTML is parsed and written
    # before the next is fetched, so thousands of pages never sit in memory at
    # once. Width of the crawl is the session pool's job, not the chunk's.
    chunk_size = max(CONCURRENCY * 8, 32)
    total_chunks = (len(urls_to_scrape) + chunk_size - 1) // chunk_size

    for index in range(0, len(urls_to_scrape), chunk_size):
        chunk = urls_to_scrape[index : index + chunk_size]
        chunk_num = index // chunk_size + 1

        results = await pool.map(chunk, parse_module_page)

        pages = []
        event_urls: dict[str, None] = {}
        for url in chunk:
            result = results[url]
            if not result.success:
                logger.info(f"  ⚠️ Failed: {url} | {result.error_message}")
                failed.append(url)
                continue
            pages.append((result, result.parsed))
            for link in result.parsed["event_links"]:
                event_urls.setdefault(link["url"], None)

        await _fetch_event_pages(pool, list(event_urls), event_cache)

        for result, parsed in pages:
            events = _module_events(parsed, event_cache)
            filename = name_for(result, parsed)
            payload = build_payload(result, parsed, events, filename)
            path = os.path.join(output_dir, filename)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
            written += 1

        logger.info(
            f"📦 Chunk {chunk_num}/{total_chunks}: wrote {len(pages)}/{len(chunk)} "
            f"module(s). {written} done, {len(event_cache)} event page(s) cached."
        )

    elapsed = round(asyncio.get_event_loop().time() - start_time, 2)
    if failed:
        logger.warning(
            f"⚠️ {len(failed)} module(s) still failed after {MAX_RETRIES} attempts "
            f"and were skipped this run (they'll be retried next crawl)."
        )
    logger.info(
        f"🏁 {label.capitalize()} crawl finished in {elapsed}s: {written} module(s) "
        f"written, {len(event_cache)} event sub-page(s) fetched, "
        f"{pool.stats['requests']} request(s) ({pool.stats['retries']} retried)."
    )


async def scrape() -> None:
    """Crawl every module into the markdown corpus that feeds Qdrant.

    Writes one file per module to ``scraped_data/``, each holding the module and
    its current-semester events rendered as markdown. Resumes by default; call
    clear_scraped_data() first for a full refresh.
    """
    await _crawl(
        OUTPUT_DIR, _build_markdown_payload, _markdown_pending, _markdown_name, "markdown"
    )


async def scrape_structured() -> None:
    """Crawl every module into typed JSON, sub-pages nested inside.

    Writes one file per module to ``structured_data/``. Independent of the
    markdown corpus and of the vector store; resumes the same way.
    """
    await _crawl(
        STRUCTURED_OUTPUT_DIR,
        _build_structured_payload,
        _structured_pending,
        _structured_name,
        "structured",
    )

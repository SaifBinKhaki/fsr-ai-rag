import asyncio
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    ProxyConfig,
    RoundRobinProxyStrategy,
)
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
import logging

from btu_parser import extract_veranstid, parse_event_page, parse_module_page

# Configure the logging format and level
logger = logging.getLogger(__name__)

# Configuration
INPUT_FILE = "btu_subject_links.txt"
OUTPUT_DIR = "scraped_data"

# Where scrape_structured() writes. Kept separate from OUTPUT_DIR so the
# markdown corpus that feeds Qdrant and the structured corpus can be refreshed
# independently — neither run can clobber the other.
STRUCTURED_OUTPUT_DIR = "structured_data"

# Number of concurrent browser tabs. Each open Chromium tab costs ~50-120MB, so
# on a 1GB Railway box this is the single biggest memory lever. Default 3 keeps
# peak browser memory to a few hundred MB; raise SCRAPE_BATCH_SIZE if you have
# more RAM. It trades a little speed for staying under the memory ceiling.
BATCH_SIZE = int(os.getenv("SCRAPE_BATCH_SIZE", "3"))

# How many times to re-attempt a URL that timed out / was blocked before giving
# up. b-tu.de sits behind an anti-bot layer that intermittently stalls requests
# from datacenter IPs, so a page that fails once often succeeds on a later, more
# spaced-out attempt. Retries run in a separate final pass with a growing delay.
MAX_RETRIES = int(os.getenv("SCRAPE_MAX_RETRIES", "2"))

# A realistic desktop UA is the single cheapest anti-bot mitigation: the default
# headless Chromium UA advertises "HeadlessChrome", which WAFs flag instantly.
USER_AGENT = os.getenv(
    "SCRAPE_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
)

# Optional proxy support. b-tu.de's anti-bot layer flags Railway's datacenter IP,
# so routing through a rotating *residential* proxy is the real fix once UA/retry
# tuning isn't enough. All of this is opt-in via env vars — with none set the
# scraper behaves exactly as before (direct connection).
#
#   SCRAPE_PROXY          One gateway endpoint. For a backconnect/rotating proxy
#                         (Bright Data, Oxylabs, Decodo, IPRoyal, ...) the provider
#                         rotates the exit IP for you, so a single endpoint is all
#                         you need. Format: "http://user:pass@host:port" or, if
#                         auth is passed separately, "http://host:port".
#   SCRAPE_PROXY_USER /   Credentials, when not embedded in SCRAPE_PROXY.
#   SCRAPE_PROXY_PASS
#   SCRAPE_PROXY_LIST     Alternatively, a comma-separated list of full proxy URLs
#                         to rotate through round-robin on our side (use when you
#                         hold a list of individual IPs rather than one gateway).
SCRAPE_PROXY = os.getenv("SCRAPE_PROXY", "").strip()
SCRAPE_PROXY_USER = os.getenv("SCRAPE_PROXY_USER", "").strip()
SCRAPE_PROXY_PASS = os.getenv("SCRAPE_PROXY_PASS", "").strip()
SCRAPE_PROXY_LIST = os.getenv("SCRAPE_PROXY_LIST", "").strip()


def _build_proxy_config() -> ProxyConfig | None:
    """Single-gateway proxy for BrowserConfig, or None when unconfigured.

    A rotating/backconnect proxy exposes one endpoint whose exit IP the provider
    rotates per request, so we just hand Playwright that endpoint via the browser.
    """
    if not SCRAPE_PROXY:
        return None
    kwargs = {"server": SCRAPE_PROXY}
    if SCRAPE_PROXY_USER:
        kwargs["username"] = SCRAPE_PROXY_USER
    if SCRAPE_PROXY_PASS:
        kwargs["password"] = SCRAPE_PROXY_PASS
    return ProxyConfig(**kwargs)


def _build_proxy_rotation() -> RoundRobinProxyStrategy | None:
    """Round-robin strategy over SCRAPE_PROXY_LIST, or None when unconfigured.

    Use this when you own a list of individual proxies instead of one gateway —
    crawl4ai cycles a different one into each request via CrawlerRunConfig.
    """
    if not SCRAPE_PROXY_LIST:
        return None
    proxies = [ProxyConfig(server=p.strip()) for p in SCRAPE_PROXY_LIST.split(",") if p.strip()]
    if not proxies:
        return None
    return RoundRobinProxyStrategy(proxies)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STRUCTURED_OUTPUT_DIR, exist_ok=True)


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
    if not os.path.isdir(OUTPUT_DIR):
        return 0

    removed = 0
    for entry in os.scandir(OUTPUT_DIR):
        if entry.is_file() and entry.name.endswith(".json"):
            os.remove(entry.path)
            removed += 1

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

    unique_urls = list(set(urls))
    logger.info(
        f"📄 Loaded {len(urls)} links from file ({len(unique_urls)} unique URLs after deduplication)."
    )
    return unique_urls


# Every entry in btu_subject_links.txt is a plain module URL, and the module
# number is already unique across all of them — so the structured corpus can use
# it directly instead of a hash nobody can reproduce by hand.
MODULE_URL_RE = re.compile(r"^https?://(?:www\.)?b-tu\.de/modul/(\d+)/?$")


def structured_filename(url: str) -> str:
    """Filename for a module's structured JSON: `modul_<number>.json`.

    Deliberately *not* url_to_safe_filename(): that prefixes an md5 slice, so
    finding module 13846 means grepping the directory instead of just opening
    `structured_data/modul_13846.json`. Module numbers are unique across the
    whole link list, so the hash buys nothing here.

    Any URL that isn't the standard module shape falls back to the hashed name,
    which keeps this safe if the link list ever grows other kinds of pages.
    """
    match = MODULE_URL_RE.match(url.strip())
    if match:
        return f"modul_{match.group(1)}.json"
    return url_to_safe_filename(url)


def url_to_safe_filename(url: str) -> str:
    """
    Creates a crash-proof filename using a short hash + path snippet.
    Prevents OS 'File name too long' errors on deep university URLs.
    """
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
    parsed = urlparse(url)
    clean_path = parsed.path.strip("/").replace("/", "_").replace(".", "_")

    # Keep filename readable by taking the last 80 chars of the path, prefixed by hash
    snippet = clean_path[-80:] if clean_path else "home"
    return f"{url_hash}_{snippet}.json"


# Module pages end with an "Events in the current semester" (Veranstaltungen im
# aktuellen Semester) block that links to individual course-event pages on the
# qisserver3 timetable system. Those sub-pages carry lecturer, room and schedule
# detail missing from the module overview, so we follow them one level deep and
# fold their content into the parent module's markdown. The `veranstid` query
# param uniquely identifies such an event link and never appears elsewhere.
VERANSTALTUNG_URL_RE = re.compile(
    r"https://www\.b-tu\.de/qisserver3/rds\?[^\s)\"'>]*veranstid=\d+"
)


def extract_nested_event_urls(markdown: str) -> list[str]:
    """Return the de-duplicated event sub-page URLs found in a module's markdown,
    preserving the order they appear in the page."""
    if not markdown:
        return []
    ordered: dict[str, None] = {}
    for url in VERANSTALTUNG_URL_RE.findall(markdown):
        ordered.setdefault(url, None)
    return list(ordered.keys())


def _clean_markdown(result) -> str:
    """Pull the pruned markdown out of a crawl result (falls back to raw)."""
    return getattr(result.markdown, "fit_markdown", result.markdown)


def _raw_markdown(result) -> str:
    """Return the unfiltered markdown of a crawl result.

    The PruningContentFilter that gives clean module overviews mis-handles the
    qisserver3 event pages — it discards their schedule / lecturer / description
    tables as 'low value' and leaves only navigation. Those pages are exactly the
    detail we're following the links for, so we keep their full markdown instead.
    """
    return getattr(result.markdown, "raw_markdown", None) or str(result.markdown)


async def _scrape_nested_event_pages(
    crawler: AsyncWebCrawler, urls: list[str], run_cfg: CrawlerRunConfig
) -> dict[str, dict]:
    """Scrape every event sub-page URL once and return a url -> {title, markdown}
    map. De-duplicating across the whole batch means a course event linked from
    two modules is only fetched a single time."""
    if not urls:
        return {}

    logger.info(f"  🔗 Scraping {len(urls)} nested event sub-page(s) for this batch...")
    results = await crawler.arun_many(urls=urls, config=run_cfg)

    nested_map: dict[str, dict] = {}
    for result in results:
        if result.success:
            nested_map[result.url] = {
                "title": result.metadata.get("title", "Event"),
                "markdown": _raw_markdown(result),
            }
        else:
            logger.info(
                f"    ⚠️ Nested failed: {result.url} | Error: {result.error_message}"
            )
    return nested_map


def _build_browser_config() -> BrowserConfig:
    """Browser settings shared by the markdown and structured crawls.

    text_mode + light_mode strip images, GPU and other heavyweight browser
    features we don't need — a big memory/bandwidth cut. The extra_args matter
    inside containers: the default /dev/shm is tiny (64MB) so
    --disable-dev-shm-usage forces Chromium to use /tmp instead of crashing, and
    --disable-gpu / --no-sandbox trim more resident memory.
    """
    proxy_config = _build_proxy_config()
    if proxy_config:
        logger.info(f"🌐 Routing through gateway proxy: {SCRAPE_PROXY}")

    return BrowserConfig(
        headless=True,
        verbose=False,
        text_mode=True,
        light_mode=True,
        user_agent=USER_AGENT,
        proxy_config=proxy_config,
        extra_args=[
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-extensions",
        ],
    )


def _build_run_config() -> CrawlerRunConfig:
    """Per-request settings shared by both crawls.

    The markdown generator is harmless for the structured crawl (which reads
    result.html instead), so one config serves both and the anti-bot tuning below
    stays defined in exactly one place.
    """
    proxy_rotation = _build_proxy_rotation()
    if proxy_rotation:
        proxy_count = len([p for p in SCRAPE_PROXY_LIST.split(",") if p.strip()])
        logger.info(f"🌐 Rotating through {proxy_count} proxies (round-robin).")

    md_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.4, threshold_type="fixed")
    )
    return CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=md_generator,
        magic=True,  # Enables stealth & anti-bot evasion
        # b-tu.de stalls flagged requests instead of refusing them, so cap the
        # per-page wait and let the retry pass reclaim stragglers rather than
        # burning 60s each. "domcontentloaded" is enough — the module content is
        # server-rendered, we don't need to wait for every deferred asset.
        page_timeout=45000,
        wait_until="domcontentloaded",
        # Jitter each tab's start so 3 requests don't arrive as a synchronized
        # burst — the pattern most anti-bot heuristics key on.
        mean_delay=1.0,
        max_range=2.0,
        semaphore_count=BATCH_SIZE,
        # Only set when SCRAPE_PROXY_LIST is used; None is ignored otherwise.
        proxy_rotation_strategy=proxy_rotation,
    )


async def scrape_batch(
    crawler: AsyncWebCrawler, batch_urls: list[str], run_cfg: CrawlerRunConfig
):
    """Scrapes a batch of URLs concurrently, then follows each page's
    'Events in the current semester' links and folds those sub-pages into the
    parent module's content_markdown.

    Returns (success_count, failed_urls) so the caller can retry the URLs that
    timed out or were blocked in a later, more spaced-out pass.
    """
    results = await crawler.arun_many(urls=batch_urls, config=run_cfg)

    # Pass 1: keep the successful pages and gather the union of their event links
    # so we can scrape all sub-pages for the batch in a single second pass.
    pages = []
    failed_urls: list[str] = []
    nested_union: dict[str, None] = {}
    for result in results:
        if not result.success:
            logger.info(f"  ⚠️ Failed: {result.url} | Error: {result.error_message}")
            failed_urls.append(result.url)
            continue
        clean_md = _clean_markdown(result)
        nested_urls = extract_nested_event_urls(clean_md)
        for url in nested_urls:
            nested_union.setdefault(url, None)
        pages.append(
            {"result": result, "clean_md": clean_md, "nested_urls": nested_urls}
        )

    # Pass 2: fetch every referenced event sub-page once.
    nested_map = await _scrape_nested_event_pages(
        crawler, list(nested_union.keys()), run_cfg
    )

    # Pass 3: write each module, appending the markdown of its event sub-pages.
    success_count = 0
    for page in pages:
        result = page["result"]
        url = result.url
        clean_md = page["clean_md"]

        sections = []
        appended_pages = []
        for nested_url in page["nested_urls"]:
            nested = nested_map.get(nested_url)
            if not nested:
                continue
            sections.append(
                f"\n\n---\n\n## Event sub-page: {nested['title']}\n"
                f"Source: {nested_url}\n\n{nested['markdown']}"
            )
            appended_pages.append(nested_url)

        full_md = clean_md + "".join(sections)

        filename = url_to_safe_filename(url)
        file_path = os.path.join(OUTPUT_DIR, filename)

        payload = {
            "doc_id": filename.replace(".json", ""),
            "metadata": {
                "url": url,
                "title": result.metadata.get("title", "Untitled Page"),
                "status_code": result.status_code,
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "word_count": len(full_md.split()) if full_md else 0,
                "nested_event_pages": appended_pages,
            },
            "content_markdown": full_md,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        success_count += 1

    return success_count, failed_urls


async def scrape():
    all_urls = load_and_clean_urls(INPUT_FILE)

    # 1. Checkpoint / Resume Logic: Filter out URLs that are already downloaded
    existing_files = set(os.listdir(OUTPUT_DIR))
    urls_to_scrape = []

    for url in all_urls:
        if url_to_safe_filename(url) not in existing_files:
            urls_to_scrape.append(url)

    skipped_count = len(all_urls) - len(urls_to_scrape)
    if skipped_count > 0:
        logger.info(
            f"⏩ Resuming job: Skipped {skipped_count} already scraped pages. {len(urls_to_scrape)} remaining."
        )

    if not urls_to_scrape:
        logger.info(
            "🎉 All pages have already been scraped! Check your output directory."
        )
        return

    # 2. Configure Browser & Crawler
    browser_cfg = _build_browser_config()
    run_cfg = _build_run_config()

    # 3. Execute in Batches
    logger.info(f"🚀 Starting parallel crawl with batch size = {BATCH_SIZE}...")
    start_time = asyncio.get_event_loop().time()
    total_scraped = 0

    failed_urls: list[str] = []
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        for i in range(0, len(urls_to_scrape), BATCH_SIZE):
            batch = urls_to_scrape[i : i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (len(urls_to_scrape) + BATCH_SIZE - 1) // BATCH_SIZE

            logger.info(
                f"\n📦 Processing Batch {batch_num}/{total_batches} ({len(batch)} URLs)..."
            )
            successes, batch_failed = await scrape_batch(crawler, batch, run_cfg)
            total_scraped += successes
            failed_urls.extend(batch_failed)
            logger.info(
                f"✅ Batch {batch_num} complete. Saved {successes}/{len(batch)} pages."
            )

            # Brief 2-second sleep between batches to let university servers breathe
            if batch_num < total_batches:
                await asyncio.sleep(2)

        # Retry pass: re-attempt everything that timed out or was blocked. Anti-bot
        # stalls are intermittent, so a page that failed under load often succeeds
        # on a quieter, more spaced-out retry. Each round waits progressively
        # longer to let any rate-limit window reset.
        for attempt in range(1, MAX_RETRIES + 1):
            if not failed_urls:
                break

            backoff = 5 * attempt
            logger.info(
                f"\n🔁 Retry {attempt}/{MAX_RETRIES}: re-attempting "
                f"{len(failed_urls)} failed URL(s) after a {backoff}s cooldown..."
            )
            await asyncio.sleep(backoff)

            retry_queue, failed_urls = failed_urls, []
            for i in range(0, len(retry_queue), BATCH_SIZE):
                batch = retry_queue[i : i + BATCH_SIZE]
                successes, batch_failed = await scrape_batch(crawler, batch, run_cfg)
                total_scraped += successes
                failed_urls.extend(batch_failed)
                await asyncio.sleep(2)

    if failed_urls:
        logger.warning(
            f"⚠️ {len(failed_urls)} URL(s) still failed after {MAX_RETRIES} "
            f"retries and were skipped this run (they'll be retried next crawl)."
        )

    elapsed = round(asyncio.get_event_loop().time() - start_time, 2)
    logger.info(
        f"\n🏁 Crawl finished in {elapsed} seconds! Successfully scraped {total_scraped} new pages."
    )


# --- STRUCTURED JSON CRAWL ---
#
# Same crawl mechanics as scrape() above (batching, proxies, retries), but each
# page's raw HTML is parsed into typed fields instead of markdown, and every
# module is written with its timetable sub-pages nested inside it.


def _event_key(url: str) -> str:
    """Stable identity for an event sub-page.

    Keyed on the `veranstid` rather than the full URL because qisserver3 rewrites
    session/tracking params on redirect: matching the returned URL against the
    requested one string-for-string silently loses pages that were fetched fine.
    """
    return extract_veranstid(url) or url


async def _fetch_event_pages(
    crawler: AsyncWebCrawler,
    urls: list[str],
    run_cfg: CrawlerRunConfig,
    event_cache: dict[str, dict],
) -> None:
    """Fetch and parse event sub-pages, filling `event_cache` in place.

    The cache is run-level, so a course event linked from twenty modules is
    fetched once for the whole crawl rather than once per batch. Failures are
    cached too — after MAX_RETRIES a page is treated as unavailable so a dead
    link can't be re-attempted by every module that references it.
    """
    pending = [u for u in urls if _event_key(u) not in event_cache]
    if not pending:
        return

    logger.info(f"  🔗 Fetching {len(pending)} new event sub-page(s)...")

    for attempt in range(MAX_RETRIES + 1):
        if not pending:
            break
        if attempt:
            # qisserver3 stalls under load exactly like the module pages do, so
            # give a failed sub-page the same spaced-out second chance.
            await asyncio.sleep(5 * attempt)
            logger.info(
                f"  🔁 Event retry {attempt}/{MAX_RETRIES} for {len(pending)} sub-page(s)..."
            )

        results = await crawler.arun_many(urls=pending, config=run_cfg)

        still_failing: list[str] = []
        returned: set[str] = set()
        for result in results:
            key = _event_key(result.url)
            returned.add(key)
            if not result.success:
                still_failing.append(result.url)
                continue
            parsed = parse_event_page(result.html, result.url)
            parsed["url"] = result.url
            parsed["status_code"] = result.status_code
            parsed["fetched"] = True
            parsed["fetch_error"] = None
            event_cache[key] = parsed

        # A URL that produced no result at all must still be retried, otherwise
        # it would silently vanish from this pass.
        still_failing.extend(u for u in pending if _event_key(u) not in returned)
        pending = [u for u in still_failing if _event_key(u) not in event_cache]

    # Whatever is left never came back; record it so the module that links to it
    # reports an honest failure instead of an empty section.
    for url in pending:
        event_cache[_event_key(url)] = {
            "url": url,
            "veranstid": extract_veranstid(url),
            "fetched": False,
            "fetch_error": f"Failed after {MAX_RETRIES} retries.",
            "status_code": None,
            "parsed": False,
        }
    if pending:
        logger.warning(f"  ⚠️ {len(pending)} event sub-page(s) unreachable this run.")


def _build_module_payload(result, parsed: dict, event_cache: dict[str, dict]) -> dict:
    """Assemble one module's structured record, sub-pages included.

    Everything the two page types carry ends up under a single `module` key, so a
    consumer never has to join two files to see a course's schedule.
    """
    filename = structured_filename(result.url)

    events = []
    for link in parsed["event_links"]:
        event = dict(event_cache.get(_event_key(link["url"]), {}))
        # How the parent module labelled this event ("430912 Vorlesung ... 2 SWS")
        # is context the sub-page itself doesn't repeat.
        event["listed_as"] = link["title"]
        event.setdefault("url", link["url"])
        event.setdefault("veranstid", link["veranstid"])
        event.setdefault("fetched", False)
        event.setdefault("fetch_error", "Sub-page was never fetched.")
        events.append(event)

    fields = dict(parsed["fields"])
    unmapped = parsed["unmapped_fields"]

    return {
        "doc_id": filename.replace(".json", ""),
        "source": {
            "url": result.url,
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
        "module": fields,
        # Non-empty only if b-tu.de adds a row we have no mapping for, which
        # keeps a layout change visible instead of silently dropping data.
        "unmapped_fields": unmapped,
        "events": events,
        "event_summary": {
            "linked": len(parsed["event_links"]),
            "fetched": sum(1 for e in events if e.get("fetched")),
            "failed": sum(1 for e in events if not e.get("fetched")),
        },
    }


async def _scrape_structured_batch(
    crawler: AsyncWebCrawler,
    batch_urls: list[str],
    run_cfg: CrawlerRunConfig,
    event_cache: dict[str, dict],
):
    """Crawl a batch of module pages, follow their events, write one file each.

    Returns (success_count, failed_urls) so the caller can retry pages that timed
    out or were blocked, mirroring scrape_batch().
    """
    results = await crawler.arun_many(urls=batch_urls, config=run_cfg)

    pages = []
    failed_urls: list[str] = []
    event_urls: dict[str, None] = {}
    for result in results:
        if not result.success:
            logger.info(f"  ⚠️ Failed: {result.url} | Error: {result.error_message}")
            failed_urls.append(result.url)
            continue
        parsed = parse_module_page(result.html, result.url)
        pages.append((result, parsed))
        for link in parsed["event_links"]:
            event_urls.setdefault(link["url"], None)

    await _fetch_event_pages(crawler, list(event_urls), run_cfg, event_cache)

    success_count = 0
    for result, parsed in pages:
        payload = _build_module_payload(result, parsed, event_cache)
        file_path = os.path.join(STRUCTURED_OUTPUT_DIR, structured_filename(result.url))
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        success_count += 1

    return success_count, failed_urls


async def scrape_structured():
    """Crawl every module page into structured JSON, sub-pages nested inside.

    Writes one file per module to STRUCTURED_OUTPUT_DIR. Like scrape(), it
    resumes: URLs whose file already exists are skipped, so a crash partway
    through a ~4,800-page crawl doesn't cost you the pages already done. Call
    clear_structured_data() first for a guaranteed-fresh full crawl.
    """
    all_urls = load_and_clean_urls(INPUT_FILE)

    # 1. Checkpoint / Resume: skip modules already written this cycle.
    existing_files = set(os.listdir(STRUCTURED_OUTPUT_DIR))
    urls_to_scrape = [
        url for url in all_urls if structured_filename(url) not in existing_files
    ]

    skipped_count = len(all_urls) - len(urls_to_scrape)
    if skipped_count > 0:
        logger.info(
            f"⏩ Resuming job: Skipped {skipped_count} already structured pages. "
            f"{len(urls_to_scrape)} remaining."
        )

    if not urls_to_scrape:
        logger.info("🎉 All pages have already been converted to structured JSON!")
        return

    browser_cfg = _build_browser_config()
    run_cfg = _build_run_config()

    logger.info(f"🚀 Starting structured crawl with batch size = {BATCH_SIZE}...")
    start_time = asyncio.get_event_loop().time()
    total_scraped = 0

    # Shared across the whole run so each event sub-page is fetched exactly once,
    # even when several modules link to the same course.
    event_cache: dict[str, dict] = {}
    failed_urls: list[str] = []

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        for i in range(0, len(urls_to_scrape), BATCH_SIZE):
            batch = urls_to_scrape[i : i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (len(urls_to_scrape) + BATCH_SIZE - 1) // BATCH_SIZE

            logger.info(
                f"\n📦 Structuring Batch {batch_num}/{total_batches} ({len(batch)} URLs)..."
            )
            successes, batch_failed = await _scrape_structured_batch(
                crawler, batch, run_cfg, event_cache
            )
            total_scraped += successes
            failed_urls.extend(batch_failed)
            logger.info(
                f"✅ Batch {batch_num} complete. Saved {successes}/{len(batch)} modules."
            )

            if batch_num < total_batches:
                await asyncio.sleep(2)

        # Retry pass for module pages, same rationale as scrape(): anti-bot
        # stalls are intermittent, so a quieter retry often succeeds.
        for attempt in range(1, MAX_RETRIES + 1):
            if not failed_urls:
                break

            backoff = 5 * attempt
            logger.info(
                f"\n🔁 Retry {attempt}/{MAX_RETRIES}: re-attempting "
                f"{len(failed_urls)} failed URL(s) after a {backoff}s cooldown..."
            )
            await asyncio.sleep(backoff)

            retry_queue, failed_urls = failed_urls, []
            for i in range(0, len(retry_queue), BATCH_SIZE):
                batch = retry_queue[i : i + BATCH_SIZE]
                successes, batch_failed = await _scrape_structured_batch(
                    crawler, batch, run_cfg, event_cache
                )
                total_scraped += successes
                failed_urls.extend(batch_failed)
                await asyncio.sleep(2)

    if failed_urls:
        logger.warning(
            f"⚠️ {len(failed_urls)} URL(s) still failed after {MAX_RETRIES} "
            f"retries and were skipped this run (they'll be retried next crawl)."
        )

    events_ok = sum(1 for e in event_cache.values() if e.get("fetched"))
    elapsed = round(asyncio.get_event_loop().time() - start_time, 2)
    logger.info(
        f"\n🏁 Structured crawl finished in {elapsed} seconds! "
        f"Wrote {total_scraped} modules and {events_ok}/{len(event_cache)} event sub-pages."
    )

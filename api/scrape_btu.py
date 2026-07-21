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

# Configure the logging format and level
logger = logging.getLogger(__name__)

# Configuration
INPUT_FILE = "btu_subject_links.txt"
OUTPUT_DIR = "scraped_data"

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
    # text_mode + light_mode strip images, GPU and other heavyweight browser
    # features we don't need for markdown extraction — big memory/bandwidth cut.
    # The extra_args matter inside containers: the default /dev/shm is tiny (64MB)
    # so --disable-dev-shm-usage forces Chromium to use /tmp instead of crashing,
    # and --disable-gpu / --no-sandbox trim more resident memory.
    proxy_config = _build_proxy_config()
    proxy_rotation = _build_proxy_rotation()
    if proxy_config:
        logger.info(f"🌐 Routing through gateway proxy: {SCRAPE_PROXY}")
    elif proxy_rotation:
        proxy_count = len([p for p in SCRAPE_PROXY_LIST.split(",") if p.strip()])
        logger.info(f"🌐 Rotating through {proxy_count} proxies (round-robin).")

    browser_cfg = BrowserConfig(
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
    md_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.4, threshold_type="fixed")
    )
    run_cfg = CrawlerRunConfig(
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

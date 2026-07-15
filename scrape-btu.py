import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

# Configuration
INPUT_FILE = "btu_subject_links.txt"
OUTPUT_DIR = "scraped_data"
BATCH_SIZE = 30  # Number of concurrent tabs. Reduce to 15 if your CPU/RAM spikes.

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_clean_urls(filepath: str) -> list[str]:
    """Reads links from txt file, removes duplicates and whitespace."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Input file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        # Strip whitespace, ignore empty lines and comment lines
        urls = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    unique_urls = list(set(urls))
    print(
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


async def scrape_batch(
    crawler: AsyncWebCrawler, batch_urls: list[str], run_cfg: CrawlerRunConfig
):
    """Scrapes a batch of URLs concurrently using Crawl4AI's arun_many."""
    results = await crawler.arun_many(urls=batch_urls, config=run_cfg)

    success_count = 0
    for result in results:
        if result.success:
            url = result.url
            filename = url_to_safe_filename(url)
            file_path = os.path.join(OUTPUT_DIR, filename)

            # Extract cleaned markdown
            clean_md = getattr(result.markdown, "fit_markdown", result.markdown)

            payload = {
                "doc_id": filename.replace(".json", ""),
                "metadata": {
                    "url": url,
                    "title": result.metadata.get("title", "Untitled Page"),
                    "status_code": result.status_code,
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                    "word_count": len(clean_md.split()) if clean_md else 0,
                },
                "content_markdown": clean_md,
            }

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            success_count += 1
        else:
            print(f"  ⚠️ Failed: {result.url} | Error: {result.error_message}")

    return success_count


async def main():
    all_urls = load_and_clean_urls(INPUT_FILE)

    # 1. Checkpoint / Resume Logic: Filter out URLs that are already downloaded
    existing_files = set(os.listdir(OUTPUT_DIR))
    urls_to_scrape = []

    for url in all_urls:
        if url_to_safe_filename(url) not in existing_files:
            urls_to_scrape.append(url)

    skipped_count = len(all_urls) - len(urls_to_scrape)
    if skipped_count > 0:
        print(
            f"⏩ Resuming job: Skipped {skipped_count} already scraped pages. {len(urls_to_scrape)} remaining."
        )

    if not urls_to_scrape:
        print("🎉 All pages have already been scraped! Check your output directory.")
        return

    # 2. Configure Browser & Crawler
    browser_cfg = BrowserConfig(headless=True, verbose=False)
    md_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.4, threshold_type="fixed")
    )
    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=md_generator,
        magic=True,  # Enables stealth & anti-bot evasion
    )

    # 3. Execute in Batches
    print(f"🚀 Starting parallel crawl with batch size = {BATCH_SIZE}...")
    start_time = asyncio.get_event_loop().time()
    total_scraped = 0

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        for i in range(0, len(urls_to_scrape), BATCH_SIZE):
            batch = urls_to_scrape[i : i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (len(urls_to_scrape) + BATCH_SIZE - 1) // BATCH_SIZE

            print(
                f"\n📦 Processing Batch {batch_num}/{total_batches} ({len(batch)} URLs)..."
            )
            successes = await scrape_batch(crawler, batch, run_cfg)
            total_scraped += successes
            print(
                f"✅ Batch {batch_num} complete. Saved {successes}/{len(batch)} pages."
            )

            # Brief 2-second sleep between batches to let university servers breathe
            if batch_num < total_batches:
                await asyncio.sleep(2)

    elapsed = round(asyncio.get_event_loop().time() - start_time, 2)
    print(
        f"\n🏁 Crawl finished in {elapsed} seconds! Successfully scraped {total_scraped} new pages."
    )


if __name__ == "__main__":
    asyncio.run(main())

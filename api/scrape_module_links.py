"""Rebuild ``btu_subject_links.txt`` from BTU's qisserver3 module catalogue.

Every other crawler in this API *consumes* that file — ``scrape_btu.scrape()``
and ``scrape_btu.scrape_structured()`` both start from it — so this is where the
list of what to crawl comes from.

The source is one qisserver3 search result, asked for in a single page via
``P_start=0&P_anzahl=9999``. It answers with a ~27MB document holding one row
per module: the module number, its title, and a link to that module's
description keyed by ``pord.pordnr``. Those two identifiers are unrelated —
module 11101 is ``pordnr=6951`` — and only this page knows the mapping, which is
why the crawlers key their files on the ``pordnr`` that is actually in the URL.

This replaced an earlier source, the ``https://www.b-tu.de/modul`` index. That
page listed ~4,900 modules against this one's ~3,200 because it also carried
modules marked "nicht mehr im Angebot" / "no longer offered". qisserver3 lists
the *currently offered* catalogue, which is what the knowledge base should
answer about.

The rebuild **replaces** the file rather than merging into it: a module the
university retires has to leave the list. Two things keep that safe — the write
is atomic, and an implausibly short scrape is refused rather than written. See
``MIN_EXPECTED_LINKS``.
"""

import asyncio
import logging
import os
import re
import tempfile
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://www.b-tu.de"

# The whole catalogue in one response. P_anzahl is the page size: 9999 is
# comfortably above the ~3,200 modules on offer, and the page prints its own
# "<n> Treffer" count, which _check_result_count() reads back to confirm we
# really did get everything rather than a silently truncated first page.
CATALOG_URL = os.getenv(
    "MODULE_CATALOG_URL",
    f"{BASE_URL}/qisserver3/rds?state=change&type=3&moduleParameter=pordpos"
    "&nextdir=change&next=TableSelectModul.vm&subdir=pord&P_start=0&P_anzahl=9999",
)

# Must stay in step with scrape_btu.INPUT_FILE — that is the file this writes.
LINKS_FILE = os.getenv("MODULE_LINKS_FILE", "btu_subject_links.txt")

# Refuse to overwrite the list when a run yields fewer than this many links.
# qisserver3 answers a flagged or half-built request with a 200 that carries no
# result table, so "fetch succeeded" is not on its own evidence that the scrape
# worked — and silently truncating the input to every other crawler is far more
# expensive than a run that refuses to write. The real table holds ~3,200 rows,
# so 1,000 flags a broken parse without tripping over semester churn.
MIN_EXPECTED_LINKS = int(os.getenv("MODULE_LINKS_MIN", "1000"))

MAX_RETRIES = int(os.getenv("MODULE_LINKS_MAX_RETRIES", "3"))
# Generous: the response is ~27MB, so this is a transfer budget, not a think time.
REQUEST_TIMEOUT = float(os.getenv("MODULE_LINKS_TIMEOUT", "180"))

USER_AGENT = os.getenv(
    "SCRAPE_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
)

# Reuses scrape_btu's / scrape_professors' proxy vars so one setting steers all
# three crawlers.
SCRAPE_PROXY = os.getenv("SCRAPE_PROXY", "").strip()
SCRAPE_PROXY_USER = os.getenv("SCRAPE_PROXY_USER", "").strip()
SCRAPE_PROXY_PASS = os.getenv("SCRAPE_PROXY_PASS", "").strip()

# The one table of results. qisserver3 gives it a stable `summary` attribute,
# which is a far better handle than position — the page has other tables for
# layout. If it is ever renamed we fall back to the whole document, which the
# per-link pattern below still filters correctly.
RESULT_TABLE_SUMMARY = "Suchergebnis"

# A module description link. Matching on the state plus pordnr rather than the
# whole query string keeps this working when qisserver3 reorders or adds
# parameters, which it does between releases.
MODULE_LINK_STATE = "modulBeschrDetailInfo"
PORDNR_RE = re.compile(r"[?&]pord\.pordnr=(\d+)")

# "<n> Treffer" — the result count the page prints above the table.
TREFFER_RE = re.compile(r"(\d[\d.]*)\s*Treffer")


class ModuleLinksError(RuntimeError):
    """A rebuild that could not be trusted, so the file was left untouched."""


def _proxy_url() -> str | None:
    """The configured gateway proxy with credentials folded in, or None."""
    if not SCRAPE_PROXY:
        return None
    if SCRAPE_PROXY_USER and "@" not in SCRAPE_PROXY:
        scheme, _, rest = SCRAPE_PROXY.partition("://")
        return f"{scheme}://{SCRAPE_PROXY_USER}:{SCRAPE_PROXY_PASS}@{rest}"
    return SCRAPE_PROXY


def module_pordnr(url: str) -> str | None:
    """The ``pord.pordnr`` a module description URL is keyed by, or None.

    This is the module's identity everywhere downstream — ``scrape_btu`` names
    its files after it — because it is the only id the URL itself carries.
    """
    match = PORDNR_RE.search(url)
    return match.group(1) if match else None


def read_module_links(filepath: str = LINKS_FILE) -> list[str]:
    """The links currently on disk, in file order. Empty when there is no file.

    Deliberately tolerant of a missing file: the first-ever rebuild has nothing
    to diff against, and that is not an error.
    """
    if not os.path.isfile(filepath):
        return []

    with open(filepath, "r", encoding="utf-8") as handle:
        return [
            line.strip()
            for line in handle
            if line.strip() and not line.strip().startswith("#")
        ]


def parse_module_links(html: str, base_url: str = CATALOG_URL) -> list[str]:
    """Every module description URL in the result table, in page order.

    De-duplicated by ``pordnr`` rather than by URL string, so the same module
    reached through two slightly different query strings is listed once.

    The href is kept as qisserver3 serves it (absolutised, HTML-unescaped)
    rather than rebuilt from a template: the query string carries the module's
    language variant among other things, and letting the site dictate the URL
    means a parameter added in a future release keeps working on its own.
    """
    soup = BeautifulSoup(html, "lxml")

    table = soup.find("table", summary=RESULT_TABLE_SUMMARY)
    if table is None:
        logger.info(
            f"  ⚠️ No table[summary='{RESULT_TABLE_SUMMARY}'] on the page — "
            "falling back to scanning the whole document."
        )
        table = soup

    # dict rather than set: insertion order is the page order we want to keep.
    ordered: dict[str, str] = {}
    for anchor in table.select("a[href]"):
        # BeautifulSoup has already turned &amp; back into &.
        href = urljoin(base_url, anchor["href"].strip())
        if MODULE_LINK_STATE not in href:
            continue
        if urlsplit(href).netloc.lower() not in ("www.b-tu.de", "b-tu.de"):
            continue

        pordnr = module_pordnr(href)
        if pordnr:
            ordered.setdefault(pordnr, href)

    return list(ordered.values())


def _check_result_count(html: str, parsed: int) -> None:
    """Warn if the page's own "<n> Treffer" disagrees with what we parsed.

    A mismatch means the request was paginated after all, or the table holds
    rows we failed to read. Worth a loud log either way — but not a refusal,
    since the count is cosmetic text that could be reworded at any time.
    """
    match = TREFFER_RE.search(html)
    if not match:
        return
    reported = int(match.group(1).replace(".", ""))
    if reported != parsed:
        logger.info(
            f"  ⚠️ Page reports {reported} results but {parsed} link(s) parsed. "
            "The catalogue may be paginated — check P_anzahl."
        )


async def fetch_catalog_html(url: str = CATALOG_URL) -> str:
    """Fetch the catalogue page, retrying with backoff on 5xx/429/network error.

    Same intermittent-block reasoning as the other crawlers: b-tu.de stalls
    flagged requests rather than refusing them, so a spaced-out retry usually
    succeeds.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
    }

    last_error = None
    async with httpx.AsyncClient(
        headers=headers,
        timeout=REQUEST_TIMEOUT,
        follow_redirects=True,
        proxy=_proxy_url(),
    ) as client:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await client.get(url)
                if response.is_success:
                    return response.text
                last_error = f"HTTP {response.status_code}"
                # A 4xx that isn't rate limiting won't fix itself on a retry.
                if response.status_code < 500 and response.status_code != 429:
                    break
            except Exception as exc:  # network error, timeout, proxy failure
                last_error = f"{type(exc).__name__}: {exc}"

            if attempt < MAX_RETRIES:
                await asyncio.sleep(2 * attempt)

    raise ModuleLinksError(
        f"Could not fetch the module catalogue at {url} after "
        f"{MAX_RETRIES} attempt(s): {last_error}"
    )


def _write_links(urls: list[str], filepath: str = LINKS_FILE) -> None:
    """Replace `filepath` with `urls`, one per line, atomically.

    The temp file is created in the destination directory so ``os.replace`` is a
    same-filesystem rename — an atomic swap. A crash mid-write therefore costs
    the new list, never the old one, which matters because every other crawler
    in this API reads this file as its only input.
    """
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=directory,
        prefix=".btu_links_",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            handle.write("\n".join(urls) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, filepath)
    except BaseException:
        # Leave no stray temp file behind if the write or swap failed.
        if os.path.exists(handle.name):
            os.remove(handle.name)
        raise


async def rebuild_module_links(
    filepath: str = LINKS_FILE,
    min_links: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Scrape the catalogue and rewrite `filepath` to match it exactly.

    The file is replaced, not merged, so a module the university has retired
    leaves the list on the next run. Returns a report of what changed:
    ``added`` and ``removed`` hold the actual URLs, not just counts.

    `min_links` overrides the ``MIN_EXPECTED_LINKS`` safety floor for this run —
    pass 0 to write whatever was parsed. `dry_run=True` computes the same report
    without touching the file.
    """
    floor = MIN_EXPECTED_LINKS if min_links is None else min_links

    logger.info(f"📄 Fetching the module catalogue: {CATALOG_URL}")
    html = await fetch_catalog_html()
    scraped = parse_module_links(html)
    logger.info(f"🔗 Parsed {len(scraped)} module links from the catalogue.")
    _check_result_count(html, len(scraped))

    if len(scraped) < floor:
        raise ModuleLinksError(
            f"Only {len(scraped)} module link(s) parsed, below the safety floor "
            f"of {floor}. The catalogue page was probably an error or anti-bot "
            f"response rather than the real result table, so {filepath} was "
            f"left untouched. Pass min_links=0 to write anyway."
        )

    previous = read_module_links(filepath)
    before, after = set(previous), set(scraped)
    added = [url for url in scraped if url not in before]
    removed = [url for url in previous if url not in after]

    if not dry_run:
        _write_links(scraped, filepath)
        logger.info(
            f"✅ Rewrote {filepath}: {len(scraped)} links "
            f"(+{len(added)} new, -{len(removed)} retired)."
        )
    else:
        logger.info(
            f"🔍 Dry run: {filepath} unchanged "
            f"(would be {len(scraped)} links, +{len(added)}, -{len(removed)})."
        )

    return {
        "source_url": CATALOG_URL,
        "file": filepath,
        "written": not dry_run,
        "dry_run": dry_run,
        "total": len(scraped),
        "previous_total": len(previous),
        "added_count": len(added),
        "removed_count": len(removed),
        "unchanged_count": len(after & before),
        "added": added,
        "removed": removed,
    }


if __name__ == "__main__":
    # `python scrape_module_links.py` refreshes the list from the command line,
    # for when you want the file current without running the API.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    report = asyncio.run(rebuild_module_links())
    print(
        f"{report['total']} links -> {report['file']} "
        f"(+{report['added_count']}, -{report['removed_count']})"
    )

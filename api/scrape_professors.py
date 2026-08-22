"""Crawl every b-tu.de faculty -> chair -> team page into one professors JSON.

The crawl mirrors how the site is organised, four levels deep:

1. the six faculty overviews list the institutes and their chairs,
2. each chair's ``nav.top-bar`` names the designations under its "Team" item,
3. each designation is either a list page holding the whole group or a
   per-person page whose sidebar indexes its siblings,
4. every one of those pages is parsed for people and the results de-duplicated.

Unlike ``scrape_btu``, this uses plain HTTP rather than a headless browser: these
pages are server-rendered with no JavaScript, so Chromium would cost ~100x the
memory and time per page for identical HTML. The anti-bot mitigations that
matter here (a real user agent, bounded concurrency, jittered pacing, retry with
backoff, and the shared ``SCRAPE_PROXY*`` settings) all carry over.

The whole run ends in a single ``structured-professors/data.json``.
"""

import asyncio
import json
import logging
import os
import random
import re
import unicodedata
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import httpx

from btu_professors_parser import (
    NON_DESIGNATION_RE,
    looks_like_person_name,
    normalize_ws,
    parse_breadcrumb,
    parse_chair_nav,
    parse_faculty_page,
    parse_people_page,
    parse_subnav,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://www.b-tu.de"

# The six faculty overviews are the only hard-coded URLs; everything below them
# is discovered by following links, so a new chair or a renamed team page is
# picked up without a code change.
FACULTY_URLS = [
    f"{BASE_URL}/fakultaet{n}/fakultaet/institute-fachgebiete" for n in range(1, 7)
]

# One aggregated file, as opposed to the file-per-page layout the module crawl
# uses — the professor corpus is small enough to hold in memory and far more
# useful as a single document.
PROFESSORS_OUTPUT_DIR = os.getenv("PROFESSORS_OUTPUT_DIR", "structured-professors")
# data.json is exactly one flat dictionary: person key -> that person's details,
# every value a plain string, number or boolean. Run counters and the leftovers
# that belong to no single person go beside it rather than cluttering the roster.
PROFESSORS_OUTPUT_FILE = os.path.join(PROFESSORS_OUTPUT_DIR, "data.json")
PROFESSORS_META_FILE = os.path.join(PROFESSORS_OUTPUT_DIR, "metadata.json")

# Concurrent HTTP requests. Well above scrape_btu's browser-tab budget because an
# httpx request costs kilobytes rather than ~100MB, but still low enough to stay
# a polite guest on a university web server.
CONCURRENCY = int(os.getenv("PROFESSORS_CONCURRENCY", "8"))

# Same intermittent-block reasoning as scrape_btu: b-tu.de stalls flagged
# requests rather than refusing them, so a spaced-out retry usually succeeds.
MAX_RETRIES = int(os.getenv("PROFESSORS_MAX_RETRIES", "3"))
REQUEST_TIMEOUT = float(os.getenv("PROFESSORS_TIMEOUT", "30"))

# Jitter between requests, so a burst doesn't look synthetic.
MIN_DELAY = float(os.getenv("PROFESSORS_MIN_DELAY", "0.05"))
MAX_DELAY = float(os.getenv("PROFESSORS_MAX_DELAY", "0.25"))

USER_AGENT = os.getenv(
    "SCRAPE_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
)

# Reuses scrape_btu's proxy vars so both crawlers are steered by one setting.
SCRAPE_PROXY = os.getenv("SCRAPE_PROXY", "").strip()
SCRAPE_PROXY_USER = os.getenv("SCRAPE_PROXY_USER", "").strip()
SCRAPE_PROXY_PASS = os.getenv("SCRAPE_PROXY_PASS", "").strip()


def _proxy_url() -> str | None:
    """The configured gateway proxy with credentials folded in, or None."""
    if not SCRAPE_PROXY:
        return None
    if SCRAPE_PROXY_USER and "@" not in SCRAPE_PROXY:
        scheme, _, rest = SCRAPE_PROXY.partition("://")
        return f"{scheme}://{SCRAPE_PROXY_USER}:{SCRAPE_PROXY_PASS}@{rest}"
    return SCRAPE_PROXY


def _is_btu_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host in ("www.b-tu.de", "b-tu.de")


def clear_professors_data() -> int:
    """Delete the exported professors JSON. Returns how many files were removed."""
    removed = 0
    for path in (PROFESSORS_OUTPUT_FILE, PROFESSORS_META_FILE):
        if os.path.isfile(path):
            os.remove(path)
            removed += 1
    if removed:
        logger.info("🧹 Removed the previous professors export.")
    return removed


class Fetcher:
    """Bounded-concurrency HTTP client that caches and retries.

    The cache is what makes the four-level crawl affordable: chairs share team
    pages (several faculty rows deep-link into the same section) and a page's
    sidebar names siblings we may reach from more than one direction, so without
    it the same URL would be fetched repeatedly.
    """

    def __init__(self, client: httpx.AsyncClient, concurrency: int = CONCURRENCY):
        self._client = client
        self._semaphore = asyncio.Semaphore(concurrency)
        self._cache: dict[str, dict] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self.stats = {"requests": 0, "cache_hits": 0, "failures": 0}

    async def get(self, url: str) -> dict:
        """Fetch a URL once; concurrent callers for it share the single result."""
        if url in self._cache:
            self.stats["cache_hits"] += 1
            return self._cache[url]

        lock = self._locks.setdefault(url, asyncio.Lock())
        async with lock:
            if url in self._cache:
                self.stats["cache_hits"] += 1
                return self._cache[url]
            result = await self._fetch(url)
            self._cache[url] = result
            return result

    async def _fetch(self, url: str) -> dict:
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            async with self._semaphore:
                await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
                try:
                    self.stats["requests"] += 1
                    response = await self._client.get(url)
                    if response.status_code >= 500 or response.status_code == 429:
                        last_error = f"HTTP {response.status_code}"
                    else:
                        return {
                            "ok": response.is_success,
                            "url": str(response.url),
                            "status_code": response.status_code,
                            "html": response.text if response.is_success else "",
                            "error": None if response.is_success else f"HTTP {response.status_code}",
                        }
                except Exception as exc:  # network error, timeout, proxy failure
                    last_error = f"{type(exc).__name__}: {exc}"

            if attempt < MAX_RETRIES:
                await asyncio.sleep(2 * attempt)

        self.stats["failures"] += 1
        logger.info(f"  ⚠️ Failed after {MAX_RETRIES} attempts: {url} | {last_error}")
        return {"ok": False, "url": url, "status_code": None, "html": "", "error": last_error}


# Titles carry no identity, so they are dropped before comparing two spellings
# of the same name.
_TITLE_TOKENS = {
    "prof", "professor", "dr", "ing", "habil", "rer", "nat", "phil", "jur",
    "med", "pol", "dipl", "apl", "jun", "hon", "univ", "mult", "sc", "msc",
    "bsc", "ma", "ba", "phd", "eng", "des", "oec", "agr", "techn", "mont",
    "honorarprofessor", "honorarprofessorin", "juniorprofessor", "emeritus",
}


def _name_tokens(name: str) -> list[str]:
    """Identity-bearing words of a name, lowercased and stripped of diacritics."""
    # German transliteration first ("Köhler" -> "koehler"), so a name spelled
    # either way folds to the same token; only then strip any other diacritic.
    folded = (name or "").lower()
    for umlaut, plain in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        folded = folded.replace(umlaut, plain)
    folded = unicodedata.normalize("NFKD", folded)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return [
        token
        for token in re.split(r"[^a-z]+", folded)
        if len(token) > 2 and token not in _TITLE_TOKENS
    ]


def _is_same_person(head_name: str, person_name: str) -> bool:
    """Whether a faculty table's "Leitung" cell names this person.

    Matching on the surname — the last identity-bearing token — rather than any
    overlap, so "Prof. Dr. Klaus Müller" is not matched to "Klaus Zimmermann"
    on the shared first name.
    """
    head = _name_tokens(head_name)
    person = _name_tokens(person_name)
    if not head or not person:
        return False
    return head[-1] in person


# Team designations that mean "this page is about whoever runs the chair", used
# to attach a CV found on a page that names nobody.
LEADERSHIP_RE = re.compile(
    r"(professor|lehrstuhlinhaber|inhaber|leitung|leiter|fachgebietsleitung|"
    r"\bchair\b|head\s+of)",
    re.I,
)


def _personal_emails(record: dict) -> list[str]:
    """The record's addresses that are built from that person's own name.

    Contact boxes routinely print a shared mailbox — "fg-ess@b-tu.de", or the
    secretary's address for booking appointments — beside a professor's name.
    Such an address is a valid way to reach them but a terrible way to identify
    them: keyed on it, one person looks like two (one entry per mailbox), and
    two colleagues who both list it look like one.
    """
    tokens = [t for t in _name_tokens(record["name"]) if len(t) >= 3]
    personal = []
    for email in record["emails"]:
        local = email.split("@")[0]
        for token in tokens:
            stem = token[:4]
            if stem in local or local[: len(stem)] == stem:
                personal.append(email)
                break
    return personal


def _has_personal_email(record: dict) -> bool:
    """Whether the record carries at least one address built from its own name."""
    return bool(_personal_emails(record))


def _person_key(person: dict) -> tuple:
    """Identity used to merge duplicate listings of the same person.

    An address is the reliable identifier, since a chair may print someone under
    both "Mitarbeiter" and "Sekretariat" with the name spelled differently. Where
    no address is published the name falls back in, lowercased and stripped of
    punctuation so "Dr. rer. nat. Guy Foghem" and "Dr rer nat Guy Foghem" merge.
    """
    # Only an address built from the person's own name identifies them; keying on
    # a shared mailbox merges everyone at the chair who lists it into one person.
    personal = _personal_emails(person)
    if personal:
        return ("email", personal[0])
    # Titles are stripped so "Prof. Dr. Jacob Spallek" and the same person listed
    # as "Prof. Dr. Jacob Spallek MSc." are one record — otherwise the CV page and
    # the contact page produce two halves of the same person.
    tokens = _name_tokens(person["name"])
    return ("name", " ".join(tokens) if tokens else person["name"].casefold())


# Fields that merge as a de-duplicated union when one person is found twice.
_LIST_FIELDS = ("emails", "phones", "faxes", "mobiles", "address_lines",
                "other_lines", "links", "designations", "source_urls")
# Fields where the first non-empty value found wins.
_SCALAR_FIELDS = ("room", "building", "orcid", "section_heading",
                  "image_url", "image_alt", "image_caption")
# Fields where the fullest version wins: a chair often prints a one-line CV on
# the team listing and the full one on the person's own page.
_LONGEST_FIELDS = ("career", "office_hours")


def _merge_into(existing: dict, person: dict) -> None:
    """Fold one sighting of a person into the record already held for them."""
    for field in _LIST_FIELDS:
        for value in person.get(field) or []:
            if value not in existing[field]:
                existing[field].append(value)
    for field in _SCALAR_FIELDS:
        existing[field] = existing.get(field) or person.get(field)
    for field in _LONGEST_FIELDS:
        incoming = person.get(field) or ""
        if len(incoming) > len(existing.get(field) or ""):
            existing[field] = incoming or None
    # True if they head any of the chairs they appear under.
    existing["is_head_of_chair"] = bool(
        existing.get("is_head_of_chair") or person.get("is_head_of_chair")
    )
    # Prefer the longer spelling: "Prof. Dr. rer. nat. habil. Daniel Hauer"
    # carries the full title set that a bare "Daniel Hauer" listing drops.
    if len(person["name"]) > len(existing["name"]):
        existing["name"] = person["name"]


def _normalized(person: dict) -> dict:
    """One page's person dict in the shape the merge and the export expect."""
    record = dict(person)
    designation = record.pop("designation", None)
    record["designations"] = [designation] if designation else []
    record["source_urls"] = [record.pop("source_url")]
    for field in _LIST_FIELDS:
        record.setdefault(field, [])
    for field in _SCALAR_FIELDS + _LONGEST_FIELDS:
        record.setdefault(field, None)
    record.setdefault("is_head_of_chair", False)
    return record


def _merge_people(people: list[dict]) -> list[dict]:
    """Collapse duplicates within one chair, keeping every designation.

    The same person legitimately appears on a category list page and on their own
    detail page, so the merge takes the union of the contact fields rather than
    letting whichever page was crawled last win.
    """
    merged: dict[tuple, dict] = {}
    for person in people:
        key = _person_key(person)
        if key in merged:
            _merge_into(merged[key], _normalized(person))
        else:
            merged[key] = _normalized(person)
    return _fold_addressless(list(merged.values()))


def _fold_addressless(records: list[dict]) -> list[dict]:
    """Fold a record without its own address into the same person's proper one.

    A chair often splits one person across two pages — a contact block on the
    team listing and a CV on their own page that repeats the name but not the
    address. Keyed on email those are two records; keyed on name they are one, so
    a final pass reconciles them by name once the addresses are known. A record
    carrying only a shared mailbox counts as address-less for this purpose.
    """
    hosts: dict[tuple, dict] = {}
    for record in records:
        tokens = tuple(_name_tokens(record["name"]))
        if tokens and _has_personal_email(record):
            hosts.setdefault(tokens, record)

    kept = []
    for record in records:
        tokens = tuple(_name_tokens(record["name"]))
        host = hosts.get(tokens) if tokens else None
        if host is None or host is record or _has_personal_email(record):
            kept.append(record)
            continue
        _merge_into(host, record)
        # Roster records also carry where the person sits; a fold must keep both
        # placements or the person silently loses a chair.
        for affiliation in record.get("affiliations") or []:
            if affiliation not in host.setdefault("affiliations", []):
                host["affiliations"].append(affiliation)
    return kept


async def _collect_people_from_page(
    fetcher: Fetcher, url: str, designation: str | None
) -> tuple[list[dict], list[dict], dict]:
    """Parse one team page.

    Returns (people, sibling person pages to follow, page-level sections). The
    sections hold a CV or office-hours text the page carries without naming who
    it belongs to — the caller decides who, if anyone, that is.
    """
    page = await fetcher.get(url)
    if not page["ok"]:
        return [], [], {}

    parsed = parse_people_page(page["html"], page["url"])
    people = []
    for person in parsed["people"]:
        record = dict(person)
        # The heading above a block is the more specific role when the page
        # groups several ("Sekretariat", "Wissenschaftliche Mitarbeiter"), but a
        # page-section heading like "Kontakt" names no job, so those fall back to
        # the Team dropdown label instead.
        heading = person.get("section_heading")
        if heading and NON_DESIGNATION_RE.match(heading):
            heading = None
        record["designation"] = heading or designation
        people.append(record)

    # A page can carry a CV or office hours with no contact block at all — the
    # name is then only in the breadcrumb or the page heading. Recovering it
    # keeps the person (and their CV) in the export instead of dropping both.
    if not people and (parsed.get("career") or parsed.get("office_hours")):
        # Only the final crumb — the page's own title — can name its subject. An
        # ancestor crumb names the chair or the team section, never a person.
        crumbs = parse_breadcrumb(page["html"], page["url"])
        candidates = [crumbs[-1] if crumbs else None, parsed["page_heading"]]
        subject = next(
            (c for c in candidates if c and looks_like_person_name(c)), None
        )
        if subject:
            people.append(
                {
                    "name": subject,
                    "section_heading": None,
                    "designation": designation,
                    "emails": [], "phones": [], "faxes": [], "mobiles": [],
                    "room": None, "building": None, "address_lines": [],
                    "orcid": None, "links": [], "other_lines": [],
                    "image_url": None, "image_alt": None, "image_caption": None,
                    "career": parsed.get("career"),
                    "office_hours": parsed.get("office_hours"),
                    "source_url": page["url"],
                    "raw_text": "",
                }
            )
            parsed["career"] = parsed["office_hours"] = None

    subnav = parse_subnav(page["html"], page["url"])
    sections = {
        "url": page["url"],
        "designation": designation,
        "career": parsed.get("career"),
        "office_hours": parsed.get("office_hours"),
        "named_people": len(people),
    }
    return people, subnav["siblings"], sections


async def _scrape_chair(fetcher: Fetcher, chair: dict) -> dict:
    """Crawl one chair: its Team dropdown, every designation, every person page."""
    entry_url = chair["url"]
    record = {
        "name": chair["name"],
        "url": entry_url,
        "head_of_chair": chair["head_of_chair"],
        "kind": chair["kind"],
        "group": chair["group"],
        "former": chair["former"],
        "chair_title": None,
        "menu_labels": [],
        "team_menu_found": False,
        "designations": [],
        "people": [],
        "pages_crawled": [],
        # "crawled" | "no_link" | "external" | "fetch_failed". Split out from the
        # message so a consumer can tell an unlinked chair (nothing to fetch)
        # from one whose site actually failed.
        "status": "crawled",
        "error": None,
        # Filled in by _attach_head_of_chair once the people are known.
        "head_of_chair_resolved": False,
        "head_of_chair_person": None,
        "unattached_career": [],
        "unattached_office_hours": [],
    }

    if not entry_url:
        record["status"] = "no_link"
        record["error"] = "No link published for this chair."
        return record
    if chair["external"]:
        record["status"] = "external"
        record["error"] = "Chair site is hosted outside b-tu.de; not crawled."
        return record

    landing = await fetcher.get(entry_url)
    if not landing["ok"]:
        record["status"] = "fetch_failed"
        record["error"] = landing["error"]
        return record

    nav = parse_chair_nav(landing["html"], landing["url"])
    record["chair_title"] = nav["chair_title"] or chair["name"]
    record["menu_labels"] = nav["menu_labels"]

    # Which pages to mine for people. The chair's own landing page is always
    # included: the 12 chairs with no Team dropdown publish their contacts there,
    # and it costs nothing extra since the fetch is already cached.
    targets: list[tuple[str, str | None]] = [(landing["url"], None)]
    for section in nav["team_sections"]:
        record["team_menu_found"] = True
        if section["url"]:
            targets.append((section["url"], None))
        for entry in section["entries"]:
            if not _is_btu_url(entry["url"]):
                continue
            # A dropdown entry is a role only if it isn't a page about the
            # chair ("Anfahrt") and isn't a member listed by name.
            label = entry["designation"]
            is_role = entry["is_designation"] and not looks_like_person_name(label)
            targets.append((entry["url"], label if is_role else None))
            if is_role:
                record["designations"].append(label)

    # Pass 1: the category pages themselves. A chair's "Team" item usually links
    # straight to its own first entry, so the same URL arrives twice — once
    # unlabelled from the menu and once carrying the designation. Keeping the
    # labelled version is what preserves the role.
    labelled: dict[str, str | None] = {}
    for url, designation in targets:
        if url not in labelled or (labelled[url] is None and designation is not None):
            labelled[url] = designation
    queued = list(labelled.items())
    seen: set[str] = set(labelled)
    first_pass = await asyncio.gather(
        *[_collect_people_from_page(fetcher, u, d) for u, d in queued]
    )

    people: list[dict] = []
    follow: list[tuple[str, str | None]] = []
    loose: list[dict] = []
    for (url, designation), (found, siblings, sections) in zip(queued, first_pass):
        record["pages_crawled"].append(url)
        people.extend(found)
        if sections.get("career") or sections.get("office_hours"):
            loose.append(sections)
        for sibling in siblings:
            if sibling["url"] in seen or not _is_btu_url(sibling["url"]):
                continue
            seen.add(sibling["url"])
            follow.append((sibling["url"], designation))

    # Pass 2: the per-person pages named by the sidebars. One level is enough —
    # every member of a category appears in the same sidebar, so their pages are
    # all discovered from whichever page of that category we visited first.
    if follow:
        second_pass = await asyncio.gather(
            *[_collect_people_from_page(fetcher, u, d) for u, d in follow]
        )
        for (url, designation), (found, _, sections) in zip(follow, second_pass):
            record["pages_crawled"].append(url)
            people.extend(found)
            if sections.get("career") or sections.get("office_hours"):
                loose.append(sections)

    record["people"] = _merge_people(people)
    record["designations"] = sorted(set(record["designations"]))
    _attach_head_of_chair(record, loose)
    return record


def _attach_head_of_chair(record: dict, loose: list[dict]) -> None:
    """Resolve the chair's "Leitung" cell to a person and place loose sections.

    The faculty table names the head of chair but publishes no contact details;
    those live on the chair's own team pages. Matching the two lets the export
    answer "how do I reach the head of this chair" without a second lookup.

    A CV or office-hours block on a leadership page that names nobody describes
    the head, so it is attached to them; anything left over is kept on the chair
    rather than dropped.
    """
    head_name = (record["head_of_chair"] or "").strip()
    vacant = not head_name or head_name.replace(" ", "") in ("N.N.", "NN", "-", "—")

    head = None
    if not vacant:
        head = next(
            (p for p in record["people"] if _is_same_person(head_name, p["name"])), None
        )
        record["head_of_chair_resolved"] = head is not None
        if head is None:
            # The chair names a head the team pages never describe with a contact
            # block. Keep them as a person anyway, so a CV found on a leadership
            # page still has someone to belong to.
            head = {
                "name": head_name, "section_heading": None, "designations": [],
                "emails": [], "phones": [], "faxes": [], "mobiles": [],
                "room": None, "building": None, "address_lines": [], "orcid": None,
                "links": [], "other_lines": [], "image_url": None, "image_alt": None,
                "image_caption": None, "career": None, "office_hours": None,
                "source_urls": [], "raw_text": "", "is_head_of_chair": True,
            }
            record["people"].append(head)

    for section in loose:
        is_leadership = LEADERSHIP_RE.search(section.get("designation") or "")
        if head is not None and is_leadership and not section["named_people"]:
            for field in ("career", "office_hours"):
                value = section.get(field)
                if value and len(value) > len(head.get(field) or ""):
                    head[field] = value
            continue
        for field in ("career", "office_hours"):
            if section.get(field):
                note = {"source_url": section["url"], "text": section[field]}
                if note not in record[f"unattached_{field}"]:
                    record[f"unattached_{field}"].append(note)

    if head is not None:
        head["is_head_of_chair"] = True
        record["head_of_chair_person"] = {
            "name": head["name"],
            "emails": head["emails"],
            "phones": head["phones"],
            "faxes": head["faxes"],
            "mobiles": head["mobiles"],
            "room": head["room"],
            "building": head["building"],
            "address_lines": head["address_lines"],
            "office_hours": head.get("office_hours"),
            "image_url": head.get("image_url"),
            "career": head.get("career"),
            "orcid": head.get("orcid"),
            "designations": head["designations"],
            "source_urls": head["source_urls"],
        }


async def _scrape_faculty(fetcher: Fetcher, url: str) -> dict:
    """Crawl one faculty overview and every chair listed on it."""
    page = await fetcher.get(url)
    if not page["ok"]:
        logger.info(f"❌ Faculty page failed: {url} | {page['error']}")
        return {
            "faculty_url": url,
            "faculty_name": None,
            "faculty_number": None,
            "error": page["error"],
            "institutes": [],
        }

    overview = parse_faculty_page(page["html"], page["url"])
    logger.info(
        f"🏛️  {overview['faculty_name']} — {overview['chair_count']} chair(s) "
        f"across {len(overview['institutes'])} institute(s)."
    )

    # Chairs are crawled concurrently across the whole faculty; the Fetcher's
    # semaphore, not this fan-out, is what bounds the load on the server.
    tasks, owners = [], []
    for position, institute in enumerate(overview["institutes"]):
        for chair in institute["chairs"]:
            tasks.append(_scrape_chair(fetcher, chair))
            owners.append(position)

    results = await asyncio.gather(*tasks)

    institutes = [
        {"institute": i["institute"], "chairs": []} for i in overview["institutes"]
    ]
    for position, chair_record in zip(owners, results):
        institutes[position]["chairs"].append(chair_record)

    people_found = sum(len(c["people"]) for c in results)
    logger.info(f"   ✅ {overview['faculty_name']}: {people_found} people found.")

    return {
        "faculty_url": url,
        "faculty_name": overview["faculty_name"],
        "faculty_number": overview["faculty_number"],
        "error": None,
        "institutes": institutes,
    }


def _build_export(faculties: list[dict], stats: dict, started_at: str) -> dict:
    """Assemble the export: one flat record per person, plus the run's counters.

    The faculty/institute/chair tree is flattened away here — each person carries
    their placement as plain fields instead, so the published file is a single
    dictionary that can be indexed by person without walking anything.
    """
    # One entry per human. A person reachable from two chairs (or listed under
    # both a live chair and its "ehemalige Fachgebiete" row) is a single record
    # carrying both affiliations, rather than two near-identical rows.
    roster: dict[tuple, dict] = {}
    for faculty in faculties:
        for institute in faculty["institutes"]:
            for chair in institute["chairs"]:
                for person in chair["people"]:
                    affiliation = {
                        "faculty": faculty["faculty_name"],
                        "faculty_number": faculty["faculty_number"],
                        "institute": institute["institute"],
                        "chair": chair["chair_title"] or chair["name"],
                        "chair_url": chair["url"],
                        "chair_is_former": chair["former"],
                        "designations": person["designations"],
                        "is_head_of_chair": person.get("is_head_of_chair", False),
                    }
                    key = _person_key(person)
                    entry = roster.get(key)
                    if entry is None:
                        entry = dict(person)
                        entry["affiliations"] = []
                        roster[key] = entry
                    else:
                        _merge_into(entry, person)
                    if affiliation not in entry["affiliations"]:
                        entry["affiliations"].append(affiliation)

    # The same fold as within a chair, now across the whole university: someone
    # listed by name only under one chair and with an address under another is
    # one person, not two entries the reader has to reconcile.
    flat = _fold_addressless(list(roster.values()))
    # Finally, one entry per human: the university issues both "surname@" and
    # "firstname.surname@" to the same person, so two personal addresses for one
    # name are aliases, not two people. Both addresses survive the merge, so a
    # wrong join would be visible in the record rather than losing anything.
    flat = _merge_same_name(flat)

    chairs = [
        chair
        for faculty in faculties
        for institute in faculty["institutes"]
        for chair in institute["chairs"]
    ]

    return {
        "roster": _flatten_roster(flat),
        "metadata": {
            "source": BASE_URL,
            "faculty_urls": FACULTY_URLS,
            "scrape_started_at": started_at,
            "scrape_finished_at": datetime.now(timezone.utc).isoformat(),
            "faculties": len(faculties),
            "institutes": sum(len(f["institutes"]) for f in faculties),
            "chairs": len(chairs),
            "chairs_with_team_menu": sum(1 for c in chairs if c["team_menu_found"]),
            "chairs_crawled": sum(1 for c in chairs if c["status"] == "crawled"),
            "chairs_without_link": sum(1 for c in chairs if c["status"] == "no_link"),
            "chairs_hosted_externally": sum(1 for c in chairs if c["status"] == "external"),
            "chairs_fetch_failed": sum(1 for c in chairs if c["status"] == "fetch_failed"),
            "chairs_with_head_named": sum(1 for c in chairs if c["head_of_chair_person"]),
            "chairs_with_head_contactable": sum(
                1
                for c in chairs
                if c["head_of_chair_person"]
                and (
                    c["head_of_chair_person"]["emails"]
                    or c["head_of_chair_person"]["phones"]
                )
            ),
            "people": len(flat),
            "people_with_email": sum(1 for p in flat if p["emails"]),
            "people_with_image": sum(1 for p in flat if p["image_url"]),
            "people_with_career": sum(1 for p in flat if p["career"]),
            "people_with_office_hours": sum(1 for p in flat if p["office_hours"]),
            "pages_requested": stats["requests"],
            "pages_from_cache": stats["cache_hits"],
            "pages_failed": stats["failures"],
        },
        "unattached_sections": [
            {
                "chair": chair["chair_title"] or chair["name"],
                "chair_url": chair["url"],
                "kind": kind,
                "source_url": note["source_url"],
                "text": note["text"],
            }
            for chair in chairs
            for kind in ("career", "office_hours")
            for note in chair[f"unattached_{kind}"]
        ],
    }


def _merge_same_name(records: list[dict]) -> list[dict]:
    """Collapse records whose names reduce to the same identity-bearing tokens."""
    hosts: dict[tuple, dict] = {}
    kept: list[dict] = []
    for record in records:
        tokens = tuple(_name_tokens(record["name"]))
        host = hosts.get(tokens) if tokens else None
        if host is None:
            if tokens:
                hosts[tokens] = record
            kept.append(record)
            continue
        _merge_into(host, record)
        for affiliation in record.get("affiliations") or []:
            if affiliation not in host.setdefault("affiliations", []):
                host["affiliations"].append(affiliation)
    return kept


# One flat record per person: every value a scalar, repeated values joined, so
# the export is a single dictionary a consumer can index without walking a tree.
def _join(values: list, separator: str = "; ") -> str | None:
    """Collapse a list of values into one string, or None when there are none."""
    seen: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.append(text)
    return separator.join(seen) or None


def _slug(name: str) -> str:
    """A readable, stable key for a person."""
    folded = "-".join(_name_tokens(name))
    return folded or re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "unknown"


def _flatten_roster(people: list[dict]) -> dict:
    """Turn the roster into ``{person key: flat record}``.

    Keyed on a slug of the name rather than an email, because most of the people
    the university publishes — student assistants and alumni above all — have no
    address, and every person must appear exactly once.
    """
    roster: dict[str, dict] = {}
    for person in people:
        affiliations = person["affiliations"]
        record = {
            "name": person["name"],
            "email": _join(person["emails"]),
            "phone": _join(person["phones"]),
            "mobile": _join(person["mobiles"]),
            "fax": _join(person["faxes"]),
            "room": person["room"],
            "building": person["building"],
            "address": _join(person["address_lines"], ", "),
            "office_hours": person["office_hours"],
            "designation": _join(person["designations"]),
            "faculty": _join([a["faculty"] for a in affiliations]),
            "institute": _join([a["institute"] for a in affiliations]),
            "chair": _join([a["chair"] for a in affiliations]),
            "chair_url": _join([a["chair_url"] for a in affiliations]),
            "is_head_of_chair": person.get("is_head_of_chair", False),
            "is_former_chair": all(a["chair_is_former"] for a in affiliations),
            "image_url": person["image_url"],
            "image_caption": person["image_caption"],
            "orcid": person["orcid"],
            "career": person["career"],
            "website": _join([link["url"] for link in person["links"]]),
            "profile_url": _join(person["source_urls"]),
            "notes": _join(person["other_lines"]),
        }
        key = _slug(person["name"])
        if key in roster:
            # Two different people whose names reduce to the same slug: keep both
            # under distinct keys rather than letting one overwrite the other.
            suffix = 2
            while f"{key}-{suffix}" in roster:
                suffix += 1
            key = f"{key}-{suffix}"
        roster[key] = record
    return roster


async def scrape_professors() -> dict:
    """Run the whole crawl and write ``structured-professors/data.json``.

    Returns the metadata block so a caller (the API route, a script) can report
    what the run produced without re-reading the file.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    logger.info(
        f"🚀 Starting professor crawl over {len(FACULTY_URLS)} faculties "
        f"(concurrency = {CONCURRENCY})..."
    )

    proxy = _proxy_url()
    if proxy:
        logger.info(f"🌐 Routing through gateway proxy: {SCRAPE_PROXY}")

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
    }
    limits = httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY)

    async with httpx.AsyncClient(
        headers=headers,
        follow_redirects=True,
        timeout=REQUEST_TIMEOUT,
        limits=limits,
        proxy=proxy,
    ) as client:
        fetcher = Fetcher(client)
        faculties = []
        for url in FACULTY_URLS:
            faculties.append(await _scrape_faculty(fetcher, url))

        export = _build_export(faculties, fetcher.stats, started_at)

    os.makedirs(PROFESSORS_OUTPUT_DIR, exist_ok=True)
    with open(PROFESSORS_OUTPUT_FILE, "w", encoding="utf-8") as handle:
        json.dump(export["roster"], handle, indent=2, ensure_ascii=False)
    with open(PROFESSORS_META_FILE, "w", encoding="utf-8") as handle:
        json.dump(
            {"metadata": export["metadata"], "unattached_sections": export["unattached_sections"]},
            handle,
            indent=2,
            ensure_ascii=False,
        )

    meta = export["metadata"]
    logger.info(
        f"🎉 Professor crawl complete: {meta['people']} people from "
        f"{meta['chairs']} chairs across {meta['faculties']} faculties. "
        f"{meta['pages_requested']} pages fetched, {meta['pages_failed']} failed. "
        f"Written to {PROFESSORS_OUTPUT_FILE}"
    )
    return meta


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(scrape_professors())

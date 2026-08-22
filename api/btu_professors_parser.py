"""Deterministic HTML -> dict parsers for the b-tu.de faculty / chair / team pages.

Same contract as ``btu_parser``: nothing here guesses at layout. Every value is
read out of a specific element, and whatever we can't classify is still kept
verbatim (``other_lines`` / ``raw_text``) so a redesign on the university side
degrades into "unlabelled but present" rather than silent loss.

Four page types, matching the four crawl levels:

* **Faculty overview** (``/fakultaetN/fakultaet/institute-fachgebiete``) — ``h2``
  institute headings followed by two-column tables of
  ``chair -> head of chair``. The chair cell links to that chair's own site.
* **Chair site** (``/fg-<slug>``) — any page in the section carries the same
  ``nav.top-bar``. The "Team" item's dropdown lists the designations that chair
  publishes (Professor, Sekretariat, Mitarbeiter, ...).
* **Team category page** — either a *list* page holding every member inline, or a
  *per-person* page whose ``ul.side-nav`` sidebar links to the siblings. Some are
  both, so the crawler parses people from every page it visits and de-duplicates.
* **Person block** — a ``div.ce-bodytext`` where the name is a heading/``strong``/
  link and the ``<br>``-separated lines under it are the contact details.
"""

import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, NavigableString, Tag

# --- TEXT NORMALISATION ---

# The site occasionally serves half-encoded UTF-7 out of its mailto obfuscator,
# leaking "Beckumer Stra+AN8-e" (ß) and "T +-49" (a literal +) into the rendered
# page. Decoding only these two shapes fixes them without touching a legitimate
# "+49" elsewhere, which a blanket utf-7 decode would corrupt.
UTF7_SEQUENCE_RE = re.compile(r"\+([A-Za-z0-9+/]{2,})-")


def _decode_utf7_artifacts(text: str) -> str:
    """Repair stray UTF-7 escapes ("+AN8-" -> "ß", "+-" -> "+")."""
    if "+" not in text:
        return text

    def repl(match: re.Match) -> str:
        try:
            return match.group(0).encode("ascii").decode("utf-7")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return match.group(0)

    return UTF7_SEQUENCE_RE.sub(repl, text).replace("+-", "+")


def normalize_ws(text: str) -> str:
    """Collapse whitespace, including the non-breaking and soft-hyphen variants.

    b-tu.de sprinkles \\u00ad (soft hyphen) inside headings like "Sti­pen­di­aten"
    for justification; stripping it is what makes those strings comparable.
    """
    if not text:
        return ""
    text = text.replace("­", "").replace("​", "")
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", _decode_utf7_artifacts(text)).strip()


# --- EMAIL DE-OBFUSCATION ---

# TYPO3 hides addresses behind a Caesar cipher over three character ranges and
# ships the shift in `data-mailto-vector`. The attribute holds the *encryption*
# offset, so decoding negates it. Ranges mirror TYPO3's own JS decoder:
# 0x2B-0x3A ("+,-./0-9:"), 0x40-0x5A ("@A-Z"), 0x61-0x7A ("a-z").
_MAILTO_RANGES = ((0x2B, 0x3A), (0x40, 0x5A), (0x61, 0x7A))


def decode_mailto_token(token: str, vector: str | int) -> str | None:
    """Decode a TYPO3 ``data-mailto-token`` into a plain "mailto:..." string."""
    try:
        offset = -int(vector)
    except (TypeError, ValueError):
        return None

    out = []
    for char in token:
        code = ord(char)
        for start, end in _MAILTO_RANGES:
            if start <= code <= end:
                shifted = code + offset
                if offset > 0 and shifted > end:
                    shifted = start + (shifted - end - 1)
                elif offset < 0 and shifted < start:
                    shifted = end - (start - shifted - 1)
                out.append(chr(shifted))
                break
        else:
            out.append(char)
    return "".join(out)


# The same address is published three ways across the site — a real mailto:, a
# TYPO3 token, or spelled out as "name(at)b-tu.de" — so every form is normalised
# back to one canonical address.
_AT_TEXT_RE = re.compile(r"\s*[\(\[\{]\s*(?:at|@)\s*[\)\]\}]\s*", re.I)
_DOT_TEXT_RE = re.compile(r"\s*[\(\[\{]\s*(?:dot|punkt)\s*[\)\]\}]\s*", re.I)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def normalize_email(raw: str) -> str | None:
    """Turn any published form of an address into "user@host", or None."""
    if not raw:
        return None
    text = normalize_ws(raw)
    if text.lower().startswith("mailto:"):
        text = text[7:]
    text = text.split("?")[0]
    text = _AT_TEXT_RE.sub("@", text)
    text = _DOT_TEXT_RE.sub(".", text)
    match = EMAIL_RE.search(text.replace(" ", ""))
    if not match:
        return None
    # Chairs do publish malformed addresses — "stefan.glasauer.(at)b-tu.de" has a
    # trailing dot on the local part. Left alone it reads as a different address
    # from the same person's correctly spelled one, and splits them into two people.
    local, _, domain = match.group(0).lower().partition("@")
    local = re.sub(r"\.{2,}", ".", local).strip(".")
    domain = re.sub(r"\.{2,}", ".", domain).strip(".")
    return f"{local}@{domain}" if local and domain else None


def _link_email(anchor: Tag) -> str | None:
    """Pull an address off an anchor however that anchor happens to encode it."""
    token = anchor.get("data-mailto-token")
    if token:
        decoded = decode_mailto_token(token, anchor.get("data-mailto-vector", 0))
        if decoded:
            found = normalize_email(decoded)
            if found:
                return found
    href = anchor.get("href") or ""
    if href.lower().startswith("mailto:"):
        found = normalize_email(href)
        if found:
            return found
    return normalize_email(anchor.get_text(" ", strip=True))


# Some b-tu.de pages emit a stray "</html>" in the middle of the document —
# inside an accordion body, for instance. lxml honours it and silently discards
# everything after, which on the worst pages is 94% of the content (a
# chair-holder's contact block and CV both sat behind one). Dropping every
# closing document tag and letting the parser close the tree itself recovers it.
STRAY_DOC_END_RE = re.compile(r"</\s*(?:html|body)\s*>", re.I)


def make_soup(html: str) -> BeautifulSoup:
    """Parse a b-tu.de page, surviving the stray document-end tags it serves."""
    return BeautifulSoup(STRAY_DOC_END_RE.sub("", html), "lxml")


def content_blocks(soup: BeautifulSoup) -> list[Tag]:
    """Every top-level ``div.user-content`` region on a page.

    A page can carry more than one: the stray document-end tags split what the
    site authored as a single region into siblings, and on some pages the second
    block holds everything that matters — a professor's contact details and CV
    behind a 35-character first block naming only their role.

    ``aside.marginal`` is included because many chairs print the "Kontakt" box —
    often the only place an address appears — in the page margin rather than in
    the main column.
    """
    found = soup.select("div.user-content, aside.marginal")
    # Guard against a nested pair being counted twice.
    return [block for block in found if not any(p in found for p in block.parents)]


# --- FACULTY OVERVIEW PAGE ---


def parse_faculty_page(html: str, url: str) -> dict:
    """Read one faculty's institute -> chair listing.

    Returns the institute sections in document order, each holding the chair rows
    printed under it. ``head_of_chair`` is the raw "Leitung" cell — "N. N." when
    the position is vacant, which we keep rather than drop so the vacancy stays
    visible.
    """
    soup = make_soup(html)
    blocks = content_blocks(soup) or [soup]

    # The breadcrumb's first crumb is the faculty itself ("Fakultät 1"); the
    # <title> only ever repeats the page name, so it is the fallback.
    faculty_name = None
    crumb = soup.select_one("nav.breadcrumb-section li span[itemprop=name]")
    if crumb:
        faculty_name = normalize_ws(crumb.get_text(" "))
    if not faculty_name:
        faculty_name = _page_title(soup)

    number = re.search(r"/fakultaet(\d+)", url)

    institutes: list[dict] = []
    current_institute = None
    current_group = None

    for element in [e for b in blocks for e in b.find_all(["h1", "h2", "h3", "table"])]:
        if element.name in ("h1", "h2", "h3"):
            heading = normalize_ws(element.get_text(" "))
            if not heading:
                continue
            # An h2 opens an institute; an h3 only sub-labels one, so it becomes
            # the group label on the rows that follow instead of a new section.
            if element.name in ("h1", "h2"):
                current_institute = {"institute": heading, "chairs": []}
                institutes.append(current_institute)
                current_group = None
            else:
                current_group = heading
            continue

        if current_institute is None:
            current_institute = {"institute": None, "chairs": []}
            institutes.append(current_institute)

        # The first header cell names what the table lists ("Fachgebiet",
        # "Arbeitsgebiet", "Honorarprofessuren", "Bezeichnung" for former chairs).
        headers = [normalize_ws(th.get_text(" ")) for th in element.select("thead th")]
        kind = headers[0] if headers else None

        for row in element.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            name = normalize_ws(cells[0].get_text(" "))
            if not name:
                continue
            anchor = cells[0].find("a", href=True)
            href = urljoin(url, anchor["href"]) if anchor else None
            current_institute["chairs"].append(
                {
                    "name": name,
                    "url": href,
                    "external": bool(href) and not _is_btu(href),
                    "head_of_chair": normalize_ws(cells[1].get_text(" ")) or None,
                    "kind": kind,
                    "group": current_group,
                    # "ehemalige Fachgebiete" tables list chairs that no longer
                    # exist; flagging them keeps them in the export without
                    # implying the people listed are current staff.
                    "former": bool(
                        re.search(r"ehemal", f"{current_institute['institute']} {kind}", re.I)
                    ),
                }
            )

    return {
        "faculty_url": url,
        "faculty_name": faculty_name,
        "faculty_number": int(number.group(1)) if number else None,
        "institutes": institutes,
        "chair_count": sum(len(i["chairs"]) for i in institutes),
    }


# Every page's <title> ends with the university's name; the chair or faculty is
# the part in front of it.
SITE_SUFFIX_RE = re.compile(r"\s*[-|]\s*BTU\b.*$", re.I)


def _page_title(soup: BeautifulSoup) -> str | None:
    """The page's own title with the site-wide suffix removed."""
    title_tag = soup.select_one("title")
    if not title_tag:
        return None
    return normalize_ws(SITE_SUFFIX_RE.sub("", normalize_ws(title_tag.get_text()))) or None


def _is_btu(url: str) -> bool:
    """True for URLs on the university's own host (the only ones we crawl)."""
    host = urlparse(url).netloc.lower()
    return host in ("www.b-tu.de", "b-tu.de")


# --- CHAIR NAVIGATION ---

# Which top-level nav item holds the staff listing. "Team" covers 214 of the 226
# chairs; the rest label the same dropdown differently, so all known synonyms are
# matched and a chair may legitimately expose more than one of them.
TEAM_MENU_RE = re.compile(
    r"^(team|teammitglieder|mitarbeiter|mitarbeiterinnen|mitarbeiter\*innen|"
    r"mitarbeitende|personen|people|staff|lehrstuhlteam|unser team|"
    r"fachgebietsleitung|zur person)",
    re.I,
)

# Team dropdown entries that are pages *about* the chair rather than about
# people. They are still crawled — a "Kontakt" page often carries the
# secretariat's details — but the label is not treated as a job designation.
NON_DESIGNATION_RE = re.compile(
    r"^(anfahrt|adresse|kontakt|sprechzeiten|wir in der presse|impressum|"
    r"anreise|lageplan|office hours|contact)",
    re.I,
)


def parse_chair_nav(html: str, url: str) -> dict:
    """Read a chair's main navigation and pick out its Team dropdown.

    Every page in a chair's section renders the same ``nav.top-bar``, so this
    works whichever page of the chair we happened to land on — which matters
    because several faculty tables deep-link straight into ``/team/...``.
    """
    soup = make_soup(html)
    nav = soup.select_one("nav.top-bar")

    chair_title = None
    heading = soup.select_one("nav.breadcrumb-section li a span[itemprop=name]")
    if heading:
        chair_title = normalize_ws(heading.get_text(" "))
    if not chair_title:
        chair_title = _page_title(soup)

    sections: list[dict] = []
    if nav:
        for item in nav.select("div.top-bar-left > ul.dropdown.menu > li.nav-item"):
            anchor = item.find("a", recursive=False)
            if anchor is None:
                continue
            label = normalize_ws(anchor.get_text(" "))
            entries = []
            for sub in item.select("ul.submenu > li"):
                sub_anchor = sub.find("a", recursive=False)
                if sub_anchor is None or not sub_anchor.get("href"):
                    continue
                sub_label = normalize_ws(sub_anchor.get_text(" "))
                entries.append(
                    {
                        "designation": sub_label,
                        "url": urljoin(url, sub_anchor["href"]),
                        "is_designation": not NON_DESIGNATION_RE.match(sub_label),
                    }
                )
            sections.append(
                {
                    "label": label,
                    "url": urljoin(url, anchor["href"]) if anchor.get("href") else None,
                    "entries": entries,
                }
            )

    team_sections = [s for s in sections if TEAM_MENU_RE.match(s["label"] or "")]
    return {
        "chair_title": chair_title,
        "has_nav": nav is not None,
        "menu_labels": [s["label"] for s in sections],
        "team_sections": team_sections,
    }


def parse_subnav(html: str, url: str) -> dict:
    """Read the ``ul.side-nav`` sidebar listing the siblings of a person page.

    On per-person categories every member has their own page and the sidebar is
    the only complete index of them; the entry for the current page is rendered
    as ``<strong>`` with no link, so it is skipped here (that person is parsed
    from the page body instead).
    """
    soup = make_soup(html)
    nav = soup.select_one("nav.hide-for-print")
    if nav is None:
        return {"heading": None, "siblings": []}

    header = nav.select_one("h3.subnav__header")
    siblings = []
    for item in nav.select("ul.side-nav li.subnav-item"):
        anchor = item.find("a", href=True)
        if anchor is None:
            continue
        for tag in anchor.find_all("dfn"):  # "3:" screen-reader position labels
            tag.extract()
        siblings.append(
            {
                "label": normalize_ws(anchor.get_text(" ")),
                "url": urljoin(url, anchor["href"]),
            }
        )
    return {
        "heading": normalize_ws(header.get_text(" ")) if header else None,
        "siblings": siblings,
    }


def parse_breadcrumb(html: str, url: str) -> list[str]:
    """The breadcrumb trail, which names the chair and the team category."""
    soup = make_soup(html)
    crumbs = []
    for item in soup.select("nav.breadcrumb-section li"):
        name = item.select_one("span[itemprop=name]")
        if name:
            crumbs.append(normalize_ws(name.get_text(" ")))
    return crumbs


# --- PERSON BLOCKS ---

# Contact lines are labelled by the site with a one-letter prefix ("T", "F") or a
# spelled-out word, in German or English.
PHONE_RE = re.compile(r"^(?:t|tel|telefon|phone|fon)\b[\s.:]*", re.I)
FAX_RE = re.compile(r"^(?:f|fax|telefax)\b[\s.:]*", re.I)
MOBILE_RE = re.compile(r"^(?:m|mob|mobil|mobile|handy)\b[\s.:]*", re.I)
ROOM_RE = re.compile(
    r"\b(?:raum|room|zimmer|zi\.|r\.)\s*[:.]?\s*([\w.\-/ ]*\d[\w.\-/]*)", re.I
)
BUILDING_RE = re.compile(
    r"\b(?:geb[äa]ude|building|lehrgeb[äa]ude|hauptgeb[äa]ude|"
    r"verf[üu]gungsgeb[äa]ude|zentrales h[öo]rsaalgeb[äa]ude|haus)\b",
    re.I,
)
POSTCODE_RE = re.compile(r"\b\d{5}\s+\w")
DIGITS_RE = re.compile(r"\d")

# A name has to survive this to be emitted as a person. Prose lines from a
# "Sprechzeiten" page are what it exists to reject.
# The lookahead accepts either an abbreviating dot or a word boundary, so
# "rer." and "nat." match while "drei" and "national" do not.
NAME_TITLE_RE = re.compile(
    r"\b(prof|dr|dipl|m\.?\s?sc|b\.?\s?sc|m\.?\s?a|b\.?\s?a|ph\.?\s?d|msc|bsc|"
    r"ing|habil|rer|nat|jur|med|phil|pd|jun|apl|hon)(?=\.|\b)",
    re.I,
)
NAME_STOPWORDS_RE = re.compile(
    r"^(sprechzeit|kontakt|anschrift|adresse|anfahrt|impressum|hinweis|"
    r"weitere|mehr |siehe |bitte |die |der |das |wir |im |am |zur |zum |"
    r"aktuell|news|nachricht|termin|montag|dienstag|mittwoch|donnerstag|"
    r"freitag|samstag|sonntag|telefon|e-?mail|fax|raum|geb[äa]ude|www\.|http)",
    re.I,
)

# Section headings that mean "the block below is a person", used to accept an
# entry that carries a bare name with no contact details (typical of alumni
# listings).
DESIGNATION_HINT_RE = re.compile(
    r"(professor|dozent|leiter|leitung|inhaber|mitarbeit|sekretariat|assisten|"
    r"team|doktorand|promovend|stipendiat|hilfskraft|hilfskr[äa]fte|"
    r"besch[äa]ftigt|ehemalig|alumni|emerit|gast|extern|lehrbeauftragt|"
    r"koordinat|techniker|verwaltung|staff|member|secretar|personal)",
    re.I,
)


HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]

# Several chairs publish their staff as a grid instead of stacked text blocks.
# The header row names the columns, so the table is read by label rather than by
# position — the column order differs from chair to chair.
TABLE_COLUMN_MAP = {
    "name": "name",
    "nachname": "name",
    "person": "name",
    "vorname": "first_name",
    "titel": "title",
    "title": "title",
    "akad. grad": "title",
    "funktion": "role",
    "position": "role",
    "t\u00e4tigkeit": "role",
    "aufgabe": "role",
    "aufgabengebiet": "role",
    "bereich": "role",
    "arbeitsgebiet": "role",
    "tel": "phone",
    "tel.": "phone",
    "telefon": "phone",
    "phone": "phone",
    "durchwahl": "phone",
    "fax": "fax",
    "e-mail": "email",
    "email": "email",
    "mail": "email",
    "b\u00fcro": "room",
    "raum": "room",
    "room": "room",
    "zimmer": "room",
    "office": "room",
    "orcid": "orcid",
    "rg": "link",
    "researchgate": "link",
    "web": "link",
    "homepage": "link",
    "profil": "link",
    "sprechzeit": "consultation",
    "sprechzeiten": "consultation",
    "sprechstunde": "consultation",
    "adresse": "address",
    "anschrift": "address",
}

# "Tel. +49(0)355-69-" over a column of bare extensions: the header carries the
# shared prefix and each cell only the last digits.
PHONE_PREFIX_RE = re.compile(r"(\+\s?\d[\d\s()\-/]{5,})$")


def _header_key(text: str) -> str | None:
    """Map a table header cell onto a canonical person field, if we know it."""
    cleaned = normalize_ws(text).lower().strip(" :*")
    if not cleaned:
        return None
    if cleaned in TABLE_COLUMN_MAP:
        return TABLE_COLUMN_MAP[cleaned]
    # Headers routinely carry an inline note ("Tel. +49(0)355-69-", "E-Mail *").
    for label, key in TABLE_COLUMN_MAP.items():
        if cleaned.startswith(label):
            return key
    return None


def _parse_person_table(table: Tag, url: str, section: str | None) -> list[dict]:
    """Read a staff grid into person records, or return [] if it isn't one.

    Requires a header row that names a "Name" column: without it there is no way
    to tell a staff table from the many other tables the site publishes, and
    guessing would invent people.
    """
    rows = table.find_all("tr")
    if len(rows) < 2:
        return []

    header_cells = rows[0].find_all(["th", "td"])
    columns = [_header_key(cell.get_text(" ")) for cell in header_cells]
    if "name" not in columns:
        return []

    phone_prefix = ""
    for cell, key in zip(header_cells, columns):
        if key == "phone":
            header_text = normalize_ws(cell.get_text(" "))
            match = PHONE_PREFIX_RE.search(header_text.rstrip("-").strip())
            if match:
                # The header keeps its trailing separator ("+49(0)355-69-") so
                # the extension appends into a dialable number.
                phone_prefix = normalize_ws(match.group(1)) + (
                    "-" if header_text.endswith("-") else " "
                )

    people = []
    for row in rows[1:]:
        cells = row.find_all(["th", "td"])
        if not cells:
            continue

        person = {
            "name": None, "title": None, "first_name": None, "role": None,
            "emails": [], "phones": [], "faxes": [], "mobiles": [],
            "room": None, "building": None, "address_lines": [], "orcid": None,
            "links": [], "other_lines": [],
        }
        profile_url = None

        for cell, key in zip(cells, columns):
            text = normalize_ws(cell.get_text(" "))
            anchors = cell.find_all("a")

            if key == "email":
                for anchor in anchors:
                    email = _link_email(anchor)
                    if email and email not in person["emails"]:
                        person["emails"].append(email)
                email = normalize_email(text)
                if email and email not in person["emails"]:
                    person["emails"].append(email)
                continue
            if key == "orcid":
                for anchor in anchors:
                    href = anchor.get("href") or ""
                    if "orcid.org" in href:
                        person["orcid"] = href.rstrip("/").rsplit("/", 1)[-1]
                if not person["orcid"] and re.fullmatch(r"[\d\-Xx]{15,}", text):
                    person["orcid"] = text
                continue
            if key == "link":
                for anchor in anchors:
                    href = anchor.get("href") or ""
                    if href and not href.startswith("#"):
                        person["links"].append(
                            {"label": text or normalize_ws(anchor.get_text(" ")),
                             "url": urljoin(url, href)}
                        )
                continue
            if not text:
                continue
            if key == "name":
                person["name"] = text
                for anchor in anchors:
                    href = anchor.get("href") or ""
                    if href and not href.startswith("#") and not _link_email(anchor):
                        profile_url = urljoin(url, href)
                continue
            if key == "phone":
                person["phones"].append(
                    text if text.startswith("+") or not phone_prefix else phone_prefix + text
                )
            elif key == "fax":
                person["faxes"].append(text)
            elif key in ("title", "first_name", "role", "room"):
                person[key] = text
            elif key == "address":
                person["address_lines"].append(text)
            elif key == "consultation":
                person["other_lines"].append(f"Sprechzeit: {text}")
            else:
                person["other_lines"].append(text)

        if not person["name"]:
            continue
        # "Titel" and "Vorname" are separate columns; the export carries one name.
        full_name = " ".join(
            part for part in (person.pop("title"), person.pop("first_name"), person["name"]) if part
        )
        if not _is_name_like(full_name):
            continue
        person["name"] = _clean_name(full_name)
        person.update(
            {"image_url": None, "image_alt": None, "image_caption": None,
             "career": None, "office_hours": None}
        )
        if profile_url:
            person["links"].insert(0, {"label": "Profil", "url": profile_url})

        role = person.pop("role")
        person["section_heading"] = role or section
        person["source_url"] = url
        person["raw_text"] = normalize_ws(row.get_text(" | "))
        people.append(person)

    return people


def looks_like_person_name(text: str) -> bool:
    """True when a label names an individual rather than a role.

    Some chairs list their members directly in the Team dropdown instead of
    grouping them ("Madlen Herzig" beside "Ehemalige"), and the crawler must not
    file a person's own name as their job designation.
    """
    return bool(text) and _is_name_like(text) and not DESIGNATION_HINT_RE.search(text)


def _iter_content_nodes(root: Tag):
    """Yield ("heading"|"body", tag) over a page in document order.

    Descent stops at every ``div.ce-bodytext`` so a person block is visited once
    even though the site nests textmedia frames inside grid-container frames —
    both wrappers resolve to the same bodytext element.
    """
    for child in root.children:
        if not isinstance(child, Tag):
            continue
        classes = child.get("class") or []
        if "ce-bodytext" in classes:
            yield "body", child
        elif child.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            yield "heading", child
        else:
            yield from _iter_content_nodes(child)


def _split_lines(block: list) -> list[dict]:
    """Split a run of nodes on <br> into lines, keeping each line's anchors.

    Anchors are carried alongside the text because that is where the address
    lives: the visible text of a mail link is often just "E-Mail".
    """
    lines: list[dict] = []
    current: list = []

    def flush():
        if not current:
            return
        text_parts, anchors = [], []
        for node in current:
            if isinstance(node, NavigableString):
                text_parts.append(str(node))
            elif isinstance(node, Tag):
                text_parts.append(node.get_text(" "))
                if node.name == "a":
                    anchors.append(node)
                anchors.extend(node.find_all("a"))
        text = normalize_ws(" ".join(text_parts))
        if text or anchors:
            lines.append({"text": text, "anchors": anchors})
        current.clear()

    for node in block:
        if isinstance(node, Tag) and node.name == "br":
            flush()
        else:
            current.append(node)
    flush()
    return lines


# Lowercase words that belong inside a name rather than marking a sentence, so
# "Wouter Verschoof-van der Vaart" is not mistaken for prose.
NAME_PARTICLES = {
    "von", "van", "de", "der", "den", "des", "du", "da", "di", "dos", "del",
    "della", "la", "le", "zu", "ter", "ten", "af", "av", "bin", "al",
    "el", "y", "i", "ibn", "mac", "mc", "o",
}

# A line that opens with a role noun is a section label ("Lehrstuhlinhaber
# (2010-2011): ...") or a sentence about one, never a name.
ROLE_LEAD_RE = re.compile(
    r"^\S*(professor|dozent|inhaber|leiter|leitung|mitarbeit|sekretariat|"
    r"assistenz|doktorand|hilfskraft|hilfskr[äa]fte|lehrbeauftragt|habilitation|"
    r"dissertation|promotion|team|gruppe|kontakt|sprechzeit)",
    re.I,
)


# Role vocabulary in full words, used to reject a job label standing in for a
# name. Deliberately stricter than DESIGNATION_HINT_RE (which only inspects the
# surrounding section heading), so a surname like "Gastmann" is never caught.
ROLE_PHRASE_RE = re.compile(
    r"\b(professor(?:in|en|ship)?|professur|dozent|inhaber|leiter|leiterin|"
    r"leitung|mitarbeiter|mitarbeiterin|mitarbeitende|sekretariat|sekret[äa]rin|"
    r"assistenz|assistentin|doktorand|promovend|stipendiat|hilfskraft|"
    r"hilfskr[äa]fte|lehrbeauftragte|koordinator|koordinatorin|verwaltung|"
    r"bewerbung|praktikant|praktikum|stellenangebot|ausschreibung|"
    r"sprechzeit|sprechstunde|anschrift|impressum|kontakt|lehrstuhlteam|"
    # Section labels from the CV pages, which sit exactly where a name would.
    r"ausbildung|akademische|qualifikation|abschluss|studium|berufserfahrung|"
    r"forschungsinteresse|forschungsschwerpunkt|publikation|ver[öo]ffentlichung|"
    r"lehrveranstaltung|mitgliedschaft|auszeichnung|vortr[äa]g|werdegang|"
    r"lebenslauf|biografie|biographie|expertise|schwerpunkt)\w*",
    re.I,
)


# Institution vocabulary, so an organisation is never recorded as a person. Only
# the part before a parenthesis is tested, leaving "Dr. Eduard Wagner (TU Berlin)"
# intact — there the institution is an annotation on a real name.
ORG_NAME_RE = re.compile(
    r"\b(btu|universit[äa]t|university|hochschule|institut|institute|"
    r"fakult[äa]t|faculty|fachgebiet|lehrstuhl|zentrum|centre|center|gmbh|"
    r"mbh|e\.\s?v\.|akademie|stiftung|verein|abteilung|department|klinik|"
    r"labor|fraunhofer|leibniz|helmholtz)\b",
    re.I,
)


def _clean_name(text: str) -> str:
    """Trim the role text some chairs append after a pipe ("... | Fachgebiet X")."""
    return normalize_ws(text.split("|")[0])


def _split_heading(tag: Tag, text: str) -> tuple[str | None, str | None]:
    """Separate a heading into (name, role) when it packs both.

    Chairs write "<h4>Rui Chen, M.Sc.<br>Doktorand der Mathematik</h4>", so the
    flattened heading reads as neither a clean name nor a clean role. Splitting on
    the line break recovers both instead of losing the person.
    """
    if looks_like_person_name(text):
        return text, None
    lines = [line["text"] for line in _split_lines(list(tag.children)) if line["text"]]
    if len(lines) > 1 and looks_like_person_name(lines[0]):
        return lines[0], normalize_ws(" ".join(lines[1:])) or None
    return None, None


def _is_title_only(text: str) -> bool:
    """True for a run of academic titles with no actual name attached.

    Some pages break the line as "Prof. Dr. phil.<br>Christer Petersen", so the
    title half must not be mistaken for the whole name.
    """
    words = [w for w in re.split(r"[\s,]+", text) if w]
    if not words:
        return False
    return all(
        NAME_TITLE_RE.search(word) or word.strip(".,-").lower() in NAME_PARTICLES
        for word in words
    )


def _is_name_like(text: str) -> bool:
    """Whether a line reads as a person's name rather than prose or a heading."""
    if not text or len(text) > 90:
        return False
    if NAME_STOPWORDS_RE.match(text) or ROLE_LEAD_RE.match(text):
        return False
    # "[Apr. 2012] Universität Bonn beruft ..." is a news item; a trailing colon
    # marks a heading that introduces a list ("Wissenschaftliche Mitarbeiter:").
    # A trailing dot is *not* disqualifying — plenty of names end in "B.Sc.".
    if text.startswith(("[", "(", "„", '"', "'")) or text.rstrip().endswith(("!", "?")):
        return False
    # "beratendes Mitglied des PhD Ausschusses" is a list entry, not a name: no
    # published name starts with a lowercase word.
    if text[:1].islower():
        return False
    # No name contains a colon or semicolon; "Labore: Allgemeine Elektrotechnik"
    # and "For Phil Andros (2011); Die Hunterklasse (2008)" do.
    if ":" in text or ";" in text:
        return False
    # "Akademische Mitarbeiterin" / "University Professor in Marketing" are job
    # labels the site prints where a name would go. Only the part before any
    # parenthesis is checked, so "M. Sc. Tian-yun Gao (Gastdoktorand)" survives.
    if ROLE_PHRASE_RE.search(text.split("(")[0]):
        return False
    if EMAIL_RE.search(text):
        return False
    if DIGITS_RE.search(text) and not NAME_TITLE_RE.search(text):
        return False

    words = [w for w in re.split(r"[\s,]+", text) if w]
    if len(words) < 2 or len(words) > 12:
        return False
    # Section headings are routinely set in caps ("AKADEMISCHE MITARBEITER").
    if text == text.upper() and len(text) > 4:
        return False

    # Two or more ordinary lowercase words means a sentence. Only the part before
    # any parenthesis is judged, because a trailing "(Akad. Mitarbeiter, 2023)"
    # is an annotation on a real name rather than prose.
    head = text.split("(")[0]
    prose = [
        word
        for word in re.split(r"[\s,]+", head)
        if word[:1].islower()
        and word.strip(".,-").lower() not in NAME_PARTICLES
        and not NAME_TITLE_RE.search(word)
    ]
    if len(prose) >= 2:
        return False

    # "Prof. Dr. phil." is a title stack, not a name.
    if _is_title_only(text):
        return False
    # "Datenbank- und Informationssysteme" is a chair, not a person: a conjunction
    # joining two capitalised words is how the site names organisational units,
    # and no personal name is built that way.
    if re.search(r"\s(?:und|and|&|/)\s", text, re.I):
        return False
    # "BTU Cottbus-Senftenberg" reads like a two-word name but is an institution.
    if ORG_NAME_RE.search(text.split("(")[0]):
        return False

    # At least two capitalised words: "Madlen Herzig", "Prof. Dr. Markus Gardill".
    return sum(1 for w in words if w[:1].isupper()) >= 2


def _classify_lines(lines: list[dict], base_url: str) -> dict:
    """Sort a person's detail lines into contact fields, keeping the leftovers."""
    details = {
        "emails": [],
        "phones": [],
        "faxes": [],
        "mobiles": [],
        "room": None,
        "building": None,
        "address_lines": [],
        "links": [],
        "orcid": None,
        "other_lines": [],
        "office_hours_lines": [],
    }

    # Once a line opens the office-hours note, the rest of the block belongs to
    # it. This matters for more than tidiness: "Wir bitten um vorherige
    # Anmeldung: <secretary@…>" would otherwise be filed as the professor's own
    # address, which is the one error worse than having no address at all.
    in_office_hours = False

    for line in lines:
        text = line["text"]
        matched = False
        if text and OFFICE_HOURS_INLINE_RE.search(text):
            in_office_hours = True
        if in_office_hours:
            for anchor in line["anchors"]:
                email = _link_email(anchor)
                if email and email not in text:
                    text = normalize_ws(f"{text} {email}")
            if text:
                details["office_hours_lines"].append(text)
            continue

        for anchor in line["anchors"]:
            email = _link_email(anchor)
            if email:
                if email not in details["emails"]:
                    details["emails"].append(email)
                matched = True
                continue
            href = anchor.get("href") or ""
            if not href or href.startswith("#"):
                continue
            absolute = urljoin(base_url, href)
            label = normalize_ws(anchor.get_text(" "))
            if "orcid.org" in absolute:
                details["orcid"] = label or absolute.rsplit("/", 1)[-1]
                matched = True
                continue
            entry = {"label": label, "url": absolute}
            if entry not in details["links"]:
                details["links"].append(entry)
            if BUILDING_RE.search(label):
                details["building"] = details["building"] or label
                matched = True

        if not text:
            continue

        email = normalize_email(text) if "@" in text or "(at)" in text.lower() else None
        if email:
            if email not in details["emails"]:
                details["emails"].append(email)
            continue

        if FAX_RE.match(text) and DIGITS_RE.search(text):
            details["faxes"].append(normalize_ws(FAX_RE.sub("", text)))
            continue
        if MOBILE_RE.match(text) and DIGITS_RE.search(text):
            details["mobiles"].append(normalize_ws(MOBILE_RE.sub("", text)))
            continue
        if PHONE_RE.match(text) and DIGITS_RE.search(text):
            details["phones"].append(normalize_ws(PHONE_RE.sub("", text)))
            continue

        room = ROOM_RE.search(text)
        if room:
            details["room"] = details["room"] or normalize_ws(room.group(1))
            before = normalize_ws(text[: room.start()]).strip(",; ")
            if before and not details["building"]:
                details["building"] = before
            matched = True
            continue
        if BUILDING_RE.search(text):
            details["building"] = details["building"] or text
            continue
        if POSTCODE_RE.search(text) or re.search(r"\b(stra[ßs]e|str\.|weg|platz|allee)\b", text, re.I):
            details["address_lines"].append(text)
            continue

        if not matched:
            details["other_lines"].append(text)

    # A building often arrives as one link labelled "Hauptgebäude, Raum 3.39";
    # the room half is already captured separately, so trim it back to the
    # building alone rather than storing the room twice.
    if details["building"]:
        room = ROOM_RE.search(details["building"])
        if room:
            head = normalize_ws(details["building"][: room.start()]).strip(",; ")
            details["room"] = details["room"] or normalize_ws(room.group(1))
            details["building"] = head or None

    return details


def _blocks_from_bodytext(bodytext: Tag) -> list[dict]:
    """Cut one ``div.ce-bodytext`` into per-person blocks.

    A block opens on a heading, on a paragraph led by ``<strong>``, or on a
    paragraph led by a non-mail link — the three ways the site marks a name.
    Everything after that, up to the next such marker, belongs to that person.
    """
    blocks: list[dict] = []
    current: dict | None = None
    section: str | None = None

    def open_block(name: str | None, nodes: list):
        nonlocal current
        current = {"name": name, "nodes": list(nodes), "section": section}
        blocks.append(current)

    for child in bodytext.children:
        if not isinstance(child, Tag):
            if isinstance(child, NavigableString) and normalize_ws(str(child)) and current:
                current["nodes"].append(child)
            continue

        if child.name in HEADING_TAGS or (
            child.name == "header" and child.find(HEADING_TAGS)
        ):
            heading = child if child.name != "header" else child.find(HEADING_TAGS)
            text = normalize_ws(heading.get_text(" "))
            heading_name, heading_role = _split_heading(heading, text)
            if heading_name:
                open_block(heading_name, [])
                if heading_role:
                    current["section"] = heading_role
            elif current is not None and current["name"] and not current["nodes"]:
                # A person page prints the name and the job title as two sibling
                # headings ("Clara Galle, M.Sc." then "Akademische
                # Mitarbeiterin"), with the contact details below both. The
                # second heading labels the person just opened rather than
                # starting a new block, or the details would detach from the name.
                current["section"] = text
            else:
                # An ordinary role heading: it labels everything that follows.
                section = text
                current = None
            continue

        if child.name in ("p", "div"):
            lead = None
            for node in child.children:
                if isinstance(node, NavigableString):
                    if normalize_ws(str(node)):
                        break
                    continue
                if isinstance(node, Tag):
                    lead = node
                break

            starts_person = False
            name = None
            if lead is not None and lead.name in ("strong", "b"):
                candidate = normalize_ws(lead.get_text(" "))
                if _is_name_like(candidate):
                    starts_person, name = True, candidate
            elif lead is not None and lead.name == "a" and not _link_email(lead):
                candidate = normalize_ws(lead.get_text(" "))
                if _is_name_like(candidate):
                    starts_person, name = True, candidate

            if starts_person:
                open_block(name, list(child.children))
                continue

            if current is None:
                # A paragraph with no name marker yet: its own first line may be
                # the name (the shape used when a heading names the role instead).
                open_block(None, list(child.children))
                continue
            current["nodes"].append(Tag(name="br"))
            current["nodes"].extend(child.children)
            continue

        if child.name in ("ul", "ol"):
            # Alumni are often just "<ul><li>Dr. Britta Kleefeld</li>...</ul>", so
            # each item is a candidate person rather than a detail line of the
            # block above it.
            items = child.find_all("li", recursive=False)
            if items and all(_is_name_like(normalize_ws(li.get_text(" "))) for li in items):
                for item in items:
                    open_block(normalize_ws(item.get_text(" ")), list(item.children))
                continue
            if current is not None:
                current["nodes"].append(Tag(name="br"))
                current["nodes"].append(child)
            continue

        if child.name == "table":
            if current is not None:
                current["nodes"].append(Tag(name="br"))
                current["nodes"].append(child)
            continue

        if current is not None:
            current["nodes"].append(child)

    return blocks


def _person_from_block(block: dict, url: str, section: str | None) -> dict | None:
    """Turn one block into a person record, or None if it isn't one."""
    lines = _split_lines(block["nodes"])
    name = block["name"]
    # A heading inside the block is more specific than the page-level section.
    section = block.get("section") or section

    if not name:
        # No heading/strong marker: the first line is the name, if it reads like
        # one — this is the shape of a per-person page under a role heading.
        while lines and not lines[0]["text"] and not lines[0]["anchors"]:
            lines.pop(0)
        if not lines:
            return None
        # "Prof. Dr. phil." on its own line, with the name on the next one.
        if (
            len(lines) > 1
            and _is_title_only(lines[0]["text"])
            and _is_name_like(f"{lines[0]['text']} {lines[1]['text']}")
        ):
            name = normalize_ws(f"{lines[0]['text']} {lines[1]['text']}")
            lines = lines[2:]
        elif _is_name_like(lines[0]["text"]):
            name = lines[0]["text"]
            lines = lines[1:]
        else:
            return None
    else:
        # Drop a leading line that just repeats the heading we already took.
        if lines and normalize_ws(lines[0]["text"]) == name:
            lines = lines[1:]

    name = _clean_name(name)
    if not _is_name_like(name):
        return None

    details = _classify_lines(lines, url)
    inline_hours = " ".join(details.pop("office_hours_lines")) or None
    has_contact = any(
        details[key] for key in ("emails", "phones", "faxes", "mobiles")
    ) or details["room"] or details["orcid"]

    # A bare name is only accepted where the surrounding heading says the block
    # lists people (alumni tables carry names and nothing else); otherwise it is
    # far more likely to be a sentence that happened to look name-shaped.
    if not has_contact:
        context = f"{section or ''} {name}"
        if not (DESIGNATION_HINT_RE.search(context) or NAME_TITLE_RE.search(name)):
            return None

    raw_text = " | ".join(line["text"] for line in lines if line["text"])
    return {
        "name": name,
        "section_heading": section,
        "image_url": None,
        "image_alt": None,
        "image_caption": None,
        "career": None,
        "office_hours": inline_hours,
        "emails": details["emails"],
        "phones": details["phones"],
        "faxes": details["faxes"],
        "mobiles": details["mobiles"],
        "room": details["room"],
        "building": details["building"],
        "address_lines": details["address_lines"],
        "orcid": details["orcid"],
        "links": details["links"],
        "other_lines": details["other_lines"],
        "source_url": url,
        "raw_text": raw_text,
    }


# --- PORTRAIT, CAREER AND OFFICE HOURS ---

# TYPO3's text-with-image element puts the portrait in a ``ce-gallery`` beside
# the ``ce-bodytext`` holding that person's details, so the two are siblings.
def _bodytext_images(bodytext: Tag, url: str) -> list[dict]:
    """Portraits published alongside one ``ce-bodytext``, in document order."""
    container = bodytext.parent
    if container is None or "ce-textpic" not in (container.get("class") or []):
        container = bodytext.find_parent(class_="ce-textpic") or bodytext.parent
    if container is None:
        return []

    return _collect_images(container, url)


def _collect_images(container: Tag, url: str) -> list[dict]:
    """Every gallery portrait inside ``container``, de-duplicated."""
    images = []
    for figure in container.select("div.ce-gallery figure, div.ce-gallery img"):
        image = figure if figure.name == "img" else figure.find("img")
        if image is None:
            continue
        # Lazy-loaded galleries keep the real file in data-lazy and ship a
        # placeholder in src.
        source = image.get("data-lazy") or image.get("src")
        if not source:
            continue
        caption = figure.find("figcaption") if figure.name != "img" else None
        entry = {
            "url": urljoin(url, source),
            "alt": normalize_ws(image.get("alt") or "") or None,
            "caption": normalize_ws(caption.get_text(" ")) if caption else None,
        }
        if entry not in images:
            images.append(entry)
    return images


# Headings that introduce a CV. Matched against the whole heading so
# "Wissenschaftlicher Werdegang" and "Vita" both land.
CAREER_HEADING_RE = re.compile(
    r"(lebenslauf|werdegang|\bvita\b|curriculum\s*vitae|^\s*cv\s*$|\bcareer\b|"
    r"biograph|biografie|laufbahn|zur\s+person|academic\s+background|"
    r"berufliche[rn]?\s+werdegang|wissenschaftliche[rn]?\s+werdegang)",
    re.I,
)

# The same idea as OFFICE_HOURS_HEADING_RE but for a cue in running text rather
# than a heading, including the phrases that introduce a booking address.
OFFICE_HOURS_INLINE_RE = re.compile(
    r"(sprechstunde|sprechzeit|office\s*hour|consultation\s*hour|"
    r"terminvereinbarung|nach\s+vereinbarung|vorherige\s+anmeldung)",
    re.I,
)

OFFICE_HOURS_HEADING_RE = re.compile(
    r"(sprechzeit|sprechstunde|office\s*hour|consultation\s*hour|konsultation)",
    re.I,
)

# Cap on the CV text kept per person. Long enough for the multi-paragraph
# biographies the site publishes, short enough that one runaway page can't
# dominate the export.
MAX_SECTION_CHARS = 8000

# Every way the site renders a "click to reveal" section title.
TOGGLE_TITLE_SELECTOR = "a.accordion-title, a.tabs-title, li.tabs-title > a"


def _section_text(nodes: list[Tag]) -> str | None:
    """Flatten a run of content nodes into readable plain text."""
    parts: list[str] = []
    for node in nodes:
        for element in node.find_all(["p", "li", "h3", "h4", "h5", "div"]) or []:
            if element.find(["p", "li"]):
                continue  # a wrapper; its children are collected on their own
            text = normalize_ws(element.get_text(" "))
            if text and text not in parts:
                parts.append(text)
        if not parts:
            text = normalize_ws(node.get_text(" "))
            if text:
                parts.append(text)
    joined = "\n".join(parts).strip()
    return joined[:MAX_SECTION_CHARS] or None


def _extra_sections(content: Tag) -> list[tuple[str, str, str]]:
    """Find every CV / office-hours section on a page, in document order.

    Returns ``(kind, heading, text)`` where kind is "career" or "office_hours".
    Two shapes carry them: an accordion (``a.accordion-title`` plus the
    ``div.accordion-content`` it toggles) and a plain heading followed by prose
    until the next heading.
    """
    found: list[tuple[str, str, str]] = []
    consumed: set[int] = set()

    # Accordions and tab strips are the same idea: a clickable title plus a panel
    # it reveals, addressed either as the title's next sibling or by the id in
    # its href. Chairs use both interchangeably for CVs.
    for title in content.select(TOGGLE_TITLE_SELECTOR):
        heading = normalize_ws(title.get_text(" "))
        kind = _section_kind(heading)
        if not kind:
            continue
        target = title.find_next_sibling(class_="accordion-content")
        if target is None:
            anchor_id = (title.get("href") or "").lstrip("#")
            target = content.find(id=anchor_id) if anchor_id else None
        if target is None:
            continue
        text = _section_text([target])
        if text:
            found.append((kind, heading, text))
            consumed.add(id(target))

    for heading_tag in content.find_all(HEADING_TAGS):
        heading = normalize_ws(heading_tag.get_text(" "))
        kind = _section_kind(heading)
        if not kind:
            continue
        text = _section_text(_section_nodes(heading_tag, consumed))
        if text:
            found.append((kind, heading, text))

    return found


def _section_nodes(heading_tag: Tag, consumed: set[int]) -> list[Tag]:
    """The content a heading introduces, whatever level the site wrapped it at.

    A heading may sit directly beside its text, be wrapped in a ``<header>``, or
    live in a header-only content element with the body in the frame that
    follows. Climbing until following content appears covers all three, and
    stopping at the next heading keeps one section from swallowing the next.
    """
    node: Tag | None = heading_tag
    for _ in range(6):  # bounded: user-content is never deeper than this
        if node is None or "user-content" in (node.get("class") or []):
            break
        nodes = []
        for sibling in node.next_siblings:
            if not isinstance(sibling, Tag):
                continue
            if sibling.name in HEADING_TAGS:
                break
            if id(sibling) in consumed:
                continue
            if sibling.find(HEADING_TAGS) is not None:
                # This block holds our text and then the next section's heading —
                # a chair that stacks "Werdegang", "Publikationen" and the rest
                # inside one content element. Keep the part before that heading
                # rather than discarding the section entirely.
                nodes.extend(_nodes_until_heading(sibling))
                break
            nodes.append(sibling)
        if nodes:
            return nodes
        node = node.parent
    return []


def _nodes_until_heading(container: Tag) -> list[Tag]:
    """Children of ``container`` preceding the first heading inside it."""
    kept: list[Tag] = []
    for child in container.children:
        if not isinstance(child, Tag):
            continue
        if child.name in HEADING_TAGS:
            break
        if child.find(HEADING_TAGS) is not None:
            kept.extend(_nodes_until_heading(child))
            break
        kept.append(child)
    return kept


def _section_kind(heading: str) -> str | None:
    """Classify a heading as introducing a CV, office hours, or neither."""
    if OFFICE_HOURS_HEADING_RE.search(heading):
        return "office_hours"
    if CAREER_HEADING_RE.search(heading):
        return "career"
    return None


_PAGE_LIST_FIELDS = ("emails", "phones", "faxes", "mobiles", "address_lines",
                    "links", "other_lines")
_PAGE_SCALAR_FIELDS = ("room", "building", "orcid", "section_heading",
                       "image_url", "image_alt", "image_caption",
                       "career", "office_hours")


def _page_subject(people: list[dict]) -> dict | None:
    """The person a page-level CV or office-hours block describes, if it is clear.

    A lone person on the page is unambiguous. Otherwise the one person carrying
    contact details is the page's subject — the other entries on a CV page are
    section headings that happened to read like names, and they never have an
    address or a phone number.
    """
    if len(people) == 1:
        return people[0]
    contactable = [
        person
        for person in people
        if person["emails"] or person["phones"] or person["room"]
    ]
    return contactable[0] if len(contactable) == 1 else None


def _dedupe_page_people(people: list[dict]) -> list[dict]:
    """Merge repeats of one person within a page into a single record.

    A person page often prints the same name twice — once as the contact block
    and again inside a "Sprechzeiten" accordion — and each copy carries a
    different half of the details. Merging by name keeps both halves and, just as
    importantly, leaves the page with a single subject so its CV can be attributed.
    """
    merged: dict[str, dict] = {}
    order: list[str] = []
    for person in people:
        key = person["name"].casefold()
        existing = merged.get(key)
        if existing is None:
            merged[key] = person
            order.append(key)
            continue
        for field in _PAGE_LIST_FIELDS:
            for value in person.get(field) or []:
                if value not in existing[field]:
                    existing[field].append(value)
        for field in _PAGE_SCALAR_FIELDS:
            existing[field] = existing.get(field) or person.get(field)
        if len(person.get("raw_text") or "") > len(existing.get("raw_text") or ""):
            existing["raw_text"] = person["raw_text"]
    return [merged[key] for key in order]


def parse_people_page(html: str, url: str) -> dict:
    """Extract every person published on one page.

    Works on both team-page shapes — the per-person page and the list page that
    stacks the whole team under role headings — because both are the same
    ``heading`` / ``ce-bodytext`` alternation underneath.
    """
    soup = make_soup(html)
    blocks = content_blocks(soup)
    if not blocks:
        return {
            "people": [], "page_heading": None, "parsed": False,
            "career": None, "office_hours": None,
        }

    page_heading = None
    people: list[dict] = []
    sections: list[tuple[str, str, str]] = []
    section: str | None = None
    # A heading that is itself a person's name, waiting to be paired with the
    # contact block that follows it (the shape used when the role is the section
    # header and the name is a sub-heading in its own content element).
    pending: dict | None = None

    def flush_pending():
        nonlocal pending
        if pending is None:
            return
        person = _person_from_block({"name": pending["name"], "nodes": []}, url, pending["section"])
        if person:
            people.append(person)
        pending = None

    for content in blocks:
      for tag in content.select("script, style, noscript"):
          tag.decompose()

      for kind, node in _iter_content_nodes(content):
        if kind == "heading":
            heading = normalize_ws(node.get_text(" "))
            if not heading:
                continue
            if page_heading is None:
                page_heading = heading
            heading_name, heading_role = _split_heading(node, heading)
            flush_pending()
            if heading_name:
                pending = {"name": heading_name, "section": heading_role or section}
            else:
                section = heading
            continue

        for table in node.find_all("table"):
            found = _parse_person_table(table, url, section)
            if found:
                flush_pending()
                people.extend(found)
            # Removed either way: a parsed table must not be re-read as text, and
            # an unparsed one is a layout grid whose cells aren't contact lines.
            table.extract()

        person_blocks = _blocks_from_bodytext(node)
        from_this_bodytext: list[dict] = []
        for index, block in enumerate(person_blocks):
            if index == 0 and pending is not None and block["name"] is None:
                block = {
                    "name": pending["name"],
                    "nodes": block["nodes"],
                    "section": block.get("section"),
                }
                block_section = pending["section"]
                pending = None
            else:
                flush_pending()
                block_section = section
            person = _person_from_block(block, url, block_section)
            if person:
                people.append(person)
                from_this_bodytext.append(person)

        # Portraits sit beside the text they belong to, but only assign one when
        # the pairing is unambiguous — a single person, or one image each in the
        # same order. Guessing on a group shot would label the wrong person.
        images = _bodytext_images(node, url)
        if images and (
            len(from_this_bodytext) == 1 or len(images) == len(from_this_bodytext)
        ):
            for person, image in zip(from_this_bodytext, images):
                person["image_url"] = image["url"]
                person["image_alt"] = image["alt"]
                person["image_caption"] = image["caption"]

      flush_pending()
      section = None
      sections.extend(_extra_sections(content))

    flush_pending()

    # A few pages publish the same block twice — once inside a grid container and
    # once on its own — which would otherwise read as two people and, worse, stop
    # a single-person page from claiming its own CV.
    people = _dedupe_page_people(people)

    # CV and office-hours sections are page-level: they follow the person the
    # page is about. On a single-person page that attribution is certain; on a
    # listing it is not, so the section is returned unattached for the caller to
    # place (or drop).
    career = "\n\n".join(t for k, _, t in sections if k == "career") or None
    office_hours = "\n\n".join(t for k, _, t in sections if k == "office_hours") or None
    subject = _page_subject(people)
    if subject is not None and not subject["image_url"]:
        # Some layouts place the portrait in a gallery of its own rather than
        # beside the text. With a single subject and a single portrait on the
        # page there is only one way to pair them.
        page_images = [img for block in blocks for img in _collect_images(block, url)]
        if len(page_images) == 1:
            subject["image_url"] = page_images[0]["url"]
            subject["image_alt"] = page_images[0]["alt"]
            subject["image_caption"] = page_images[0]["caption"]
    if subject is not None:
        subject["career"] = subject.get("career") or career
        subject["office_hours"] = subject.get("office_hours") or office_hours
        career = office_hours = None

    return {
        "people": people,
        "page_heading": page_heading,
        "parsed": True,
        # Non-null only when the page holds several people (or none), so the
        # crawler can decide who — if anyone — the text describes.
        "career": career,
        "office_hours": office_hours,
    }

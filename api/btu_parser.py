"""Deterministic HTML -> dict parsers for the two b-tu.de page types we crawl.

Both pages are plain server-rendered tables, so nothing here guesses: every value
is read out of a specific cell. Anything we don't recognise is still captured
verbatim (``unmapped_fields`` / ``unmapped_sections``) so a layout change on the
university side degrades into "unlabelled but present" rather than silent loss.

Two page types:

* **Module overview** (``https://www.b-tu.de/modul/<n>``) — one two-column table
  of ``label: value`` rows. The site serves the same layout with either German or
  English labels, so ``MODULE_FIELD_MAP`` maps both onto one canonical key set.
* **Event sub-page** (``qisserver3/rds?...veranstid=<n>``) — the timetable entry
  linked from a module's "events in the current semester" row. It is a stack of
  captioned tables, always German regardless of the parent module's language,
  identified by their stable ``summary`` attribute.
"""

import re
from urllib.parse import urljoin, urlparse, parse_qs

from bs4 import BeautifulSoup

# --- MODULE PAGE ---

# Canonical key <- the German and English label the site prints for that row.
# Both variants are listed for every field so a module renders identically
# whichever language the page was authored in.
MODULE_FIELD_MAP = {
    "Modulnummer": "module_number",
    "Module Number": "module_number",
    "Modultitel": "module_title",
    "Module Title": "module_title",
    "Einrichtung": "department",
    "Department": "department",
    "Verantwortlich": "responsible_staff",
    "Responsible Staff Member": "responsible_staff",
    "Lehr- und Prüfungssprache": "language_of_instruction",
    "Language of Teaching / Examination": "language_of_instruction",
    "Dauer": "duration",
    "Duration": "duration",
    "Angebotsturnus": "frequency_of_offer",
    "Frequency of Offer": "frequency_of_offer",
    "Leistungspunkte": "credits",
    "Credits": "credits",
    "Lernziele": "learning_outcome",
    "Learning Outcome": "learning_outcome",
    "Inhalte": "contents",
    "Contents": "contents",
    "Empfohlene Voraussetzungen": "recommended_prerequisites",
    "Recommended Prerequisites": "recommended_prerequisites",
    "Zwingende Voraussetzungen": "mandatory_prerequisites",
    "Mandatory Prerequisites": "mandatory_prerequisites",
    "Lehrformen und Arbeitsumfang": "forms_of_teaching",
    "Forms of Teaching and Proportion": "forms_of_teaching",
    "Unterrichtsmaterialien und Literaturhinweise": "teaching_materials",
    "Teaching Materials and Literature": "teaching_materials",
    "Modulprüfung": "module_examination",
    "Module Examination": "module_examination",
    "Prüfungsleistung/en für Modulprüfung": "assessment_mode",
    "Assessment Mode for Module Examination": "assessment_mode",
    "Bewertung der Modulprüfung": "evaluation_of_examination",
    "Evaluation of Module Examination": "evaluation_of_examination",
    "Teilnehmerbeschränkung": "participant_limit",
    "Limited Number of Participants": "participant_limit",
    "Zuordnung zu Studiengängen": "study_programmes",
    "Part of the Study Programme": "study_programmes",
    "Bemerkungen": "remarks",
    "Remarks": "remarks",
    "Veranstaltungen zum Modul": "module_components",
    "Module Components": "module_components",
    "Veranstaltungen im aktuellen Semester": "components_current_semester",
    "Components to be offered in the Current Semester": "components_current_semester",
    "Nachfolgemodul/e": "follow_up_modules",
    "Follow-up Module/s": "follow_up_modules",
    "Auslaufmodul": "phase_out_module",
    "Phase-out Module": "phase_out_module",
}

# Which label set a page used tells us the language it was authored in; the
# module number row is the cheapest unambiguous marker of each.
GERMAN_MARKER = "Modulnummer"
ENGLISH_MARKER = "Module Number"

# Rows whose value is a list of links to timetable sub-pages rather than prose.
EVENT_LIST_FIELD = "components_current_semester"

# Rows that carry both a lead-in sentence and a list (e.g. "Nachfolgemodul seit:
# 22.01.2025" followed by the successor module), so neither half can be dropped.
NOTE_PLUS_LIST_FIELDS = {"follow_up_modules", "phase_out_module"}


# --- EVENT SUB-PAGE ---

# The `summary` attribute is the stable machine-readable section id. Captions are
# not usable as keys: they carry the group name ("Termine Gruppe: 1-Gruppe") and
# vary in number ("Zugeordnete Person" vs "Zugeordnete Personen").
EVENT_SECTION_BASIC = "Grunddaten zur Veranstaltung"
EVENT_SECTION_DATES = "Übersicht über alle Veranstaltungstermine"
EVENT_SECTION_PERSONS = "Verantwortliche Dozenten"
EVENT_SECTION_PROGRAMMES = "Übersicht über die zugehörigen Studiengänge"
EVENT_SECTION_MODULES = "Übersicht über die zugehörigen Prüfungen"
EVENT_SECTION_INSTITUTIONS = "Übersicht über die zugehörigen Einrichtungen"
# Free-text block, present mainly on exam entries, carrying the exam form, date,
# time and room. Same th/td shape as Grunddaten, so it reuses that parser.
EVENT_SECTION_CONTENT = "Weitere Angaben zur Veranstaltung"

EVENT_CONTENT_FIELD_MAP = {
    "Beschreibung": "description",
    "Lehrmethoden und Lernziele": "teaching_methods_and_objectives",
}

EVENT_BASIC_FIELD_MAP = {
    "Veranstaltungsart": "event_type",
    "Veranstaltungsnummer": "event_number",
    "Semester": "semester",
    "SWS": "sws",
    "Erwartete Teilnehmer/-innen": "expected_participants",
    "Studienjahr": "academic_year",
    "Hyperlink": "hyperlink",
}

EVENT_DATE_COLUMN_MAP = {
    "Tag": "day",
    "Zeit": "time",
    "Rhythmus": "rhythm",
    "Dauer": "duration",
    "Raum": "room",
    # The header is marked up as "Raum-<br>plan", so reading the cell with a
    # separator yields the spaced spelling. Both are listed rather than stripping
    # the space generically, which would corrupt labels like "Lehr- und ...".
    "Raum-plan": "room_plan",
    "Raum- plan": "room_plan",
    "Lehrperson": "lecturer",
    "Bemerkung": "remark",
    "fällt aus am": "cancelled_on",
    "Max. Teilnehmer/-innen": "max_participants",
}

EVENT_PERSON_COLUMN_MAP = {
    "Zugeordnete Person": "person",
    "Zuständigkeit": "responsibility",
}

EVENT_PROGRAMME_COLUMN_MAP = {
    "Studiengang": "programme",
    "Semester": "semester",
    "PO": "po",
    "Bemerkung": "remark",
}

EVENT_MODULE_COLUMN_MAP = {
    "Modulnummer": "module_number",
    "Modultitel": "module_title",
}


def normalize_ws(text: str) -> str:
    """Collapse the site's tab/newline/nbsp padding into single spaces.

    The templates indent cell content with long tab runs and use &nbsp; inside
    sentences, so raw ``get_text()`` output is unusable as a value without this.
    """
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _slugify(label: str) -> str:
    """Fallback key for a label we have no canonical mapping for."""
    slug = re.sub(r"[^\w]+", "_", label.lower(), flags=re.UNICODE).strip("_")
    return slug or "unnamed"


def extract_veranstid(url: str) -> str | None:
    """Return the timetable event id from a qisserver3 event URL.

    The param is namespaced as ``veranstaltung.veranstid``, so we match on the
    suffix rather than an exact key.
    """
    for key, values in parse_qs(urlparse(url).query).items():
        if key.endswith("veranstid") and values:
            return values[0]
    return None


def _cell_links(cell, base_url: str) -> list[dict]:
    """Every anchor in a cell as {text, url}, resolved against the page URL."""
    links = []
    for anchor in cell.find_all("a"):
        href = anchor.get("href")
        if not href:
            continue
        links.append(
            {
                "text": normalize_ws(anchor.get_text(" ")),
                "url": urljoin(base_url, href),
            }
        )
    return links


def _cell_items(cell) -> list[str]:
    """List-item texts of a cell, or [] when the cell holds plain prose."""
    return [normalize_ws(li.get_text(" ")) for li in cell.find_all("li")]


def _find_module_table(soup: BeautifulSoup):
    """Return the module overview table, or None on a page that has none.

    We search by content rather than taking the first table: a 404 page still
    renders the cookie-consent table, and matching on it would produce a
    plausible-looking but entirely wrong record.
    """
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all("td", recursive=False)
            if not cells:
                continue
            label = normalize_ws(cells[0].get_text(" ")).rstrip(":")
            if label in (GERMAN_MARKER, ENGLISH_MARKER):
                return table
    return None


def parse_module_page(html: str, url: str) -> dict:
    """Parse a module overview page into structured fields.

    Returns a dict with ``parsed`` False when the page is not a module overview
    (404s and error pages), so the caller can record the miss instead of writing
    an empty-looking module.
    """
    soup = BeautifulSoup(html, "lxml")

    page_title = normalize_ws(soup.title.get_text(" ")) if soup.title else None
    heading = normalize_ws(soup.h1.get_text(" ")) if soup.h1 else None

    table = _find_module_table(soup)
    if table is None:
        return {
            "parsed": False,
            "parse_error": "No module overview table found (error or 404 page).",
            "page_title": page_title,
            "heading": heading,
            "language": None,
            "fields": {},
            "unmapped_fields": [],
            "event_links": [],
        }

    fields: dict = {}
    unmapped: list[dict] = []
    event_links: list[dict] = []
    notes: list[str] = []
    language = None
    last_key = None

    for row in table.find_all("tr"):
        cells = row.find_all("td", recursive=False)
        if len(cells) < 2:
            continue

        label = normalize_ws(cells[0].get_text(" ")).rstrip(":")
        value_cell = cells[1]
        text = normalize_ws(value_cell.get_text(" "))
        items = _cell_items(value_cell)
        links = _cell_links(value_cell, url)

        if label == GERMAN_MARKER:
            language = "de"
        elif label == ENGLISH_MARKER:
            language = "en"

        # The site prints the title's other-language rendering as a label-less
        # row directly beneath the title, so it is only identifiable by position.
        if not label:
            if last_key == "module_title" and text:
                fields["module_title_translated"] = text
            elif text:
                # Standalone annotations printed without a label, e.g. "Das Modul
                # ist für das Fachübergreifende Studium zugelassen." on ~200
                # modules. They belong to the module, not to the row above them.
                notes.append(text)
            continue

        key = MODULE_FIELD_MAP.get(label)
        if key is None:
            unmapped.append({"label": label, "text": text, "items": items, "links": links})
            last_key = None
            continue

        if key == EVENT_LIST_FIELD:
            # Each entry links to a timetable sub-page we follow one level deep.
            # Entries without a link (rare, e.g. an exam with no detail page) are
            # still recorded so the module's course list stays complete.
            entries = []
            for li in value_cell.find_all("li"):
                anchor = li.find("a")
                href = urljoin(url, anchor["href"]) if anchor and anchor.get("href") else None
                entries.append(
                    {
                        "title": normalize_ws(li.get_text(" ")),
                        "url": href,
                        "veranstid": extract_veranstid(href) if href else None,
                    }
                )
            fields[key] = entries
            event_links = [e for e in entries if e["url"]]
        elif key in NOTE_PLUS_LIST_FIELDS:
            # Keep the lead-in sentence and the listed module apart: the date in
            # "Nachfolgemodul seit: 22.01.2025" is not part of the module entry.
            note = text
            for item in items:
                note = note.replace(item, "")
            fields[key] = {
                "note": normalize_ws(note),
                "items": items,
                "links": links,
            }
        else:
            # A cell that used <ul> is genuinely a list of values; a plain cell is
            # one string. Preserving that distinction keeps consumers from having
            # to re-split prose on guessed delimiters.
            fields[key] = items if items else text

        last_key = key

    if notes:
        fields["notes"] = notes

    return {
        "parsed": True,
        "parse_error": None,
        "page_title": page_title,
        "heading": heading,
        "language": language,
        "fields": fields,
        "unmapped_fields": unmapped,
        "event_links": event_links,
    }


def _table_matrix(table, base_url: str) -> tuple[list[str], list[list[dict]]]:
    """Split a table into its header labels and its body cells.

    Each body cell is ``{text, links}`` so a room or person keeps the URL that
    sits behind its label.
    """
    columns: list[str] = []
    rows: list[list[dict]] = []

    for row in table.find_all("tr"):
        headers = row.find_all("th")
        cells = row.find_all("td")
        if headers and not cells:
            columns = [normalize_ws(h.get_text(" ")) for h in headers]
            continue
        if not cells:
            continue
        rows.append(
            [
                {"text": normalize_ws(c.get_text(" ")), "links": _cell_links(c, base_url)}
                for c in cells
            ]
        )

    return columns, rows


def _rows_as_dicts(
    table, base_url: str, column_map: dict, default_key: str = "value"
) -> list[dict]:
    """Turn a header+body table into a list of records.

    Unmapped columns keep a slug of their German header, so an added column shows
    up under a readable key instead of being dropped. Links found in a cell are
    attached as ``<key>_url``.
    """
    columns, rows = _table_matrix(table, base_url)

    keys = []
    for index, column in enumerate(columns):
        if not column:
            # The leading icon column has an empty header and no data value.
            keys.append(f"column_{index}")
        else:
            keys.append(column_map.get(column, _slugify(column)))

    records = []
    for cells in rows:
        record: dict = {}
        for index, cell in enumerate(cells):
            key = keys[index] if index < len(keys) else (
                default_key if len(cells) == 1 else f"column_{index}"
            )
            record[key] = cell["text"]
            if cell["links"]:
                record[f"{key}_url"] = cell["links"][0]["url"]
        # The leading column holds only an expand icon. Its text is always empty,
        # but its link points at the per-date breakdown, so keep that under a
        # readable name instead of the positional placeholder.
        record.pop("column_0", None)
        details_url = record.pop("column_0_url", None)
        if details_url:
            record["details_url"] = details_url
        if any(value for value in record.values()):
            records.append(record)
    return records


def _parse_label_value_table(table, base_url: str, field_map: dict) -> dict:
    """Parse a block whose rows pair N row-headers with N values.

    The Grunddaten semester row carries two pairs at once (``Semester | SWS``
    against ``SS 2026 | 2.0``), so labels and values are zipped per row rather
    than assumed to be a single pair.
    """
    basic: dict = {}
    for row in table.find_all("tr"):
        headers = [normalize_ws(h.get_text(" ")) for h in row.find_all("th")]
        cells = row.find_all("td")
        for label, cell in zip(headers, cells):
            if not label:
                continue
            key = field_map.get(label, _slugify(label))
            basic[key] = normalize_ws(cell.get_text(" "))
            links = _cell_links(cell, base_url)
            if links:
                basic[f"{key}_url"] = links[0]["url"]
    return basic


def parse_event_page(html: str, url: str) -> dict:
    """Parse a timetable event sub-page into structured sections.

    Every section is optional — plenty of events have no assigned person or no
    study programmes — so missing tables yield empty lists rather than an error.
    """
    soup = BeautifulSoup(html, "lxml")

    result = {
        "parsed": False,
        "parse_error": None,
        "page_title": normalize_ws(soup.title.get_text(" ")) if soup.title else None,
        "heading": normalize_ws(soup.h1.get_text(" ")) if soup.h1 else None,
        "basic_data": {},
        "schedule_groups": [],
        "assigned_persons": [],
        "study_programmes": [],
        "belongs_to_modules": [],
        "institutions": [],
        "additional_information": {},
        "unmapped_sections": [],
    }

    tables = soup.find_all("table")
    for table in tables:
        summary = table.get("summary")
        caption = table.find("caption")
        caption_text = normalize_ws(caption.get_text(" ")) if caption else None

        if summary == EVENT_SECTION_BASIC:
            result["basic_data"] = _parse_label_value_table(
                table, url, EVENT_BASIC_FIELD_MAP
            )
        elif summary == EVENT_SECTION_CONTENT:
            result["additional_information"] = _parse_label_value_table(
                table, url, EVENT_CONTENT_FIELD_MAP
            )
        elif summary == EVENT_SECTION_DATES:
            # One table per group; the group name lives in the caption, e.g.
            # "Termine Gruppe: 1-Gruppe". An event can carry several groups.
            group = caption_text or ""
            group = re.sub(r"^Termine\s+Gruppe:\s*", "", group).strip()
            result["schedule_groups"].append(
                {
                    "group": group or None,
                    "dates": _rows_as_dicts(table, url, EVENT_DATE_COLUMN_MAP),
                }
            )
        elif summary == EVENT_SECTION_PERSONS:
            result["assigned_persons"] = _rows_as_dicts(
                table, url, EVENT_PERSON_COLUMN_MAP
            )
        elif summary == EVENT_SECTION_PROGRAMMES:
            result["study_programmes"] = _rows_as_dicts(
                table, url, EVENT_PROGRAMME_COLUMN_MAP
            )
        elif summary == EVENT_SECTION_MODULES:
            result["belongs_to_modules"] = _rows_as_dicts(
                table, url, EVENT_MODULE_COLUMN_MAP
            )
        elif summary == EVENT_SECTION_INSTITUTIONS:
            # This one has no header row at all — a single unlabelled column.
            result["institutions"] = _rows_as_dicts(table, url, {}, default_key="name")
        elif caption_text:
            # A captioned table we have no mapping for is real page content, so
            # keep it verbatim rather than discarding it.
            columns, rows = _table_matrix(table, url)
            result["unmapped_sections"].append(
                {
                    "summary": summary,
                    "caption": caption_text,
                    "columns": columns,
                    "rows": [[cell["text"] for cell in row] for row in rows],
                }
            )

    # A real event page always renders Grunddaten; anything else is an error page
    # served with a 200, which we must not record as a successfully parsed event.
    if result["basic_data"]:
        result["parsed"] = True
    else:
        result["parse_error"] = "No 'Grunddaten' section found (error or expired page)."

    result["veranstid"] = extract_veranstid(url)
    return result

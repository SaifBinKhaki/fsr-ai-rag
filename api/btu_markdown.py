"""Render the parsed module/event dicts from ``btu_parser`` as markdown.

The markdown corpus in ``scraped_data/`` is what Qdrant is built from, so this
is the text the RAG answers are grounded in. It used to come from running an
HTML-to-markdown converter over the whole downloaded page; on qisserver3 that
approach carries too much junk to be worth keeping:

* the page chrome (skip links, "Funktionen:", print/bookmark icons) lands ahead
  of the real content, and
* every event page embeds its QR code as a base64 ``data:`` URI — tens of KB of
  noise that would be chunked and embedded like any other text.

Since ``btu_parser`` already reads these pages exactly — every value out of a
known cell — rendering from that structure instead gives a corpus with no
chrome at all. It is also *deterministic*: the same page always produces byte
-identical markdown, so ``ingest_to_qdrant``'s content hash only changes when
the university actually changed something, rather than whenever a converter
happens to prune differently.

Headings are English whatever the page's own language is. The canonical keys
from ``btu_parser`` are language-neutral, so a German and an English module get
the same skeleton and only the values differ.
"""

# (canonical field key, heading) in the order a reader would want them. Ordering
# here rather than following dict order keeps output stable across pages that
# happen to omit a field.
MODULE_SUMMARY_FIELDS = [
    ("module_number", "Module Number"),
    ("department", "Department"),
    ("responsible_staff", "Responsible"),
    ("language_of_instruction", "Language of Instruction"),
    ("duration", "Duration"),
    ("frequency_of_offer", "Frequency of Offer"),
    ("credits", "Credits"),
    ("participant_limit", "Participant Limit"),
]

MODULE_SECTIONS = [
    ("learning_outcome", "Learning Outcome"),
    ("contents", "Contents"),
    ("recommended_prerequisites", "Recommended Prerequisites"),
    ("mandatory_prerequisites", "Mandatory Prerequisites"),
    ("forms_of_teaching", "Forms of Teaching and Workload"),
    ("teaching_materials", "Teaching Materials and Literature"),
    ("module_examination", "Module Examination"),
    ("assessment_mode", "Assessment Mode"),
    ("evaluation_of_examination", "Evaluation of Examination"),
    ("study_programmes", "Study Programmes"),
    ("module_components", "Module Components"),
    ("remarks", "Remarks"),
    ("notes", "Notes"),
]

EVENT_BASIC_FIELDS = [
    ("event_number", "Event Number"),
    ("event_type", "Event Type"),
    ("semester", "Semester"),
    ("sws", "SWS"),
    ("expected_participants", "Expected Participants"),
    ("academic_year", "Academic Year"),
    ("hyperlink", "Hyperlink"),
]


def _is_empty(value) -> bool:
    """Whether a parsed value carries nothing worth printing."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


def _lines(value) -> list[str]:
    """A parsed value as markdown lines: scalars inline, lists as bullets.

    ``btu_parser`` returns a bare string for single-valued cells and a list when
    the cell held several items, so both shapes have to render sensibly.
    """
    if isinstance(value, str):
        return [value.strip()]
    if isinstance(value, dict):
        # A link cell: {"title": ..., "url": ...}.
        title = value.get("title") or value.get("name") or ""
        url = value.get("url") or value.get("name_url") or ""
        return [f"- [{title}]({url})" if url else f"- {title}"]
    out = []
    for item in value:
        if isinstance(item, dict):
            out.extend(_lines(item))
        elif str(item).strip():
            out.append(f"- {str(item).strip()}")
    return out


def _section(heading: str, value, level: int = 2) -> list[str]:
    """A ``## heading`` block, or nothing at all when the value is empty."""
    if _is_empty(value):
        return []
    return ["", f"{'#' * level} {heading}", ""] + _lines(value)


def render_event_markdown(event: dict, url: str = "", level: int = 2) -> str:
    """One course event (the qisserver3 timetable entry) as a markdown section.

    `level` is the heading depth, so the same renderer serves both a standalone
    event document and an event nested under its parent module.
    """
    basic = event.get("basic_data") or {}
    heading = event.get("heading") or basic.get("event_type") or "Event"

    out = [f"{'#' * level} {heading}"]
    if url:
        out += ["", f"Source: {url}"]

    facts = [
        f"- **{label}:** {basic[key]}"
        for key, label in EVENT_BASIC_FIELDS
        if not _is_empty(basic.get(key))
    ]
    if facts:
        out += [""] + facts

    for group in event.get("schedule_groups") or []:
        name = group.get("group") or ""
        out += ["", f"{'#' * (level + 1)} Schedule" + (f" — {name}" if name else ""), ""]
        for date in group.get("dates") or []:
            parts = [
                date.get("day"),
                date.get("time"),
                date.get("rhythm"),
                date.get("duration"),
                date.get("room"),
            ]
            line = " | ".join(p for p in parts if p and str(p).strip())
            if line:
                out.append(f"- {line}")

    people = [
        "- " + " — ".join(
            p for p in (person.get("person"), person.get("responsibility")) if p
        )
        for person in event.get("assigned_persons") or []
        if person.get("person")
    ]
    if people:
        out += ["", f"{'#' * (level + 1)} Lecturers / Staff", ""] + people

    programmes = [
        "- " + " | ".join(
            str(p) for p in (
                prog.get("programme"), prog.get("semester"), prog.get("po_version")
            ) if p and str(p).strip()
        )
        for prog in event.get("study_programmes") or []
        if prog.get("programme")
    ]
    if programmes:
        out += ["", f"{'#' * (level + 1)} Study Programmes", ""] + programmes

    modules = [
        f"- {mod.get('module_number', '')} {mod.get('module_title', '')}".rstrip()
        for mod in event.get("belongs_to_modules") or []
    ]
    if modules:
        out += ["", f"{'#' * (level + 1)} Belongs to Modules", ""] + modules

    institutions = [
        f"- {inst.get('name')}"
        for inst in event.get("institutions") or []
        if inst.get("name")
    ]
    if institutions:
        out += ["", f"{'#' * (level + 1)} Institutions", ""] + institutions

    extra = event.get("additional_information") or {}
    for key, value in extra.items():
        if not _is_empty(value):
            out += _section(key.replace("_", " ").title(), value, level=level + 1)

    # Anything the parser met but didn't recognise is still printed, so a
    # layout change on the university side degrades to "unlabelled but present".
    for section in event.get("unmapped_sections") or []:
        out += _section(
            section.get("caption") or "Additional Section",
            section.get("rows") or section.get("text") or "",
            level=level + 1,
        )

    return "\n".join(out).strip()


def render_module_markdown(module: dict, url: str, events: list[dict] | None = None) -> str:
    """A module and its current-semester events as one markdown document.

    `events` are the already-parsed event records linked from this module — the
    same dicts ``scrape_btu`` caches — each appended as a nested section. That
    is what puts lecturer, room and schedule detail into the same chunk
    neighbourhood as the module it belongs to.
    """
    fields = module.get("fields") or {}
    number = fields.get("module_number") or ""
    title = fields.get("module_title") or module.get("heading") or "Module"

    out = [f"# {f'Module {number}: ' if number else ''}{title}".rstrip()]
    out += ["", f"Source: {url}"]

    translated = fields.get("module_title_translated")
    if not _is_empty(translated):
        out += ["", f"*{translated}*"]

    facts = [
        f"- **{label}:** {', '.join(v) if isinstance(v, list) else v}"
        for key, label in MODULE_SUMMARY_FIELDS
        if not _is_empty(v := fields.get(key))
    ]
    if facts:
        out += [""] + facts

    for key, heading in MODULE_SECTIONS:
        out += _section(heading, fields.get(key))

    for field in module.get("unmapped_fields") or []:
        out += _section(
            field.get("label") or "Additional Field", field.get("value") or ""
        )

    for event in events or []:
        out += ["", "---", ""]
        if not event.get("fetched", True):
            # The sub-page never came back. Say so rather than leaving a gap, so
            # a reader (and the RAG) can tell "no events" from "not retrieved".
            listed = event.get("listed_as") or event.get("url", "")
            out += [f"## Event (not retrieved): {listed}", "",
                    f"- **Error:** {event.get('fetch_error', 'unknown')}"]
            continue
        out.append(render_event_markdown(event, event.get("url", ""), level=2))

    return "\n".join(out).strip() + "\n"

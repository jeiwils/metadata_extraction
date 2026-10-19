"""
The MARC 21 record for one book, written as MARCXML - the mapping from the
pipeline's data to MARC fields lives here, so it can be read and changed in
one place.

Used by b_extract_metadata.py; it can also be used on its own:
    record = build_record(metadata, ocr, model="qwen2.5vl:7b")
    write_record(record, Path("book.xml"))

Mapping (fields without data are left out):

    Leader     00000nam a22000003c 4500   new record, language material, monograph,
                                     Unicode, encoding level 3 (abbreviated),
                                     ISBD punctuation omitted
    008        date entered, date type/year, form of item 'o' (online),
               literary form '1' when the pages say it's fiction; unknown
               positions use the fill character '|'
    020 ## $a  ISBN                  $q qualifier (paperback, ebook...) when printed
    022 ## $a  ISSN
    024 7# $a  DOI                   $2 doi
    245 0# $a  title                 $b subtitle; 2nd indicator = leading article
                                     to skip when filing ('The ' = 4)
    250 ## $a  edition
    264 #1 $a  place                 $b publisher  $c year - RDA wording in
                                     brackets when one isn't given
    490 0# $a  series                $v number in the series
    500 ## $a  note                  one field per note
    588 0# $a  source of description (fixed text, SOURCE_NOTE)
    720 1# $a  person   $e author / editor   (uncontrolled name: not checked
    720 2# $a  organisation                   against an authority file)
    883 0# $a  how the metadata was made  $d date  $q agency   fully machine-generated
    883 0# $8  per field: the OCR match score as confidence $c (0-1), linked by
               $8 to the 245, 264 or 720 it describes
    930 ## $a  submission portal
    999 ## $a  review reason (records sent to review only)

What the XML contains is checked against the Library of Congress MARCXML schema
(schema/MARC21slim.xsd) when that file is there - see validate_record.
"""

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

MARC_NAMESPACE = "http://www.loc.gov/MARC21/slim"
SCHEMA_FILE = Path(__file__).parent / "schema" / "MARC21slim.xsd"

# Values that are BL policy rather than MARC - change them here
AGENCY_CODE = "Uk"                 # MARC organization code in 883 $q (Uk = British Library)
ENCODING_LEVEL = "3"               # Leader/17: 3 = abbreviated level
SUBMISSION_TAG, SUBMISSION_TEXT = "930", "submission portal"
REVIEW_TAG = "999"                 # local field for the review reason
SOURCE_NOTE = "Description based on images of the first pages of the digital file."

# Leading articles skipped when filing titles (245 second indicator)
ARTICLES = ("the ", "a ", "an ")

# Words that mark a name as an organisation rather than a person (720 first indicator 2)
ORGANISATION_WORDS = {
    "ltd", "limited", "plc", "inc", "llp", "llc", "company", "corporation", "centre", "center",
    "university", "college", "school", "institute", "foundation", "trust", "society",
    "association", "council", "department", "ministry", "office", "agency", "authority",
    "commission", "committee", "group", "laboratory", "library", "press", "government",
}

ET.register_namespace("", MARC_NAMESPACE)


def _tag(name: str) -> str:
    return f"{{{MARC_NAMESPACE}}}{name}"


class RecordBuilder:
    """Collects MARC fields and writes them, sorted by tag, as one MARCXML <record>."""

    def __init__(self, leader: str):
        self.leader = leader
        self.fields = []        # (tag, element)

    def control(self, tag: str, value: str) -> None:
        field = ET.Element(_tag("controlfield"), tag=tag)
        field.text = value
        self.fields.append((tag, field))

    def data(self, tag: str, ind1: str, ind2: str, *subfields: tuple[str, str]) -> None:
        """A data field; subfields with empty values are left out, and so is a field with none left."""
        subfields = [(code, value) for code, value in subfields if value]
        if not any(code not in "8" for code, _ in subfields):
            return
        field = ET.Element(_tag("datafield"), tag=tag, ind1=ind1, ind2=ind2)
        for code, value in subfields:
            ET.SubElement(field, _tag("subfield"), code=code).text = value
        self.fields.append((tag, field))

    def element(self) -> ET.Element:
        record = ET.Element(_tag("record"))
        ET.SubElement(record, _tag("leader")).text = self.leader
        for _, field in sorted(self.fields, key=lambda f: f[0]):   # stable: same tags keep their order
            record.append(field)
        ET.indent(record)
        return record


def nonfiling_characters(title: str) -> str:
    """245 second indicator: how many characters of a leading article to skip ('The ' -> '4')."""
    for article in ARTICLES:
        if title.lower().startswith(article):
            return str(len(article))
    return "0"


def is_organisation(name: str) -> bool:
    """True for 'Fera Science Ltd', False for 'Henwick, Mark'."""
    words = re.findall(r"[a-z]+", name.lower())
    return "," not in name or any(w in ORGANISATION_WORDS for w in words)


def fixed_field_008(metadata: dict, fiction: bool, today: datetime) -> str:
    """The 40 characters of the 008 field for a book, filled as far as the data allows."""
    year = metadata["date"]
    positions = (
        today.strftime("%y%m%d")            # 00-05 date entered on file
        + ("s" + year if year else "nuuuu")  # 06 date type, 07-10 year
        + "    "                             # 11-14 second date
        + "xx "                              # 15-17 place of publication: not coded
        + "||||"                             # 18-21 illustrations
        + "|"                                # 22 target audience
        + "o"                                # 23 form of item: online
        + "||||"                             # 24-27 nature of contents
        + "|||"                              # 28 government publication, 29 conference, 30 festschrift
        + "|"                                # 31 index
        + " "                                # 32 undefined
        + ("1" if fiction else "|")          # 33 literary form: 1 = fiction
        + "|"                                # 34 biography
        + "|||"                              # 35-37 language: not coded
        + "|"                                # 38 modified record
        + "d"                                # 39 cataloguing source: other
    )
    assert len(positions) == 40, len(positions)
    return positions


def build_record(metadata: dict, ocr: dict | None, model: str, review_reason: str = "",
                 generated: datetime | None = None) -> ET.Element:
    """
    One book's MARC record (see the mapping at the top of this module).

    metadata - tidy_answer's fields; ocr - check_against_ocr's findings, or None;
    review_reason - why the book needs a cataloguer (empty when it passed).
    """
    generated = generated or datetime.now(timezone.utc)
    ocr = ocr or {}
    scores = ocr.get("scores", {})
    author_scores = ocr.get("author_scores") or [None] * len(metadata["author"])
    editors = ocr.get("editors", set())
    ids = ocr.get("identifiers", {"isbn": {}, "issn": [], "doi": ""})
    process = f"metadata extracted using Ollama with {model}"
    day = generated.strftime("%Y%m%d")

    record = RecordBuilder(leader=f"00000nam a2200000{ENCODING_LEVEL}c 4500")
    record.control("008", fixed_field_008(metadata, ocr.get("fiction", False), generated))

    # Identifiers
    for isbn, qualifier in ids["isbn"].items():
        record.data("020", " ", " ", ("a", isbn), ("q", qualifier))
    for issn in ids["issn"]:
        record.data("022", " ", " ", ("a", issn))
    record.data("024", "7", " ", ("a", ids["doi"]), ("2", "doi" if ids["doi"] else ""))

    # Each field with an OCR confidence gets a link number; its 883 carries the score
    links = []

    def link(score: float | None) -> str:
        if score is None:
            return ""
        links.append(score)
        return f"{len(links)}\\p"

    # Title, edition, publication
    title_main, subtitle = metadata["title_main"], metadata["subtitle"]
    record.data("245", "0", nonfiling_characters(title_main or subtitle),
                ("8", link(scores.get("title")) if title_main or subtitle else ""), ("a", title_main or subtitle), ("b", subtitle if title_main else ""))
    record.data("250", " ", " ", ("a", metadata["edition"]))
    record.data("264", " ", "1", ("8", link(scores.get("publisher"))),
                ("a", metadata["place"] or "[Place of publication not identified]"),
                ("b", metadata["publisher"] or "[publisher not identified]"),
                ("c", metadata["date"] or "[date of publication not identified]"))

    # Series and notes
    record.data("490", "0", " ", ("a", metadata["series"]), ("v", metadata["series_number"]))
    for note in metadata["notes"]:
        record.data("500", " ", " ", ("a", note))
    record.data("588", "0", " ", ("a", SOURCE_NOTE))

    # Names (uncontrolled)
    for name, score in zip(metadata["author"], author_scores):
        record.data("720", "2" if is_organisation(name) else "1", " ", ("8", link(score)),
                    ("a", name), ("e", "editor" if name in editors else "author"))

    # Provenance: how the record was made, then the confidence of each linked field
    record.data("883", "0", " ", ("a", process), ("d", day), ("q", AGENCY_CODE))
    for number, score in enumerate(links, start=1):
        record.data("883", "0", " ", ("8", f"{number}\\p"), ("a", process), ("c", f"{score:.2f}"),
                    ("d", day), ("q", AGENCY_CODE))

    # Local fields
    record.data(SUBMISSION_TAG, " ", " ", ("a", SUBMISSION_TEXT))
    record.data(REVIEW_TAG, " ", " ", ("a", review_reason))
    return record.element()


def validate_record(record: ET.Element) -> None:
    """
    Check the record against the Library of Congress MARCXML schema, if
    schema/MARC21slim.xsd exists (download it from
    https://www.loc.gov/standards/marcxml/schema/MARC21slim.xsd). Raises
    ValueError naming the first problem. Without the schema file, does nothing.
    """
    if not SCHEMA_FILE.is_file():
        return
    from lxml import etree        # only needed for validation
    schema = _schema_cache.get("schema")
    if schema is None:
        schema = _schema_cache["schema"] = etree.XMLSchema(etree.parse(str(SCHEMA_FILE)))
    document = etree.fromstring(ET.tostring(record, encoding="utf-8"))
    if not schema.validate(document):
        raise ValueError(f"MARC record not valid against MARC21slim.xsd: {schema.error_log.last_error}")


_schema_cache = {}


def write_record(record: ET.Element, xml_file: Path) -> None:
    """Validate the record and write it, via a temporary file so a crash never leaves half a file."""
    validate_record(record)
    temp_file = xml_file.with_suffix(".xml.part")
    ET.ElementTree(record).write(temp_file, encoding="utf-8", xml_declaration=True)
    temp_file.replace(xml_file)
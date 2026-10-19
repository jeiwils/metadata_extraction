"""
Module 2: page images -> MARC 21 records (MARCXML).

Reads a folder holding one sub-folder of page images per book - as made by
a_extract_pages.py, or scans/photos you put there yourself:

    <pages>/<book name>/p01.png, p02.png ...    (or IMG_0001.jpg, ... - any image
                                                  files, in file-name order)

For each book (see assess_book and process_book):
    1. check_page_images() - drops blank pages (e.g. the back of a half-title) and
                             measures sharpness. A book with only blank pages goes
                             to review without asking the model.
    2. read_metadata()     - the vision model reads the page images and returns
                             title, subtitle, authors, edition, year, place,
                             publisher, series and notes.
                             With agreement_check on, it reads them a second time.
    3. check_against_ocr() - with ocr_check on, a separate OCR program reads the
                             same pages; each field the model gave is scored 0-1
                             for how well the page text backs it up; the big text
                             on the title page and the line after the title must
                             be in the record; editors and fiction are noted.
    4. decide_outcome()    - checks the answer: pass, or review with the reasons.
    5. save_result()       - writes the book's MARC record (see marc_record.py):
                             pass:   <output>/processed/<book name>.xml
                             review: <output>/review/<book name>.xml, with the
                             reason in field 999. If the book folder has a source.txt
                             (written by a_extract_pages.py), the original document is
                             copied next to its XML, for High-Wire.

Every book gets a row in <output>/manifest.csv. A run carries on where the last
one stopped: books already in the manifest are skipped, except ones that errored.

Usage:
    python b_extract_metadata.py configs/<settings>.toml

Settings (all required):
    [metadata]
    pages = '...'                # folder with one sub-folder of page images per book
    output = '...'               # where processed/, review/ and manifest.csv go
    model = 'qwen2.5vl:7b'       # Ollama model name (see: ollama list)
    prompt = '../prompts/updated_extraction_prompt.txt'   # relative to the settings file
    num_ctx = 8192               # model context window; images use many tokens each
    max_image_side = 1024        # page images are shrunk to this many pixels on the
                                 # long side before going to the model
    blank_page_ink = 0.0005      # share of dark pixels below which a page counts as blank
                                 # (a sparse half-title measured 0.002, so keep this low)
    min_sharpness = 100.0        # sharpness score below which images count as blurry
    agreement_check = true       # true: read every book twice and send it to review if
                                 # the two readings disagree (doubles the model time)
    ocr_check = true             # true: check the answer against an OCR reading of the pages
                                 # (needs: pip install rapidocr_onnxruntime)
    ocr_review_below = 0.7       # send to review if the title or publisher scores below this
                                 # (0 = record the scores but never send to review because of them)

    [run]                        # shared with a_extract_pages.py
    limit = 0                    # 0 = everything; e.g. 20 for a test run
    random_sample = false        # true: a fixed random sample instead of the first N
    overwrite = false            # true: redo books already in the manifest
"""

import csv
import json
import re
import shutil
import sys
import time
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
import ollama

import marc_record
from z_utils import Config, items_to_process, natural_sort_key

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# The answer the model must give - Ollama holds it to exactly these fields. The length
# limits stop a model that gets stuck repeating itself inside a field (seen in testing:
# a field that ran on until the reply was cut off); tidy_answer applies the real limits.
METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "maxLength": 300},
        "subtitle": {"type": "string", "maxLength": 300},
        "author": {"type": "array", "maxItems": 40, "items": {"type": "string", "maxLength": 150}},
        "edition": {"type": "string", "maxLength": 100},
        "date": {"type": "string", "maxLength": 40},
        "place": {"type": "string", "maxLength": 100},
        "publisher": {"type": "string", "maxLength": 200},
        "series": {"type": "string", "maxLength": 200},
        "series_number": {"type": "string", "maxLength": 40},
        "notes": {"type": "array", "maxItems": 3, "items": {"type": "string", "maxLength": 200}},
    },
    "required": ["title", "subtitle", "author", "edition", "date", "place", "publisher",
                 "series", "series_number", "notes"],
}

MANIFEST_COLUMNS = [
    "book", "outcome", "reason", "title", "pages_used", "max_ink", "max_sharpness",
    "ocr_chars", "ocr_title", "ocr_author", "ocr_date", "ocr_publisher", "isbn", "issn", "doi",
    "error", "model", "timestamp",
]

# Where each outcome's files go. Everything that needs a person goes to review/.
OUTCOME_FOLDERS = {"pass": "processed", "review": "review", "blank": "review"}

# The model's answer needs a few hundred tokens at most. The cap makes a model that
# gets stuck (endless spaces or repetition inside the answer) give up in seconds.
MAX_REPLY_TOKENS = 1024


# Words that mark a name as an organisation rather than a person
CORPORATE_WORDS = {
    "ltd", "limited", "plc", "inc", "llp", "llc", "co", "company", "corporation",
    "centre", "center", "university", "college", "school", "institute", "institution",
    "foundation", "trust", "society", "association", "council", "department",
    "ministry", "office", "agency", "authority", "commission", "committee", "group",
    "network", "programme", "program", "project", "service", "services", "board",
    "bank", "library", "research", "partnership", "alliance", "federation", "union",
    "organisation", "organization", "consortium", "collaboration", "team", "unit",
    "laboratory", "laboratories", "press", "publishing", "publishers", "government",
}

# A UK postcode, a street, a phone number, an email or a website
ADDRESS_PATTERN = re.compile(
    r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b"
    r"|\b(street|road|lane|avenue|square|house|floor)\b"
    r"|\btel\b|@|www\.|https?://",
    re.IGNORECASE,
)

# A series name rather than a publisher: 'Warwick Economics Research Papers'
SERIES_PATTERN = re.compile(
    r"\b(papers?|series|working paper|discussion paper|technical report|briefing)\b",
    re.IGNORECASE,
)




# ================================================================ settings and input

def read_settings(settings_file: str | Path) -> dict:
    """
    Read and check this module's settings.
    """
    config = Config(settings_file)
    pages, output = config.path("metadata", "pages"), config.path("metadata", "output")
    prompt = config.path("metadata", "prompt")
    if not pages.is_dir():
        raise SystemExit(f"Pages folder not found: {pages}")
    if pages in output.parents:
        raise SystemExit(f"Output folder {output} is inside the pages folder {pages}")
    if not prompt.is_file():
        raise SystemExit(f"Prompt file not found: {prompt}")
    return {
        "settings_file": config.file,
        "pages": pages,
        "output": output,
        "model": config.value("metadata", "model"),
        "prompt": prompt.read_text(encoding="utf-8").strip(),
        "num_ctx": config.value("metadata", "num_ctx"),
        "max_image_side": config.value("metadata", "max_image_side"),
        "blank_page_ink": config.value("metadata", "blank_page_ink"),
        "min_sharpness": config.value("metadata", "min_sharpness"),
        "agreement_check": config.value("metadata", "agreement_check"),
        "ocr_check": config.value("metadata", "ocr_check"),
        "ocr_review_below": config.value("metadata", "ocr_review_below"),
        "limit": config.value("run", "limit"),
        "random_sample": config.value("run", "random_sample"),
        "overwrite": config.value("run", "overwrite"),
    }


def page_images(book: Path) -> list[Path]:
    """
    A book's page images, in file-name order ('IMG_2' before 'IMG_10').
    """
    return sorted((p for p in book.iterdir() if p.suffix.lower() in IMAGE_TYPES), key=natural_sort_key)


def find_books(pages_folder: Path) -> list[Path]:
    """
    Every sub-folder of the pages folder that holds page images (one per book).
    """
    return sorted(folder for folder in pages_folder.iterdir()
                  if folder.is_dir() and not folder.name.endswith(".part") and page_images(folder))


# ================================================================ working with images

def load_image(path: Path, flags: int) -> np.ndarray:
    """
    Read an image file. (Goes via np.fromfile because OpenCV's own imread fails on
    Windows paths containing accented characters, which book titles often have.)
    """
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"Could not read image {path.name}")
    return image


def measure_page(path: Path) -> dict:
    """
    How much of the page is printed on ('ink': share of dark pixels, near 0 on a
    blank page) and how sharp it is ('sharpness': low when blurry).
    """
    grey = load_image(path, cv2.IMREAD_GRAYSCALE)
    return {
        "ink": float((grey < 128).mean()),
        "sharpness": float(cv2.Laplacian(grey, cv2.CV_64F).var()),
    }


def crop_empty_margins(image: np.ndarray) -> np.ndarray:
    """
    Cut away the empty margins around what is printed on the page. (The model's
    work grows with image area, so margins cost time for nothing - and once
    they're gone, shrinking the page to max_image_side leaves the text larger.)

    'Printed' means differing from the page's background colour in either
    direction, so light text on a dark cover is kept as well as dark on light.
    """
    # Find every pixel that differs clearly from the background colour
    background = np.median(image.reshape(-1, 3), axis=0)
    differs = (np.abs(image.astype(np.int16) - background).max(axis=2) > 30).astype(np.uint8)
    points = cv2.findNonZero(differs)
    if points is None:
        return image

    # Crop to the box around them, with a small margin
    x, y, w, h = cv2.boundingRect(points)
    pad = int(0.02 * max(image.shape[:2]))
    top, left = max(0, y - pad), max(0, x - pad)
    return image[top:y + h + pad, left:x + w + pad]


def image_for_model(path: Path, max_side: int) -> bytes:
    """
    The page as PNG bytes for the model: margins cropped, shrunk to max_side pixels.
    """
    image = crop_empty_margins(load_image(path, cv2.IMREAD_COLOR))
    scale = max_side / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, png = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Could not convert image {path.name}")
    return png.tobytes()


# ================================================================ asking the model

def check_model_installed(model: str) -> None:
    """Stop at the start, not on the first book, if Ollama isn't running or the model is missing."""
    try:
        ollama.show(model)
    except ollama.ResponseError:
        raise SystemExit(f"Model '{model}' is not available in Ollama - run: ollama pull {model}")
    except Exception as e:
        raise SystemExit(f"Can't reach Ollama ({e}). Is it running?")


def ask_model(pages: list[tuple[int, Path]], settings: dict, max_side: int | None = None) -> dict:
    """
    Send the model the page images and the instructions; return its answer.

    Tries twice. If the first answer is empty, cut off or not valid JSON, the
    second try adds a little randomness - at temperature 0 the model would
    repeat exactly the same mistake.
    """
    model, num_ctx = settings["model"], settings["num_ctx"]
    images = [image_for_model(path, max_side or settings["max_image_side"]) for _, path in pages]
    page_numbers = ", ".join(str(number) for number, _ in pages)

    error = None
    for temperature in (0, 0.4):
        # Ask. The instructions go in a system message: they're the same for every
        # book, so Ollama can reuse its work on them instead of redoing it each time.
        response = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": settings["prompt"]},
                      {"role": "user", "images": images,
                       "content": f"The images are pages {page_numbers} of the book, in order."}],
            format=METADATA_SCHEMA,
            # Ollama's default context window is small; page images can silently overflow it
            options={"temperature": temperature, "num_ctx": num_ctx, "num_predict": MAX_REPLY_TOKENS},
        )
        tokens = response.prompt_eval_count or 0
        details = (f"{model}: request used {tokens} of {num_ctx} context tokens, "
                   f"answer {response.eval_count or 0} tokens, stop reason '{response.done_reason}'")

        # A request that fills the whole context window was almost certainly cut
        # short, so the model didn't see all the pages - trying again won't help
        if tokens >= 0.95 * num_ctx:
            raise ValueError(f"input probably truncated - {details}. "
                             f"Lower max_image_side or raise num_ctx")

        # A usable answer?
        answer = (response.message.content or "").strip()
        try:
            return json.loads(answer)
        except json.JSONDecodeError:
            error = ValueError(f"no valid answer after 2 tries - {details}; "
                               f"answer began {answer[:200]!r}")

        # An empty answer with no status means Ollama never ran the request
        # (seen in testing) - give it a moment before trying again
        if not response.done:
            time.sleep(5)
    raise error


def trim_to_sentence(text: str, limit: int) -> str:
    """
    Shorten text to at most `limit` characters, ending at a full stop so it never
    stops mid-sentence. If even the first sentence is too long, cut at a word
    and add '...'.
    """
    if len(text) <= limit:
        return text
    head = text[:limit]
    sentence_ends = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", head)]
    if sentence_ends and sentence_ends[-1] >= limit // 3:
        return head[:sentence_ends[-1]]
    return head[:limit - 3].rsplit(" ", 1)[0].rstrip(",;:") + "..."


def tidy_answer(answer: dict) -> dict:
    """
    Tidy the model's answer, enforcing what the schema can't: the date as a
    four-digit year and at most 3 short notes. 'title' is the full title (main
    title and subtitle joined with ': '), which the checks compare against the
    page; 'title_main' and 'subtitle' are kept apart for MARC 245 $a and $b.
    """
    def text(value) -> str:
        # Collapse line breaks and repeated spaces copied from the page layout
        return " ".join(value.split()) if isinstance(value, str) else ""

    title_main, subtitle = text(answer.get("title")), text(answer.get("subtitle"))
    year = re.search(r"\b(1[4-9]\d\d|20\d\d)\b", text(answer.get("date")))
    return {
        "title": f"{title_main}: {subtitle}" if title_main and subtitle else title_main or subtitle,
        "title_main": title_main,
        "subtitle": subtitle,
        "author": [text(a) for a in answer.get("author", []) if text(a)],
        "edition": text(answer.get("edition")),
        "date": year.group(1) if year else "",
        "place": text(answer.get("place")),
        "publisher": text(answer.get("publisher")),
        "series": text(answer.get("series")),
        "series_number": text(answer.get("series_number")),
        "notes": [trim_to_sentence(text(n), 200) for n in answer.get("notes", []) if text(n)][:3],
    }


# ================================================================ OCR cross-check

_ocr_engine = None


def ocr_engine():
    """
    The OCR program, loaded once on first use - so it only needs installing
    when ocr_check is on.
    """
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            raise SystemExit("ocr_check is on but the OCR package is missing - "
                             "run: pip install rapidocr_onnxruntime")
        _ocr_engine = RapidOCR()
    return _ocr_engine


def plain_words(text: str, min_length: int = 1) -> list[str]:
    """Lower-case words without accents or punctuation: 'Género, Ltd.' -> ['genero', 'ltd']."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= min_length]


# Short words that appear in almost any text, so finding them proves nothing
COMMON_WORDS = {"the", "and", "for", "with", "from", "into", "its", "our", "their", "this", "that"}


def word_similarity(word: str, ocr_word: str) -> float:
    """
    How alike a word from the answer and a word OCR read are: 1.0 the same, around
    0.9 one letter off, low unrelated. Words OCR ran together still count
    ('turnover' in 'laborturnover'). Short words must match exactly.
    """
    if word == ocr_word or (len(word) >= 5 and word in ocr_word):
        return 1.0
    if len(word) < 4 or abs(len(word) - len(ocr_word)) > 3:
        return 0.0
    return SequenceMatcher(None, word, ocr_word).ratio()


class OcrReading:
    """
    What OCR read on a book's pages, and how well a field the model gave is
    backed up by it.

    Matching is forgiving, because OCR makes small mistakes and the model can
    read lettering OCR can't: case, accents and punctuation are ignored, a word
    one or two letters off still counts, and the words need not be in order.
    But the words must be found close together, as they'd be printed - not
    scattered across five pages of text, where common words turn up anyway.
    """

    def __init__(self, lines: list[str]):
        self.lines = lines
        self.text = "\n".join(lines)
        self.words = plain_words(self.text)
        self._found = {}

    def places(self, word: str) -> dict[int, float]:
        """Where on the pages a word (or something like it) is: {word position: similarity}."""
        if word not in self._found:
            similarity = {w: word_similarity(word, w) for w in set(self.words)}
            self._found[word] = {i: similarity[w] for i, w in enumerate(self.words) if similarity[w] >= 0.75}
        return self._found[word]

    def phrase_score(self, words: list[str]) -> float:
        """
        Share of the words found together within a short stretch of the page
        text (the phrase's length plus 3 words), at the best-matching place.
        """
        places = [self.places(w) for w in words]
        span = len(words) + 3
        best = 0.0
        for start in sorted({i for found in places for i in found}):
            total = sum(max((s for i, s in found.items() if start <= i < start + span), default=0.0)
                        for found in places)
            best = max(best, total / len(words))
        return best

    def score(self, text: str, min_length: int = 3, ignore: set[str] = frozenset()) -> float | None:
        """
        How well the page text backs up a field: 1.0 = all found, 0 = none found;
        None if the field has no words to check. A title and its subtitle are
        often printed apart, so each part (split at ':', ';' or ' - ') is
        scored on its own and the scores combined by length.
        """
        parts = [[w for w in plain_words(part, min_length) if w not in ignore and w not in COMMON_WORDS]
                 for part in re.split(r"[:;]| [-–] ", text)]
        parts = [part for part in parts if part]
        if not parts:
            return None
        return sum(self.phrase_score(part) * len(part) for part in parts) / sum(len(part) for part in parts)


# Words that say nothing about which publisher it is, so they aren't checked
GENERIC_PUBLISHER_WORDS = {"ltd", "limited", "inc", "plc", "llc", "company",
                           "publishing", "publishers", "publisher", "press"}

# Lines that give the publication year: '© 2024 Jones', 'Copyright 2026', 'First published
# 2025', 'Published March 2026'. OCR often reads © as '@', '(c)' or a lone 'C'. Only at the
# start of a line (or a © anywhere), so 'was published in 2019' in body text doesn't count.
IMPRINT_PATTERN = re.compile(r"©|^\W*(?:[@c]|\(c\))\s|^\W*(?:copyright|(?:first )?published)", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")


def year_on_page(year: str, text: str) -> bool:
    """True if the year is printed in full ('2026') or as the end of a date ('23-Apr-26')."""
    if year in text:
        return True
    short = re.escape(year[2:])
    return re.search(rf"\b\d{{1,2}}[-/.](?:\d{{1,2}}|[A-Za-z]{{3,9}})[-/.]{short}\b", text) is not None


# Large text that is never part of a title, author or publisher: headings, dates, web addresses
HEADING_WORDS = {
    "abstract", "synopsis", "contents", "foreword", "preface", "summary", "introduction",
    "acknowledgements", "executive", "chapter", "research", "article", "report", "page",
    "edition", "volume", "issue", "www", "com", "org",
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "january", "february", "march", "april", "june", "july", "august", "september",
    "october", "november", "december",
}

# Title-page display type: lines at least this share of the tallest line's height...
LARGE_TEXT_SHARE = 0.7
# ...and at least this many times the height of a typical line on the page. Pages
# without text this much bigger than the rest (e.g. dense report pages) aren't checked.
DISPLAY_TEXT_SIZE = 1.6
# A line after the title at least this share of the title's height may be its subtitle
SUBTITLE_SIZE = 0.75


def line_height(box) -> float:
    """Height in pixels of an OCR text line, from its four corner points."""
    return ((box[3][1] - box[0][1]) + (box[2][1] - box[1][1])) / 2


def words_in_record(metadata: dict) -> set[str]:
    """Every word of the record's title, authors and publisher, plus acronyms of the publisher ('LSE')."""
    known = set(plain_words(" ".join([metadata["title"], metadata["publisher"], *metadata["author"]]), 3))
    publisher = [w for w in plain_words(metadata["publisher"]) if w not in {"of", "for", "and", "the", "in", "on"}]
    known |= {"".join(w[0] for w in publisher[i:j])
              for i in range(len(publisher)) for j in range(i + 2, len(publisher) + 1)}
    return known


def share_in_record(text: str, known: set[str]) -> float | None:
    """
    Share of an OCR line's words that are in `known` (allowing OCR slips); None
    if the line has no words worth checking (only headings, dates or numbers).
    """
    words = [w for w in plain_words(text, 3)
             if w not in COMMON_WORDS and w not in HEADING_WORDS and not w.isdigit()]
    if not words:
        return None
    found = sum(1 for w in words if any(max(word_similarity(w, k), word_similarity(k, w)) >= 0.8 for k in known))
    return found / len(words)


def find_title_page(pages: list[list[tuple]], title: str) -> list[tuple] | None:
    """The OCR lines of the page on which the record's title matches best (None if it matches nowhere)."""
    best_score, title_page = 0.0, None
    for lines in pages:
        score = OcrReading([text for _, text in lines]).score(title) or 0.0
        if score > best_score:
            best_score, title_page = score, lines
    return title_page if best_score >= 0.5 else None


def large_text_not_in_record(title_page: list[tuple], metadata: dict) -> list[str]:
    """
    Title check 1, by text size. On the title page, the biggest text is almost
    always the title (or the author or publisher). Any line of big text whose
    words are mostly NOT in the record's title, authors or publisher suggests the
    model left part of the title out - e.g. 'Fapas - Food Chemistry Proficiency
    Test Report 14286' above 'Fatty Acids in Olive Oil'. Returns those lines.
    """
    # The page's display type: lines much taller than the typical line
    sized = [(line_height(box), text) for box, text in title_page if len(re.findall(r"[A-Za-z]", text)) >= 3]
    if not sized:
        return []
    tallest = max(h for h, _ in sized)
    typical = sorted(h for h, _ in sized)[len(sized) // 2]
    large = [text for h, text in sized if h >= LARGE_TEXT_SHARE * tallest and h >= DISPLAY_TEXT_SIZE * typical]

    # Which of those lines are mostly missing from the record?
    known = words_in_record(metadata)
    return [text for text in large if (share := share_in_record(text, known)) is not None and share < 0.5]


def line_after_title_not_in_record(title_page: list[tuple], metadata: dict) -> str | None:
    """
    Title check 2, for a missing subtitle. Subtitles are printed smaller than the
    title, so check 1 misses them - but they sit on the line straight after it.
    If the line after the record's title is nearly as big as the title's last
    line, isn't a date, and is mostly NOT in the record, it's probably a subtitle
    the model left out ('Fish Identification' under 'Fapas - Food Chemistry
    Proficiency Test Report 29113'). Only applied when the title is at least as
    big as the page's typical text, so dense report pages aren't checked.
    """
    # The page's lines from top to bottom, with their heights
    lines = sorted((box[0][1], line_height(box), text) for box, text in title_page
                   if len(re.findall(r"[A-Za-z]", text)) >= 3)
    if not lines:
        return None
    typical = sorted(h for _, h, _ in lines)[len(lines) // 2]

    # The last line that is part of the record's title
    title_words = set(plain_words(metadata["title"], 3))
    in_title = [i for i, (_, _, text) in enumerate(lines) if (share_in_record(text, title_words) or 0) >= 0.6]
    if not in_title or in_title[-1] + 1 >= len(lines):
        return None
    title_height = lines[in_title[-1]][1]
    if title_height < typical:
        return None

    # The line after it: nearly as big, and not in the record?
    _, height, text = lines[in_title[-1] + 1]
    share = share_in_record(text, words_in_record(metadata))
    if share is not None and share < 0.5 and SUBTITLE_SIZE * title_height <= height <= 1.05 * title_height:
        return text
    return None


# "Edited by" (or "Editors:", "(eds)") - the names that follow are editors, not authors
EDITED_PATTERN = re.compile(r"\bedited\s+by\b|\(eds?\.?\)|\beditors?\s*:", re.IGNORECASE)

# Statements printed on the copyright page of novels, short stories and poetry
FICTION_PATTERN = re.compile(r"work of fiction|products? of the author.s imagination|used fictitiously",
                             re.IGNORECASE)


def editors_among_authors(text: str, authors: list[str]) -> set[str]:
    """
    The record's authors who are really editors: named straight after "Edited by"
    (on the same line or the next). Names after "Edited by" that aren't in the
    record - e.g. a novel's desk editor - are ignored.
    """
    editors = set()
    for match in EDITED_PATTERN.finditer(text):
        following = " ".join(text[match.end():].lstrip(" :\n").split("\n")[:2])
        words = set(plain_words(following))
        for name in authors:
            surname = plain_words(name.split(",")[0])
            if surname and surname[-1] in words:
                editors.add(name)
    return editors


# Identifiers. An ISBN or ISSN only counts on a line labelled "ISBN"/"ISSN" (so phone numbers
# and report numbers don't), and only if its check digit is right (so OCR misreads don't).
ISBN_PATTERN = re.compile(r"(?<![\dX])((?:97[89][\s-]?)?(?:\d[\s-]?){9}[\dX])(?![\dX])", re.IGNORECASE)
ISSN_PATTERN = re.compile(r"(?<!\d)(\d{4})[\s-]?(\d{3}[\dX])(?![\dX])", re.IGNORECASE)
DOI_PATTERN = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>,;]+)", re.IGNORECASE)
ISBN_QUALIFIER = re.compile(r"\b(hardback|hardcover|paperback|pbk|hbk|ebook|e-book|epub|pdf|kindle|print|online)\b",
                            re.IGNORECASE)


def valid_isbn(digits: str) -> bool:
    """True if a 10- or 13-character ISBN has the right check digit."""
    if len(digits) == 13 and digits.isdigit():
        return sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(digits)) % 10 == 0
    if len(digits) == 10 and digits[:9].isdigit() and (digits[9].isdigit() or digits[9] == "X"):
        return sum((10 - i) * (10 if d == "X" else int(d)) for i, d in enumerate(digits)) % 11 == 0
    return False


def valid_issn(digits: str) -> bool:
    """True if an 8-character ISSN has the right check digit."""
    total = sum((8 - i) * int(d) for i, d in enumerate(digits[:7]))
    check = (11 - total % 11) % 11
    return digits[7] == ("X" if check == 10 else str(check))


def isbn_qualifier(text: str) -> str:
    """'paperback', 'ebook' etc. if printed in this bit of text (OCR often reads 'ebook' as 'eb0ok')."""
    match = ISBN_QUALIFIER.search(text.replace("0", "o"))
    return match.group(1).lower() if match else ""


def find_identifiers(lines: list[str]) -> dict:
    """
    ISBNs (with qualifiers like 'paperback' or 'ebook' when printed beside them),
    ISSNs and the document's DOI, from the OCR text of the pages. Only identifiers
    that pass these checks are kept, so a misread digit means none rather than a
    wrong one.
    """
    lines = [re.sub(r"[\u2010-\u2015]", "-", line) for line in lines]   # typographic dashes -> '-'
    isbns, issns, dois = {}, [], []
    for i, line in enumerate(lines):
        # ISBNs, on lines labelled ISBN (OCR sometimes drops the I). The number can
        # wrap onto the next line, so look there too if the line itself has none.
        if re.search(r"\bI?SBN\b", line, re.IGNORECASE):
            text = line if ISBN_PATTERN.search(line) else " ".join(lines[i:i + 2])
            matches = list(ISBN_PATTERN.finditer(text))
            for n, match in enumerate(matches):
                digits = re.sub(r"[\s-]", "", match.group(1)).upper()
                if valid_isbn(digits) and digits not in isbns:
                    # 'ISBN 978... (paperback)', or 'Ebook ISBN 978...'
                    after = text[match.end():matches[n + 1].start() if n + 1 < len(matches) else None]
                    before = text[matches[n - 1].end() if n else 0:match.start()]
                    isbns[digits] = isbn_qualifier(after) or isbn_qualifier(before)

        # ISSNs, on lines labelled ISSN
        if re.search(r"\bISSN", line, re.IGNORECASE):
            for match in ISSN_PATTERN.finditer(" ".join(lines[i:i + 2])):
                digits = (match.group(1) + match.group(2)).upper()
                issn = f"{digits[:4]}-{digits[4:]}"
                if valid_issn(digits) and issn not in issns:
                    issns.append(issn)

        # DOIs; a DOI split over two lines ('https://doi.org/10.3310/' + 'LEXB9006') is joined up
        joined = line + lines[i + 1] if line.rstrip().endswith("/") and i + 1 < len(lines) else line
        for match in DOI_PATTERN.finditer(joined):
            doi = match.group(1).rstrip(".)]")
            dois.append((bool(re.match(r"\W*doi\b", line, re.IGNORECASE)), doi))

    # The document's own DOI: one on a line starting "DOI", else the one printed most often
    # (the pages may also cite other works by their DOIs)
    doi = ""
    if dois:
        labelled = [d for is_label, d in dois if is_label]
        doi = labelled[0] if labelled else max({d for _, d in dois}, key=[d for _, d in dois].count)
    return {"isbn": isbns, "issn": issns, "doi": doi}


def check_against_ocr(pages: list[tuple[int, Path]], metadata: dict) -> dict:
    """
    Step 3: read the pages with OCR, and check the model's answer against them:
      - score each field for how well the page text backs it up - 1.0 = found as
        given, around 0.9 = found with a letter or two different, low = not found
        (title, authors, year and publisher)
      - collect the years in copyright and 'published' lines
      - look for title-page text missing from the title: big text
        (large_text_not_in_record) and a subtitle (line_after_title_not_in_record)
      - note which authors are editors, and whether it's fiction
      - find the ISBNs, ISSNs and DOI (find_identifiers)
    """
    # Read every page, keeping only lines OCR is reasonably sure of, with their position
    engine = ocr_engine()
    page_lines = []
    for _, path in pages:
        result, _ = engine(load_image(path, cv2.IMREAD_COLOR))
        page_lines.append([(box, text) for box, text, confidence in (result or []) if float(confidence) >= 0.5])
    lines = [text for page in page_lines for _, text in page]
    reading = OcrReading(lines)

    # Score each author by their surname (or the whole name, for an organisation)
    author_scores = [reading.score(name.split(",")[0], min_length=2) for name in metadata["author"]]
    known = [s for s in author_scores if s is not None]

    # Years in copyright or 'published' lines (the year can be on the next line)
    stated_years = set()
    for i, line in enumerate(lines):
        if IMPRINT_PATTERN.search(line):
            stated_years.update(YEAR_PATTERN.findall(" ".join(lines[i:i + 2])))

    # Title-page text missing from the title
    title_page = find_title_page(page_lines, metadata["title"])

    return {
        "characters": len(reading.text),
        "scores": {
            "title": reading.score(metadata["title"]),
            "author": sum(known) / len(known) if known else None,
            "date": (1.0 if year_on_page(metadata["date"], reading.text) else 0.0) if metadata["date"] else None,
            "publisher": reading.score(metadata["publisher"], ignore=GENERIC_PUBLISHER_WORDS),
        },
        "author_scores": author_scores,
        "stated_years": stated_years,
        "large_text_missing": large_text_not_in_record(title_page, metadata) if title_page else [],
        "subtitle_missing": line_after_title_not_in_record(title_page, metadata) if title_page else None,
        "editors": editors_among_authors(reading.text, metadata["author"]),
        "fiction": FICTION_PATTERN.search(" ".join(lines)) is not None,
        "identifiers": find_identifiers(lines),
    }


# ================================================================ checking the answer

def simplify(text: str) -> str:
    """Lower case, punctuation turned into spaces - for comparing text loosely."""
    return re.sub(r"[^\w]+", " ", text.lower()).strip()


def surnames(authors: list[str]) -> set[str]:
    """The last word of each surname: 'van Rens, Thijs' -> 'rens'."""
    found = set()
    for name in authors:
        words = simplify(name.split(",")[0] if "," in name else name).split()
        if words:
            found.add(words[-1])
    return found


def part_of(short: str, full: str) -> bool:
    """True if `short` is a shorter part of `full`: 'Report 12' and 'Report 12: Water Quality'."""
    short, full = simplify(short), simplify(full)
    return bool(short) and len(short) < len(full) and short in full


def differences(first: dict, second: dict) -> list[str]:
    """
    The fields on which two readings of the same pages really differ, each with
    both values - e.g. "publisher ('ESCoE' vs 'Economic Statistics Centre of
    Excellence')" - so a cataloguer can see what to choose between.
    Small differences (case, punctuation, a fuller form of the same name) don't count.
    """
    def similar(a: str, b: str, threshold: float) -> bool:
        a, b = simplify(a), simplify(b)
        return a == b or SequenceMatcher(None, a, b).ratio() >= threshold

    def same_title(a: str, b: str) -> bool:
        # One reading including the subtitle (or the series part) and the other
        # leaving it out isn't a disagreement - assess_book keeps the fuller one
        return similar(a, b, 0.9) or part_of(a, b) or part_of(b, a)

    def personal_authors(reading: dict) -> list[str]:
        # An organisation given as author because it's the publisher ('Fera Science
        # Ltd') means the same as no personal author, so it doesn't count as a difference
        publisher = simplify(reading["publisher"])
        return [n for n in reading["author"]
                if not (publisher and (simplify(n) in publisher or publisher in simplify(n)))]

    def same_publisher(a: str, b: str) -> bool:
        # One being a fuller form of the other ('X' vs 'X, University of Y') counts as the same
        sa, sb = simplify(a), simplify(b)
        return similar(a, b, 0.85) or bool(sa and sb and (sa in sb or sb in sa))

    def show(value) -> str:
        shown = ("; ".join(value) if isinstance(value, list) else value) or "(empty)"
        return shown if len(shown) <= 80 else shown[:77] + "..."

    differing = []
    if not same_title(first["title"], second["title"]):
        differing.append("title")
    if surnames(personal_authors(first)) != surnames(personal_authors(second)):
        differing.append("author")
    if first["date"] != second["date"]:
        differing.append("date")
    if not same_publisher(first["publisher"], second["publisher"]):
        differing.append("publisher")
    labels = {"author": "authors", "date": "year"}
    return [f"{labels.get(f, f)} ('{show(first[f])}' vs '{show(second[f])}')" for f in differing]





def looks_like_first_last(name: str) -> bool:
    """True for a person's name written 'First Last' instead of 'Last, First'."""
    if "," in name or any(c in name for c in "®™&()"):
        return False
    words = name.split()
    if not 2 <= len(words) <= 4:
        return False
    if any(w.lower().strip(".") in CORPORATE_WORDS for w in words):
        return False
    return words[0][:1].isupper() and words[-1][:1].isupper()


# ================================================================ the steps for one book

def check_page_images(book: Path, settings: dict) -> tuple[dict, list[tuple[int, Path]]]:
    """
    Step 1: look at the page images, without the model.

    Blank pages (e.g. the back of a half-title) are dropped so the model doesn't
    spend time on them. Returns the book's manifest row - with outcome 'blank'
    if every page is blank - and the pages to send, each with its page number.
    """
    images = page_images(book)
    if not images:
        raise ValueError("no page images found for this book")

    measurements = [measure_page(path) for path in images]
    pages = [(number, path)
             for number, (path, m) in enumerate(zip(images, measurements), start=1)
             if m["ink"] >= settings["blank_page_ink"]]

    row = {
        "book": book.name,
        "pages_used": " ".join(str(number) for number, _ in pages),
        "max_ink": round(max(m["ink"] for m in measurements), 4),
        "max_sharpness": round(max((measurements[n - 1]["sharpness"] for n, _ in pages), default=0.0), 1),
    }
    if not pages:
        row.update(outcome="blank", reason="blank pages")
    return row, pages


def read_metadata(pages: list[tuple[int, Path]], settings: dict, second_reading: bool = False) -> dict:
    """
    Step 2: the model reads the pages. The second reading (agreement check) uses
    images 20% smaller, so it's a genuinely different look at the pages while
    staying repeatable.
    """
    max_side = int(settings["max_image_side"] * 0.8) if second_reading else None
    return tidy_answer(ask_model(pages, settings, max_side=max_side))


def decide_outcome(row: dict, metadata: dict, settings: dict, second_reading: dict | None = None,
                   ocr: dict | None = None) -> tuple[str, str]:
    """
    Step 4: check the answer. Returns ('pass', ...) if everything below holds,
    otherwise ('review', every problem found).

      - the page images are sharp enough to read
      - there is a title and a year, and the year isn't in the future
      - people's names are in 'Last, First' order
      - the publisher is a name, not an address or a series name
      - with agreement_check on: a second reading gives the same title,
        authors, year and publisher
      - with ocr_check on: OCR could read text on the pages; the title and
        publisher score at least ocr_review_below; the big text on the title
        page and the line after the title are in the record (see
        large_text_not_in_record, line_after_title_not_in_record); and if the
        pages have a copyright or 'published' statement, the year is one of its years
    """
    problems = []

    # The images
    if row["max_sharpness"] < settings["min_sharpness"]:
        problems.append("images blurry")

    # Title and year
    if not metadata["title"]:
        problems.append("no title")
    if not metadata["date"]:
        stated = sorted(ocr["stated_years"]) if ocr else []
        problems.append("no year" + (f" (OCR found {', '.join(stated)} in a copyright/published line)"
                                     if stated else ""))
    elif int(metadata["date"]) > datetime.now().year:
        problems.append(f"year {metadata['date']} is in the future")

    # Name order. A name matching the publisher is the issuing body ('Natural England'), not a person.
    publisher = simplify(metadata["publisher"])
    first_last = [name for name in metadata["author"] if looks_like_first_last(name)
                  and not (publisher and (simplify(name) in publisher or publisher in simplify(name)))]
    if first_last:
        problems.append("author not 'Last, First': " + ", ".join(first_last[:2]))

    # Publisher
    if ADDRESS_PATTERN.search(metadata["publisher"]):
        problems.append("publisher includes an address")
    if SERIES_PATTERN.search(metadata["publisher"]):
        problems.append("publisher looks like a series name")

    # Agreement between two readings
    if second_reading is not None and (differing := differences(metadata, second_reading)):
        problems.append("two readings disagree on " + ", ".join(differing))

    # OCR cross-check
    if ocr is not None:
        if ocr["characters"] < 20:
            problems.append("OCR could read almost no text on these pages")
        else:
            for field in ("title", "publisher"):
                score = ocr["scores"][field]
                if score is not None and score < settings["ocr_review_below"]:
                    problems.append(f"{field} not found on the pages by OCR (match {score:.2f})")
            if ocr["large_text_missing"]:
                shown = ", ".join(f"'{t[:60]}'" for t in ocr["large_text_missing"][:3])
                problems.append(f"large text on the title page not in the record: {shown}")
            if ocr["subtitle_missing"]:
                problems.append(f"line after the title not in the record - a missing subtitle? "
                                f"'{ocr['subtitle_missing'][:80]}'")
            stated = ocr["stated_years"]
            if metadata["date"] and stated and metadata["date"] not in stated:
                problems.append(f"year {metadata['date']} differs from the copyright/published "
                                f"year on the pages ({', '.join(sorted(stated))})")

    if problems:
        return "review", "; ".join(problems)
    return "pass", "all checks passed"


def assess_book(book: Path, settings: dict) -> tuple[dict, dict, dict | None]:
    """
    Steps 1-4 for one book, without saving anything. Returns its manifest row
    (including outcome and reason), the metadata, and the OCR scores (None when
    ocr_check is off or every page was blank).
    """
    # 1. Check the page images
    row, pages = check_page_images(book, settings)
    if row.get("outcome") == "blank":
        row["model"] = settings["model"]
        return row, tidy_answer({}), None

    # 2. Read the metadata (twice, with agreement_check on)
    metadata = read_metadata(pages, settings)
    second_reading = read_metadata(pages, settings, second_reading=True) if settings["agreement_check"] else None
    if second_reading and part_of(metadata["title"], second_reading["title"]):
        # the second reading included the subtitle: take its title, in all three forms
        metadata.update({k: second_reading[k] for k in ("title", "title_main", "subtitle")})

    # 3. Check it against OCR
    ocr = check_against_ocr(pages, metadata) if settings["ocr_check"] else None
    if ocr:
        row["ocr_chars"] = ocr["characters"]
        ids = ocr["identifiers"]
        row.update(isbn="; ".join(ids["isbn"]), issn="; ".join(ids["issn"]), doi=ids["doi"])
        for field, score in ocr["scores"].items():
            row[f"ocr_{field}"] = "" if score is None else round(score, 2)

    # 4. Pass or review?
    outcome, reason = decide_outcome(row, metadata, settings, second_reading, ocr)
    row.update(outcome=outcome, reason=reason, title=metadata["title"], model=settings["model"])
    return row, metadata, ocr


# ================================================================ saving

def original_document(book: Path) -> Path | None:
    """The original document, if a_extract_pages.py recorded it in the book's source.txt."""
    note = book / "source.txt"
    if not note.is_file():
        return None
    document = Path(note.read_text(encoding="utf-8").strip())
    if not document.is_file():
        raise ValueError(f"original document not found: {document}")
    return document


def save_result(book: Path, metadata: dict, row: dict, ocr: dict | None, output: Path) -> None:
    """
    Step 5: put the book's MARC record (MARCXML, see marc_record.py) - and a copy
    of its original document, with the same name, when known - into the folder
    for its outcome.
    """
    outcome = row["outcome"]

    # Remove anything an earlier run left in the other folder, so a book whose
    # outcome has changed doesn't end up in both
    target = output / OUTCOME_FOLDERS[outcome]
    target.mkdir(parents=True, exist_ok=True)
    for folder in set(OUTCOME_FOLDERS.values()) - {OUTCOME_FOLDERS[outcome]}:
        for old_file in (output / folder).glob(f"{book.name}.*"):
            old_file.unlink()

    # Copy the original document beside its record, with the same name (temporary file
    # first, so a crash never leaves half a copy)
    document = original_document(book)
    if document:
        content_file = target / (book.name + document.suffix.lower())
        temp_file = content_file.with_name(content_file.name + ".part")
        shutil.copy2(document, temp_file)
        temp_file.replace(content_file)

    review_reason = row["reason"] if outcome != "pass" else ""
    record = marc_record.build_record(metadata, ocr, row.get("model", "none"), review_reason)
    marc_record.write_record(record, target / f"{book.name}.xml")


def process_book(book: Path, settings: dict) -> dict:
    """
    all five steps for one book; returns its manifest row.
    """
    row, metadata, ocr = assess_book(book, settings)
    save_result(book, metadata, row, ocr, settings["output"])
    return row


# ================================================================ running the whole folder

def books_already_done(manifest: Path) -> set[str]:
    """
    books whose latest manifest row is a finished result (anything but an error).
    """
    if not manifest.exists():
        return set()
    latest = {}
    with open(manifest, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            latest[row["book"]] = row["outcome"]
    return {book for book, outcome in latest.items() if outcome != "error"}


def add_to_manifest(manifest: Path, row: dict) -> None:
    """
    append one book's row to manifest.csv, writing the header if the file is new.
    """
    is_new = not manifest.exists()
    with open(manifest, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)





def main(settings_file: str) -> None:
    # read the settings and check the model (and OCR, if used) are there
    settings = read_settings(settings_file)
    output = settings["output"]
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(settings["settings_file"], output / "run_config.toml")  # record of the settings used
    manifest = output / "manifest.csv"
    check_model_installed(settings["model"])
    if settings["ocr_check"]:
        ocr_engine()

    # decide which books to process
    done = set() if settings["overwrite"] else books_already_done(manifest)
    books = items_to_process(find_books(settings["pages"]), settings["limit"], settings["random_sample"])
    if settings["limit"]:
        print(f"[test run] {len(books)} books ({'random sample' if settings['random_sample'] else 'first in order'})")

    # process them one by one, recording each in the manifest
    counts = {"pass": 0, "review": 0, "blank": 0, "error": 0, "skipped": 0}
    for book in books:
        if book.name in done:
            counts["skipped"] += 1
            continue
        try:
            row = process_book(book, settings)
        except Exception as e:
            row = {"book": book.name, "outcome": "error", "error": str(e)}
        row["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        add_to_manifest(manifest, row)
        counts[row["outcome"]] += 1
        detail = row.get("error") or row.get("reason", "")
        print(f"[{row['outcome']}] {book.name}" + (f": {detail}" if detail else ""))

    print("\nDone: " + ", ".join(f"{n} {k}" for k, n in counts.items()))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python b_extract_metadata.py configs/<settings>.toml")
    main(sys.argv[1])
# Title-page metadata extraction

Reads the first pages of digital documents with a local vision model and writes
one MARC 21 record (MARCXML) per book. Books that pass every check are ready to
load; the rest go to a cataloguer for review, with the reason recorded.

## How it works

1. **Page images.** The first five pages of each document (cover, title page and
   the back of the title page) are saved as images. This works for PDF, EPUB,
   XPS and multi-page TIFF.
2. **Image checks.** Blank pages are skipped; if every page is blank, the book
   goes straight to review. Blurred images also send a book to review.
3. **Model reading.** A vision language model (Qwen2.5-VL 7B, run locally
   through Ollama) reads the pages. It returns a fixed set of fields: title,
   subtitle, authors, edition, year, place, publisher, series and number, and up
   to three notes.
4. **Second reading.** The model reads the pages again at a slightly smaller
   size. If the two readings disagree on title, authors, year or publisher, the
   book goes to review.
5. **Text cross-check.** Separate OCR software (RapidOCR) reads the printed text
   on the same pages. The model's answer is checked against it (see below).
6. **Decision.** Each book is marked pass or review, with the reason recorded.
7. **Record.** A MARC 21 record is written for each book. It goes into a
   `processed` or `review` folder with a copy of the original document.

## How OCR is used

The model decides the answer: it can read stylised cover lettering that OCR
can't, and its answer is never replaced by the OCR text. OCR is used to check it.
Each field is scored 0-1 for how much of it is actually printed on the pages
(1.00 = found as given; low = probably misread or made up). Matching ignores
case, accents and punctuation.

OCR sends a book to review when:

- the title or publisher scores below 0.7
- large text on the title page isn't in the record (usually a title with its
  main part missing, or a wrong publisher)
- the line straight after the title isn't in the record (a missing subtitle)
- the model's year isn't one of the years in a copyright or "first published"
  line
- it could read almost no text

OCR also supplies, without ever causing a review:

- ISBN, ISSN and DOI, taken only from labelled lines with a valid check digit
- editors, for names after "Edited by"
- fiction, from statements such as "a work of fiction"

## All reasons for review

The images are blurry or all blank; there's no title or no year; the year is in
the future; an author isn't in "Last, First" order; the publisher looks like an
address or a series name; the two readings disagree; or an OCR check above
fails. Every reason found is listed in the record's 999 field and in
`manifest.csv`.

## Output

| | |
|---|---|
| `processed\` | Passed every check: the MARCXML record and the original document, with the same name |
| `review\` | The same, for books that need a cataloguer |
| `manifest.csv` | One row per book: outcome, reason, OCR scores, identifiers |

The record contains 020/022/024 (ISBN, ISSN, DOI), 245 (title), 250 (edition),
264 (place, publisher, year), 490 (series), 500 (notes), 588 (source of
description), 720 (names, uncontrolled), 883 (machine-generated provenance, with
each field's OCR score as a confidence value) and 999 (review reason). The full
mapping is at the top of `marc_record.py`. Every record is checked against the
Library of Congress MARCXML schema (`schema\MARC21slim.xsd`) before it's written.

## One-time setup

Needs Python 3.11+ and [Ollama](https://ollama.com/download). In PowerShell, in
the project folder:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull qwen2.5vl:7b
```

In `configs\export_configs.toml` (full run) and `configs\test_configs.toml`
(a fixed sample of 20, for trials), set `[pages] documents` to your folder of
documents. Every setting is explained in the file.

## Running

```powershell
.\.venv\Scripts\Activate.ps1
python a_extract_pages.py configs\export_configs.toml       # documents -> page images
python b_extract_metadata.py configs\export_configs.toml    # page images -> MARC records
```

Both can be stopped and rerun; finished items are skipped. Use
`configs\test_configs.toml` instead for a trial run.

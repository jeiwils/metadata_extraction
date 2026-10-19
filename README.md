# Title-page metadata extraction

Extracts title, subtitle, authors, edition, year, place, publisher, series,
notes and identifiers from images of a book's first pages with a local vision
model (via Ollama), and writes one MARC 21 record per book, as MARCXML.

Two modules, which can be used together or on their own:

| Module | Input | Output |
|---|---|---|
| `a_extract_pages.py` | A folder of documents: PDF, XPS, EPUB, multi-page TIFF (sub-folders searched too) | One folder of page images per document |
| `b_extract_metadata.py` | A folder with one sub-folder of page images per book | A MARCXML record per book, sorted into `processed\` and `review\` |

Already have scans as one image per page? Skip `a_extract_pages.py`: put each
book's images in a folder of its own and point `b_extract_metadata.py` at the
folder that contains them.

## Files

```
a_extract_pages.py      Module 1: documents -> page images
b_extract_metadata.py   Module 2: page images -> MARC records
marc_record.py          The mapping from the extracted data to MARC fields (used by module 2)
z_utils.py              Shared helpers both modules need (config reading, test samples)
requirements.txt        Python packages
configs\
    example_configs.toml  Example settings: copy it and set your documents folder (see Settings)
prompts\
    updated_extraction_prompt.txt    The instructions the model gets
    original_extraction_prompt.txt   The earlier version, kept for comparison
schema\
    MARC21slim.xsd        The Library of Congress MARCXML schema, for checking each record
testing\
    evaluate.py           Optional: scores the model against a ground_truth.csv you create (see below)
runs\                   Everything the pipeline writes: page images, records, manifests
```

Kept out of git (see `.gitignore`), so they stay on your machine: `runs\`, your
own config files in `configs\`, `testing\ground_truth.csv`, and `docs\`.

## One-time setup

1. **Python 3.11 or newer** from python.org (tick "Add python.exe to PATH").
   Check: `py --version`

2. **Ollama** from https://ollama.com/download (only needed for `b_extract_metadata.py`).
   It runs in the background (llama icon in the system tray). Check: `ollama --version`

3. **Virtual environment and packages** (PowerShell, in the project folder):
   ```powershell
   cd D:\tests\metadata_extraction
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   python -c "import pymupdf, cv2, ollama, rapidocr_onnxruntime, lxml; print('packages OK')"
   ```
   The MARCXML schema is already in `schema\`, so every record is checked against
   it before it's written. If it's ever missing, fetch it again:
   ```powershell
   Invoke-WebRequest https://www.loc.gov/standards/marcxml/schema/MARC21slim.xsd -OutFile schema\MARC21slim.xsd
   ```
   (Without it the records are still written, just not checked.)
   If `Activate.ps1` is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
   once. If that isn't allowed, skip activation and use `.\.venv\Scripts\python.exe`
   in place of `python` below.

4. **The model** named under `[metadata] model` in the config:
   ```powershell
   ollama pull qwen2.5vl:7b
   ollama list
   ```

## Settings

One config file holds the settings for both modules. Start by copying the
example - one copy for full runs and one for trials, for instance:

```powershell
Copy-Item configs\example_configs.toml configs\export_configs.toml
Copy-Item configs\example_configs.toml configs\test_configs.toml
```

In each, set `[pages] documents` to your folder of documents. For the full-run
copy, also set `[run] limit = 0` and `random_sample = false`, and point its
`output` and `pages` paths at a folder of their own (e.g. `../runs/full_run/...`)
so the two never mix.

Each module reads only its own section plus `[run]`:

- `[pages]` - for `a_extract_pages.py`: where the documents are, where the page
  images go, how many pages, resolution.
- `[metadata]` - for `b_extract_metadata.py`: where the page images are, where the
  results go, the model and the checks.
- `[run]` - for both: test-sample size, and whether to redo finished items.

When running both modules, `[metadata] pages` is normally the same folder as
`[pages] output`. Paths can be relative to the config file. Every setting is
required: a missing one stops the run with a message saying which. The
example config uses relative paths for everything inside the project
(`../runs/...`, `../prompts/...`), so only `[pages] documents` is absolute.

## Running it

```powershell
cd D:\tests\metadata_extraction
.\.venv\Scripts\Activate.ps1
python a_extract_pages.py configs\export_configs.toml       # documents -> page images
python b_extract_metadata.py configs\export_configs.toml    # page images -> MARC records
```
Both can be stopped and rerun: finished items are skipped.

For a trial run use `configs\test_configs.toml`. It takes the same fixed sample every
time, so before/after comparisons are fair.

## What comes out

`a_extract_pages.py`, in `[pages] output`:

| | |
|---|---|
| `<folder>__<file name>\p01.png, p02.png ...` | The first pages of each document |
| `<folder>__<file name>\source.txt` | Path of the original document |

`b_extract_metadata.py`, in `[metadata] output`:

| | |
|---|---|
| `processed\` | Passed every check: the MARCXML record, plus the original document with the same name, if known (for High-Wire) |
| `review\` | Needs a cataloguer: the record and document as above, with the reason in field 999 $a. Books whose pages were all blank are here too, with an almost empty record |
| `manifest.csv` | One row per book: outcome (`pass`, `review`, `blank` or `error`), reason, pages used, image measurements, OCR match scores, ISBN/ISSN/DOI |

The original document is copied next to its record when the book folder has a
`source.txt` (i.e. it came from `a_extract_pages.py`); otherwise the record is
written on its own.

### The MARC record

`marc_record.py` builds it; the mapping is listed at the top of that file, and
the values that are BL policy (the agency code in 883, encoding level, the 588
wording, the local tags) are constants there, easy to change.

| Field | Contents |
|---|---|
| Leader, 008 | Language material, monograph, abbreviated level, no ISBD punctuation; 008 has the year, form of item 'o' (online), and literary form '1' when the pages say it's fiction. Positions the pipeline can't know use the fill character `\|` |
| 020 / 022 / 024 | ISBN (`$q` paperback, ebook...), ISSN, DOI (`024 7# ... $2 doi`), taken from the OCR text |
| 245 | `$a` title, `$b` subtitle; second indicator skips a leading The/A/An when filing |
| 250 | Edition |
| 264 #1 | `$a` place, `$b` publisher, `$c` year - '[publisher not identified]' etc. when one isn't printed |
| 490 0# | Series `$a`, number `$v` |
| 500 | Notes the model found worth recording (e.g. the period a report covers) |
| 588 | 'Description based on images of the first pages of the digital file.' |
| 720 | Each name, uncontrolled: `1#` person, `2#` organisation; `$e author` or `$e editor` |
| 883 0# | Provenance: 'metadata extracted using Ollama with qwen2.5vl:7b', date, agency. Further 883s carry each field's OCR match score as `$c` (confidence), linked by `$8` to the 245, 264 or 720 it describes |
| 930 | 'submission portal' |
| 999 | Review reason (review records only) |

With `schema\MARC21slim.xsd` in place, every record is checked against
the schema before it's written; a record that fails stops that book with an
error in the manifest rather than reaching High-Wire. The schema checks
structure (tags, indicators, subfields), not the cataloguing.

A record goes to review if: the images are blurry or all blank; there's no title
or no year; the year is in the future; an author isn't in "Last, First" order;
the publisher looks like an address or a series name; with
`agreement_check = true`, two readings of the pages disagree on title, authors,
year or publisher (the reason shows both readings); or, with `ocr_check = true`,
the OCR cross-check below fails.

### The OCR cross-check (`ocr_check`)

A separate OCR program (RapidOCR, which runs on the CPU and needs no internet
once installed) reads the same page images. Each field the model gave is then
scored 0-1 for how well the printed text backs it up:

- **1.00** - found on the page as given
- **around 0.8-0.95** - found, with a letter or two different (normal for OCR)
- **low** - mostly not on the page: the model may have misread or made it up

Matching is forgiving: case, accents and punctuation are ignored; the words of a
title must be found close together on the page, but not necessarily in order;
title and subtitle are matched separately (they are often printed apart);
'Ltd', 'Publishing' and similar are ignored in publisher names.

The model still decides the answer - it can read stylised cover lettering that
OCR can't, and its answer is never replaced by the OCR text. OCR only sends a
book to review when:

- the title or publisher scores below `ocr_review_below` (default 0.7)
- big text on the title page, or the line straight after the title, isn't in
  the record (see below)
- the pages have a copyright or 'first published' statement and the model's
  year isn't one of its years
- OCR could read almost no text (e.g. pages that are all pictures)

The scores are in `manifest.csv` (`ocr_title`, `ocr_author`, `ocr_date`,
`ocr_publisher`) and in the record as 883 `$c` confidence values linked to the
245, 264 and 720 fields, so a cataloguer can see which fields to look at first.
A blank score means there was nothing to check (e.g. no publisher given).

**The text-size check.** OCR also measures how tall each line of text is. On a
title page the biggest text is almost always the title (or the author or
publisher), so if a line of big text is mostly missing from the record's title,
authors and publisher, the book goes to review, e.g. `large text on the title
page not in the record: 'Fapas'`. This catches a title with its main part left
out (report14286 came out as just "Fatty Acids in Olive Oil"), and often a wrong
publisher too, since a big logo usually belongs to the publisher. It is only
applied on pages with text clearly bigger than the rest (1.6 times the typical
line), and ignores headings like "Abstract", dates, and acronyms of the
publisher ("LSE"). It does **not** catch a missing subtitle: subtitles are
usually printed smaller. Tested on 53 books: 4 flags, all real problems, no
false alarms - but it was tuned on those same books, so watch for false alarms
on covers with big series names or slogans.

**The subtitle check.** Subtitles are printed smaller than the title, so the
text-size check misses them; instead, the line straight after the title is
checked. If it's nearly as big as the title's last line, isn't a date, and isn't
in the record, the book goes to review: `line after the title not in the record -
a missing subtitle? 'Fish Identification'`. It only applies where the title is
at least as big as the page's typical text, so dense report pages are skipped,
and it can't see a subtitle printed on the same line as the title. On today's
batch it caught all three dropped subtitles (report29113, reportGeMMC12,
reportDWC108) with no false alarms; NECR508 is flagged for "Project Summary
Report", a borderline case for a cataloguer.

**Identifiers (recorded, never a reason for review).** ISBNs, ISSNs and the
document's DOI are taken from the OCR text and written to 020, 022 and 024 in
the record and to manifest.csv, e.g. `020 ## $a 9781805177043 $q paperback`,
`022 ## $a 0963-8091`, `024 7# $a 10.3310/LEXB9006 $2 doi`. An ISBN or ISSN only
counts on a line labelled ISBN/ISSN and with a correct check digit, so an OCR
misread gives no number rather than a wrong one. ISBNs are written as digits
only; a qualifier (paperback, ebook, hardback, pdf...) is added when printed
beside the number. Where the pages also cite other works' DOIs, the one on a
line starting "DOI" is used, else the one printed most often. On the 33 books
checked, every identifier matched the documents' own text.

**Editors and fiction (recorded, never a reason for review).**
- Authors named straight after "Edited by" (or "Editors:", "(eds)") get
  `720 ... $e editor` instead of `$e author`.
- If the pages say "work of fiction", "products of the author's imagination" or
  "used fictitiously", 008 position 33 (literary form) is set to '1' (fiction).

To record the scores without them ever sending a book to review, set
`ocr_review_below = 0`. To switch the check off entirely, `ocr_check = false`.

To redo the metadata (after changing the prompt, model or checks), delete
`manifest.csv` from the results folder and rerun `b_extract_metadata.py`. Old
records are cleaned up automatically. To redo the page images (after changing
`n_pages` or `dpi`), delete the pages folder and rerun `a_extract_pages.py`.

## Speed

Blank pages are skipped and empty margins cropped automatically before the model
sees a page. Without a graphics card, expect **minutes per book**; time it first
with `limit = 5` (compare the first and last `timestamp` in `manifest.csv`).
What else makes it faster, biggest effect first:

- `agreement_check = false` - halves the time (each book is read once).
- `ocr_check = false` - saves the OCR step: about 10 seconds for a sparse title
  page, up to 45 for five pages of dense report text (measured on 2 CPU cores).
  Usually small next to the model's time.
- `n_pages` - every page is roughly the same amount of work. Use 4 for scans
  that start at the half-title; 5 if a cover page comes first.
- `max_image_side` - 1024 is a good balance; going lower risks unreadable small print.
- A smaller model, e.g. `qwen3-vl:4b` - check the accuracy cost with `testing\evaluate.py`.

## Checking accuracy

`testing\evaluate.py` scores the pipeline against correct values you supply in a
`testing\ground_truth.csv`, one row per book (catalogue records are ideal):

```
book,title,author,date,publisher,on_first_pages,notes
batch01__annual_report_2024,Annual Report 2024: Water Quality in Northern Rivers,Example Research Institute,2024,Example Research Institute,title;author;date;publisher,
```
- `book` is the book's folder name in the pages folder.
- `author`: 'Last, First' names separated by `; `; an organisation as written.
- Several acceptable answers for title or publisher: separate with ` || `.
- `on_first_pages` (optional): fields actually printed on the page images.
- `notes` is ignored - use it for your own remarks.

Start with the books in your test sample; for figures worth quoting, extend it
to 50-100 books, filled in by a cataloguer from the document or the catalogue
record. The file is ignored by git, so it can hold your own collection's details.

Then:
```powershell
python testing\evaluate.py configs\test_configs.toml testing\ground_truth.csv
```
It runs every book in the file through the same checks as a real run (nothing
is saved to processed\ or review\) and scores each field:

- **title** - right if it matches a correct title, or its main part before the
  colon, allowing small slips (so a missing subtitle still counts as right)
- **author** - the first author right: for a person, the surname, written
  'Last, First'; for an organisation, the name. Also an overlap score for the
  whole list
- **year** - exact
- **publisher** - right if the answer contains a correct name (ignoring Ltd,
  The...): a fuller form passes, a shortened one doesn't unless listed with ` || `

`summary.md` and the screen show accuracy for all books, for those that passed
and for those sent to review, plus how many wrong answers the OCR scores caught
and how many right ones they flagged by mistake. `results.csv` has every book's
answer beside the correct one. Use it to compare settings (e.g. run once with
`agreement_check = true` and once with `false`), models or prompts. Answers are
cached, so re-scoring after correcting the ground truth is instant; add
`--rerun` after changing the prompt.

## Troubleshooting

- **"Can't reach Ollama"**: start Ollama from the Start menu, or run `ollama serve`
  in a separate window.
- **"... is missing '...' under [...]"**: the config lacks that setting; copy it
  from `configs\export_configs.toml`.
- **Empty replies or "no valid JSON"**: the error line shows how many tokens the
  request used and why the model stopped. For the cause, look at the end of
  `%LOCALAPPDATA%\Ollama\server.log` straight after it happens.
- **Slow**: run `ollama ps` while it's working. "100% CPU" is normal without a
  graphics card; see "Speed" above.
- **Out of memory**: close other programs, lower `num_ctx` (8192 is enough for
  5 pages at 1024 px), or switch to a smaller model.
- **"OCR package is missing"**: run `pip install rapidocr_onnxruntime`, or set
  `ocr_check = false`.
- **"input probably truncated"**: the pages didn't fit in `num_ctx`. Raise it, or
  lower `max_image_side` or `n_pages`.
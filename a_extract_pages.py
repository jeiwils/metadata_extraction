"""
Module 1: documents -> page images.

Saves the first pages of every document under [pages] documents (sub-folders
included) as PNG images, one folder per document:

    <output>/<parent folder>__<file name>/p01.png, p02.png, ...
                                          source.txt  (where the original document is)

Works for anything PyMuPDF opens as a multi-page document: PDF, XPS, EPUB and
multi-page TIFF scans. Scans that are already one image file per page don't
need this step - put each book's images in a folder of its own and go
straight to b_extract_metadata.py.

Usage:
    python a_extract_pages.py configs/<settings>.toml

Settings (all required):
    [pages]
    documents = '...'   # folder containing the documents (sub-folders included)
    output = '...'      # where the page-image folders go (not inside documents)
    n_pages = 4         # how many pages to take from the start of each document
    dpi = 200           # image resolution

    [run]               # shared with b_extract_metadata.py
    limit = 0           # 0 = everything; e.g. 20 for a test run
    random_sample = false   # true: a fixed random sample instead of the first N
    overwrite = false   # true: redo documents that are already done
"""

import shutil
import sys
from pathlib import Path

import pymupdf

from z_utils import Config, items_to_process

DOCUMENT_TYPES = {".pdf", ".xps", ".oxps", ".epub", ".tif", ".tiff"}


def find_documents(folder: Path) -> list[Path]:
    """Every document in the folder and its sub-folders, in alphabetical order."""
    return [p for p in sorted(folder.rglob("*"))
            if p.is_file() and p.suffix.lower() in DOCUMENT_TYPES]


def book_name(document: Path) -> str:
    """
    The name used for everything made from this document: '<parent folder>__<file name>'.
    Including the parent folder keeps names unique when two folders hold files with the same name.
    """
    return f"{document.parent.name}__{document.stem}"


def save_first_pages(document: Path, out_folder: Path, n_pages: int, dpi: int) -> int:
    """
    Save the first n_pages of the document as p01.png, p02.png, ... in out_folder
    (all of them if it is shorter), plus source.txt recording where the
    original is. Returns the number of pages saved.

    Everything is written to a temporary '.part' folder first and renamed at the
    end, so out_folder only ever exists complete - that is how a rerun can tell
    a document is done.
    """
    temp_folder = out_folder.with_name(out_folder.name + ".part")
    shutil.rmtree(temp_folder, ignore_errors=True)
    temp_folder.mkdir(parents=True)
    try:
        # Render the pages
        with pymupdf.open(document) as doc:
            if doc.page_count == 0:
                raise ValueError("document has no pages")
            count = min(n_pages, doc.page_count)
            for i in range(count):
                doc[i].get_pixmap(dpi=dpi).save(temp_folder / f"p{i + 1:02d}.png")
        # Record where the original document is, for b_extract_metadata.py
        (temp_folder / "source.txt").write_text(str(document), encoding="utf-8")
    except Exception:
        shutil.rmtree(temp_folder, ignore_errors=True)
        raise

    # Swap the finished folder into place (replacing an old one when overwriting)
    shutil.rmtree(out_folder, ignore_errors=True)
    temp_folder.rename(out_folder)
    return count


def main(settings_file: str) -> None:
    # Read the settings and check the folders make sense
    config = Config(settings_file)
    documents_folder = config.path("pages", "documents")
    output_folder = config.path("pages", "output")
    n_pages, dpi = config.value("pages", "n_pages"), config.value("pages", "dpi")
    limit, random_sample = config.value("run", "limit"), config.value("run", "random_sample")
    overwrite = config.value("run", "overwrite")

    if not documents_folder.is_dir():
        raise SystemExit(f"Documents folder not found: {documents_folder}")
    if output_folder == documents_folder or documents_folder in output_folder.parents:
        raise SystemExit(f"Output folder {output_folder} is inside the documents folder {documents_folder}")
    output_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.file, output_folder / "run_config.toml")  # record of the settings used

    # Decide which documents to process
    documents = items_to_process(find_documents(documents_folder), limit, random_sample)
    if limit:
        print(f"[test run] {len(documents)} documents ({'random sample' if random_sample else 'first in order'})")

    # Process them one by one
    counts = {"done": 0, "skipped": 0, "failed": 0}
    names_seen = {}
    for document in documents:
        name = book_name(document)

        # Windows file names ignore upper/lower case, so 'Book' and 'book' would collide
        if name.lower() in names_seen:
            print(f"[duplicate name] {document} clashes with {names_seen[name.lower()]} - skipped")
            counts["skipped"] += 1
            continue
        names_seen[name.lower()] = document

        # Already done in an earlier run?
        out_folder = output_folder / name
        if out_folder.is_dir() and not overwrite:
            counts["skipped"] += 1
            continue

        try:
            pages = save_first_pages(document, out_folder, n_pages, dpi)
            counts["done"] += 1
            print(f"[ok] {name} ({pages} pages)")
        except Exception as e:
            counts["failed"] += 1
            print(f"[failed] {document}: {e}")

    print("\nDone: " + ", ".join(f"{n} {k}" for k, n in counts.items()))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python a_extract_pages.py configs/<settings>.toml")
    main(sys.argv[1])
"""
Measure how accurate the model is, against correct values you supply - with and
without the pass/review checks.

Every book listed in the ground truth goes through b_extract_metadata.py's
assess_book() - the same checks as a real run (page images, model, OCR if on,
pass/review) but nothing is saved to processed/ or review/. The model reads
every book whose pages aren't all blank, whatever its outcome, so the same
answers can be scored two ways:
    without the checks:  all books
    with the checks:     only the books that passed

Usage:
    python testing/evaluate.py configs/test_configs.toml testing/ground_truth.csv [--rerun]

Model answers are saved in <metadata output>/evaluation/answers/, so re-scoring
after correcting the ground truth is instant. --rerun asks the model again
(do this after changing the model or the prompt).

Ground truth CSV columns:
    book             - the book's folder name in the pages folder
    title, publisher - several acceptable answers separated by ' || '
    author           - 'Last, First' names separated by '; '
    date             - four-digit year (blank if none)
    on_first_pages   - optional: which fields are visible on the page images,
                       e.g. 'title;author'. Adds a table of accuracy for just those.
"""

import csv
import json
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

# the pipeline modules are in the project folder, one level up
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import b_extract_metadata as em  # noqa: E402

FIELDS = ["title", "author", "date", "publisher"]


# ================================================================ comparing answers

def simplify(text: str) -> str:
    """Lower case, no accents, no punctuation, '&' as 'and' - for comparing loosely."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("&", " and ")
    return " ".join("".join(c if c.isalnum() else " " for c in text).split())


def accepted_answers(value: str) -> list[str]:
    """The ' || '-separated alternatives in a ground-truth cell."""
    return [v.strip() for v in (value or "").split("||") if v.strip()]


def title_correct(answer: str, truth: str) -> bool:
    """Right if it matches a correct title - or its main part, before the colon - allowing small slips."""
    given = simplify(answer)
    if not given:
        return not accepted_answers(truth)
    for correct in accepted_answers(truth):
        full, main = simplify(correct), simplify(correct.split(":")[0])
        if max(SequenceMatcher(None, given, full).ratio(), SequenceMatcher(None, given, main).ratio()) >= 0.9:
            return True
    return False


def surname_of(name: str) -> str:
    """Last word of the surname part: 'Vosough Ahmadi, Bouda' -> 'ahmadi'."""
    part = name.split(",")[0] if "," in name else name
    words = simplify(part).split()
    return words[-1] if words else ""


def author_scores(answer: list[str], truth: str) -> dict:
    """
    How well the authors match:
      first_ok  - first author right, and written 'Last, First'
      f1        - overall overlap of the two author lists (1.0 = identical)
    """
    correct = [surname_of(n) for n in truth.split(";") if n.strip()]
    if not correct:
        return {"f1": float(not answer), "first_ok": not answer}
    if not answer:
        return {"f1": 0.0, "first_ok": False}

    answer_words = [set(simplify(name).split()) for name in answer]
    found = sum(any(s in words for words in answer_words) for s in correct)       # correct authors given
    right = sum(any(s in words for s in correct) for words in answer_words)       # given authors that are correct
    precision, recall = right / len(answer), found / len(correct)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    # First author: a person must be 'Last, First' with the right surname; an
    # organisation ('Fera Science Ltd') must contain the correct name
    first_truth = truth.split(";")[0].strip()
    if "," in first_truth:
        first_ok = "," in answer[0] and surname_of(answer[0]) == correct[0]
    else:
        first_ok = simplify(first_truth) in simplify(answer[0])
    return {"f1": f1, "first_ok": first_ok}


def publisher_correct(answer: str, truth: str) -> bool:
    """
    Right if the answer contains a correct name, ignoring 'Ltd', 'Limited', 'Inc'
    and 'The': 'Fera Science Ltd (Fera)' is right for 'Fera Science Ltd', but
    'London School of Economics' is wrong for 'Centre for Analysis of Social
    Exclusion, London School of Economics'. List acceptable shorter forms with ' || '.
    """
    ignore = {"ltd", "limited", "inc", "the"}
    given = " ".join(w for w in simplify(answer).split() if w not in ignore)
    correct_answers = accepted_answers(truth)
    if not correct_answers:
        return not given
    for correct in correct_answers:
        wanted = " ".join(w for w in simplify(correct).split() if w not in ignore)
        if given and wanted in given:
            return True
    return False


def score_answer(answer: dict, truth: dict) -> dict:
    """Right/wrong for each field of one book."""
    authors = author_scores(answer["author"], truth["author"])
    return {
        "title": title_correct(answer["title"], truth["title"]),
        "author": authors["first_ok"],
        "author_f1": round(authors["f1"], 3),
        "date": answer["date"] == truth["date"].strip(),
        "publisher": publisher_correct(answer["publisher"], truth["publisher"]),
    }


# ================================================================ getting the answers

def answer_for_book(book: Path, settings: dict, answers_folder: Path, rerun: bool) -> dict:
    """
    The model's answer and outcome for one book - from the saved copy if there is
    one for the same model and settings, otherwise by running the steps.
    """
    # Saved from an earlier run?
    saved = answers_folder / f"{book.name}.json"
    run_with = [settings["model"], f"agreement_check={settings['agreement_check']}",
                f"ocr_check={settings['ocr_check']}", f"ocr_review_below={settings['ocr_review_below']}"]
    if saved.exists() and not rerun:
        result = json.loads(saved.read_text(encoding="utf-8"))
        if result["run_with"] == run_with:
            return result

    # Run the steps
    row, metadata, ocr = em.assess_book(book, settings)
    result = {"run_with": run_with, "outcome": row["outcome"], "reason": row["reason"], "metadata": metadata,
              "ocr_scores": ocr["scores"] if ocr else {}}
    saved.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


# ================================================================ reporting

def accuracy_row(rows: list[dict], label: str) -> str:
    """One line of the summary table: share of books right, per field."""
    if not rows:
        return f"| {label} | 0 | - | - | - | - | - |"
    n = len(rows)
    share = lambda field: f"{100 * sum(r[field] for r in rows) / n:.0f}%"
    f1 = sum(r["author_f1"] for r in rows) / n
    return f"| {label} | {n} | {share('title')} | {share('author')} | {f1:.2f} | {share('date')} | {share('publisher')} |"


def accuracy_where_visible(rows: list[dict]) -> list[str]:
    """Accuracy per field, counting only books where that field is on the page images."""
    lines = []
    for field in FIELDS:
        visible = [r for r in rows if field in r["on_first_pages"]]
        if visible:
            right = sum(r[field] for r in visible)
            lines.append(f"| {field} | {right}/{len(visible)} | {100 * right / len(visible):.0f}% |")
    return lines


def ocr_report(rows: list[dict], threshold: float) -> list[str]:
    """
    How well the OCR match scores separate right answers from wrong ones, per
    field: books scoring below the threshold that were wrong are catches; ones
    that were right are false alarms. Shows whether ocr_review_below is set well.
    Title and publisher are the fields whose score can send a book to review;
    author is shown for information. (The year is checked differently - against
    copyright lines - so it isn't in this table.)
    """
    lines = []
    for field in ("title", "publisher", "author"):
        scored = [r for r in rows if r.get(f"ocr_{field}") is not None]
        if not scored:
            continue
        low = [r for r in scored if r[f"ocr_{field}"] < threshold]
        wrong = [r for r in scored if not r[field]]
        caught = sum(1 for r in low if not r[field])
        lines.append(f"| {field} | {len(scored)} | {len(wrong)} | {caught} | {len(low) - caught} |")
    if not lines:
        return []
    return [f"## OCR match scores below {threshold}", "",
            "| Field | Books scored | Wrong answers | Wrong and scored low (caught) | Right but scored low (false alarm) |",
            "|---|---|---|---|---|", *lines, ""]


def main(settings_file: str, truth_file: str, rerun: bool) -> None:
    # Read the settings and the ground truth
    settings = em.read_settings(settings_file)
    evaluation_folder = settings["output"] / "evaluation"
    answers_folder = evaluation_folder / "answers"
    answers_folder.mkdir(parents=True, exist_ok=True)
    em.check_model_installed(settings["model"])

    with open(truth_file, newline="", encoding="utf-8-sig") as f:
        truth = {row["book"]: row for row in csv.DictReader(f)}
    books = {book.name: book for book in em.find_books(settings["pages"])}
    for name in sorted(set(truth) - set(books)):
        print(f"[missing] {name} is in the ground truth but has no folder of page images")

    # Get each book's answer and score it
    results = []
    for name in sorted(set(truth) & set(books)):
        try:
            result = answer_for_book(books[name], settings, answers_folder, rerun)
        except Exception as e:
            print(f"[error] {name}: {e}")
            continue
        answer, correct = result["metadata"], truth[name]
        scores = score_answer(answer, correct)
        results.append({
            "book": name, "outcome": result["outcome"], **scores,
            "on_first_pages": [f for f in (correct.get("on_first_pages") or "").split(";") if f],
            "answer_title": answer["title"], "true_title": correct["title"],
            "answer_author": "; ".join(answer["author"]), "true_author": correct["author"],
            "answer_date": answer["date"], "true_date": correct["date"],
            "answer_publisher": answer["publisher"], "true_publisher": correct["publisher"],
            "reason": result["reason"],
            **{f"ocr_{field}": score for field, score in result.get("ocr_scores", {}).items()},
        })
        print(f"[{result['outcome']}] {name}  " + " ".join(f"{k}={'ok' if scores[k] else 'X'}" for k in FIELDS))

    if not results:
        raise SystemExit("Nothing scored.")

    # Per-book details
    with open(evaluation_folder / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        columns = list(dict.fromkeys(key for r in results for key in r))
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in results:
            writer.writerow({**r, "on_first_pages": ";".join(r["on_first_pages"])})

    # Summary
    passed = [r for r in results if r["outcome"] == "pass"]
    not_passed = [r for r in results if r["outcome"] != "pass"]
    lines = [
        f"# Evaluation: {settings['model']}",
        "",
        "## With and without the pass/review checks",
        "",
        "| Books | Count | Title | First author (Last, First) | Author overlap | Year | Publisher |",
        "|---|---|---|---|---|---|---|",
        accuracy_row(results, "All (no checks)"),
        accuracy_row(passed, "Passed the checks"),
        accuracy_row(not_passed, "Sent to review"),
        "",
        f"The checks passed {len(passed)} of {len(results)} books.",
        "",
        *ocr_report(results, settings["ocr_review_below"] or 0.7),
        *(["## Only where the answer is on the page images", "",
           "| Field | Right | Accuracy |", "|---|---|---|", *accuracy_where_visible(results), ""]
          if any(r["on_first_pages"] for r in results) else []),
        "Per-book answers and scores: results.csv",
    ]
    summary = "\n".join(lines)
    (evaluation_folder / "summary.md").write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--rerun"]
    if len(args) != 2:
        raise SystemExit("Usage: python testing/evaluate.py <settings.toml> <ground_truth.csv> [--rerun]")
    main(args[0], args[1], rerun="--rerun" in sys.argv)
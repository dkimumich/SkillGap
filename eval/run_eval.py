"""Run SkillGap against hand-labelled job postings and report agreement.

Usage (from the repo root):
    python eval/run_eval.py
    python eval/run_eval.py --resume eval/data/resume.txt --labels eval/labels.csv

Writes the report to eval/results.md and prints it.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from skillgap import match
from skillgap.evaluation import confusion, evaluate

ROOT = Path(__file__).parent


def fmt(value: float | None, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0%}" if pct else f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resume", type=Path, default=ROOT / "data" / "resume.txt")
    parser.add_argument("--labels", type=Path, default=ROOT / "labels.csv")
    parser.add_argument("--postings", type=Path, default=ROOT / "postings")
    parser.add_argument("--out", type=Path, default=ROOT / "results.md")
    args = parser.parse_args()

    resume = args.resume.read_text(encoding="utf-8")
    with args.labels.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["id"].strip()]

    results = []
    for row in rows:
        posting = (args.postings / f"{row['id']}.txt").read_text(encoding="utf-8")
        results.append((row, match(resume, posting)))

    labels = [row["label"].strip().lower() for row, _ in results]
    scorers = {
        "SkillGap (full score)": [r.score for _, r in results],
        "Skill coverage only": [
            50 if r.skill_coverage is None else 100 * r.skill_coverage for _, r in results
        ],
        "TF-IDF only": [100 * r.tfidf_similarity for _, r in results],
    }

    lines = [
        "# SkillGap evaluation",
        "",
        f"{len(results)} hand-labelled postings scored against one resume.",
        "",
        "| Scorer | Pairwise agreement | Spearman | Label accuracy |",
        "| --- | --- | --- | --- |",
    ]
    for name, scores in scorers.items():
        rep = evaluate(scores, labels)
        cells = [fmt(rep.pairwise, True), fmt(rep.spearman), fmt(rep.accuracy, True)]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    full = scorers["SkillGap (full score)"]
    table = confusion(full, labels)
    lines += [
        "",
        "Pairwise agreement: of postings you labelled differently, how often the better fit "
        "scored higher (50% = coin flip). Label accuracy uses good >= 60, okay >= 40.",
        "",
        "## Confusion (rows = your label, columns = SkillGap's)",
        "",
        "| | poor | okay | good |",
        "| --- | --- | --- | --- |",
    ]
    for actual in ("poor", "okay", "good"):
        cells = " | ".join(str(table[actual][p]) for p in ("poor", "okay", "good"))
        lines.append(f"| **{actual}** | {cells} |")

    lines += [
        "",
        "## Per posting",
        "",
        "| Posting | Your label | Score | Confidence | Missing required |",
        "| --- | --- | --- | --- | --- |",
    ]
    for (row, r), label in sorted(zip(results, labels, strict=True), key=lambda x: -x[0][1].score):
        name = f"{row['company']} – {row['role']}".strip(" –")
        missing = ", ".join(r.missing_required) or "none"
        lines.append(f"| {name} | {label} | {r.score} | {r.confidence} | {missing} |")

    report = "\n".join(lines) + "\n"
    args.out.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

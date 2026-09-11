"""Score every detector on the same held-out split and write the results.

    python evaluate_guards.py

Writes results/metrics.md
"""

from pathlib import Path

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from guard.data import load_split
from guard.heuristics import HeuristicDetector
from guard.promptguard import PromptGuardDetector

OUT_PATH = Path("results/metrics.md")
THRESHOLD = 0.5


def evaluate(detector, rows):
    """Run one detector over the test rows and collect what it got wrong."""
    truth, predicted, misses, false_alarms = [], [], [], []

    for index, (text, label, category) in enumerate(rows, 1):
        if index % 25 == 0:
            print(f"    {index}/{len(rows)}")
        prediction = detector.predict(text, THRESHOLD)
        truth.append(label)
        predicted.append(prediction)
        if label == 1 and prediction == 0:
            misses.append((text, category, detector.score(text)))
        if label == 0 and prediction == 1:
            false_alarms.append((text, category, detector.score(text)))

    precision, recall, f1, _support = precision_recall_fscore_support(
        truth, predicted, average="binary", zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "misses": misses, "false_alarms": false_alarms,
    }


def misses_by_category(misses, rows):
    """How many injections of each attack type slipped through."""
    totals = {}
    for _text, label, category in rows:
        if label == 1:
            totals[category] = totals.get(category, 0) + 1

    missed = {}
    for _text, category, _score in misses:
        missed[category] = missed.get(category, 0) + 1

    return [(category, missed.get(category, 0), total)
            for category, total in sorted(totals.items())]


def render(results, rows):
    injected = sum(1 for _t, label, _c in rows if label == 1)
    lines = [
        "# Detector results",
        "",
        f"Held-out split: {len(rows)} chunks, {injected} injected, "
        f"{len(rows) - injected} clean.",
        f"Decision threshold {THRESHOLD}. The split comes from "
        "`guard/data.py`, so every detector sees the same rows.",
        "",
        "## Scores",
        "",
        "| Detector | Precision | Recall | F1 | Missed injections | False alarms |",
        "|---|---|---|---|---|---|",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result['precision']:.3f} | {result['recall']:.3f} "
            f"| {result['f1']:.3f} | {result['fn']} | {result['fp']} |"
        )

    lines += ["", "## Confusion matrices", ""]
    for name, result in results.items():
        lines += [
            f"**{name}**", "",
            "| | predicted clean | predicted injected |",
            "|---|---|---|",
            f"| actually clean | {result['tn']} | {result['fp']} |",
            f"| actually injected | {result['fn']} | {result['tp']} |",
            "",
        ]

    lines += ["## Which attacks slip through", ""]
    for name, result in results.items():
        lines += [f"**{name}**", "",
                  "| Attack type | Missed | Total | Miss rate |", "|---|---|---|---|"]
        for category, missed, total in misses_by_category(result["misses"], rows):
            rate = missed / total if total else 0
            lines.append(f"| {category} | {missed} | {total} | {rate:.0%} |")
        lines.append("")

    lines += ["## False alarms on clean text", ""]
    for name, result in results.items():
        lines += [f"**{name}**", ""]
        if not result["false_alarms"]:
            lines += ["None.", ""]
            continue
        shown = result["false_alarms"][:8]
        for text, category, score in shown:
            flat = text.replace("\n", " ").strip()
            lines.append(f"- `{flat[:110]}` ({category}, score {score:.3f})")
        if len(result["false_alarms"]) > len(shown):
            lines.append(f"- ... and {len(result['false_alarms']) - len(shown)} more")
        lines.append("")

    lines += [
        "## How to read these numbers",
        "",
        "The test half holds 21 of the 75 templates, and the model trained on",
        "none of them. Every test sentence is a phrasing the fine-tuned model",
        "never saw during training. Several attack types rest on a single test",
        "template, so read the per-category miss rates as one example each,",
        "not as a rate.",
        "",
        "Treat the heuristics column as optimistic. I wrote the injection",
        "templates in `build_dataset.py` and the patterns in",
        "`guard/heuristics.py` in the same week, so they share vocabulary that",
        "a real attacker never agreed to use. Prompt Guard read neither file,",
        "which leaves its column as the one nobody tuned.",
        "",
        "The corpus comes out of templates. A real supplier writes sentences no",
        "template here anticipates, so every recall figure here sits above what",
        "these detectors would reach on invoices arriving from outside.",
        "",
        "## Two earlier results this file no longer reports",
        "",
        "The first run gave precision 1.000 and zero false alarms for both",
        "untrained detectors. Templates without substitution slots had",
        "collapsed to one row each during deduplication, and the split handed",
        "those rows to the training half. Adding slots to every template took",
        "the corpus from 767 rows to 1152 and the hard negatives from 96 to 228.",
        "",
        "The second run split by row and gave the fine-tuned model 1.000 on",
        "every metric. 191 of the 231 test rows had a twin in the training half",
        "differing only in an item number, a date or a company name. The model",
        "had memorised templates. `guard/data.py` now splits by template.",
        "",
    ]

    return "\n".join(lines) + "\n"


def main():
    _train, test = load_split()
    print(f"Test split: {len(test)} rows")

    detectors = [HeuristicDetector(), PromptGuardDetector()]
    try:
        from guard.distilbert import DistilBertDetector
        detectors.append(DistilBertDetector())
    except FileNotFoundError as error:
        print(f"  skipping distilbert-lora: {error}")
    results = {}
    for detector in detectors:
        print(f"  {detector.name}")
        results[detector.name] = evaluate(detector, test)
        if hasattr(detector, "save_cache"):
            detector.save_cache()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(render(results, test), encoding="utf-8")
    print(f"\nWritten to {OUT_PATH}\n")

    for name, result in results.items():
        print(f"{name:<22} P {result['precision']:.3f}  "
              f"R {result['recall']:.3f}  F1 {result['f1']:.3f}  "
              f"missed {result['fn']}  false alarms {result['fp']}")


if __name__ == "__main__":
    main()

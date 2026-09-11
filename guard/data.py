"""Load the corpus and split it by template.

Every detector and the training script take their rows from here, so they all
score and train on the same partition.

The split runs over templates, never over rows. build_dataset.py produces each
row by filling a template's slots with an item number, a date or a company
name, so two rows from one template differ only in those values. Splitting by
row put 191 of 231 test rows beside a twin in the training half, and the
fine-tuned model scored a perfect 1.000 by recognising sentences it had
already seen. Holding out whole templates leaves the test half made of
phrasings the model has never met.
"""

import csv
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

DATASET_PATH = Path("data/injections/dataset.csv")
SPLIT_SEED = 42

# Four folds puts roughly a quarter of the templates in the test half. The two
# smallest attack categories have four templates each; four folds is the most
# that still gives each of them one template in the test half.
TEST_FOLDS = 4
VALIDATION_FOLDS = 5


def load_rows():
    """Read the corpus as a list of (text, label, category, template_id)."""
    with DATASET_PATH.open(encoding="utf-8") as handle:
        return [(row["text"], int(row["label"]), row["category"],
                 row["template_id"])
                for row in csv.DictReader(handle)]


def _hold_out(rows, folds):
    """Split rows into (kept, held_out), one whole fold of templates held out.

    Stratifying on category keeps every attack type present on both sides of
    the split. Grouping on template_id keeps each template on one side only.
    """
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True,
                                    random_state=SPLIT_SEED)
    categories = [category for _t, _l, category, _g in rows]
    templates = [template for _t, _l, _c, template in rows]
    kept_index, held_index = next(splitter.split(rows, categories, templates))
    return [rows[i] for i in kept_index], [rows[i] for i in held_index]


def _strip(rows):
    """Drop the template id; detectors take (text, label, category)."""
    return [(text, label, category) for text, label, category, _t in rows]


def load_split():
    """Return (train, test) as lists of (text, label, category)."""
    train, test = _hold_out(load_rows(), TEST_FOLDS)
    return _strip(train), _strip(test)


def load_train_validation():
    """Return (train, validation) from inside the training half.

    The validation half picks the best epoch during fine-tuning. It is split
    by template for the same reason as the test half: choosing an epoch on
    templates the model trains on rewards memorising them.
    """
    train, _test = _hold_out(load_rows(), TEST_FOLDS)
    train, validation = _hold_out(train, VALIDATION_FOLDS)
    return _strip(train), _strip(validation)


def template_overlap():
    """Count test templates that also appear in the training half. Should be 0."""
    train, test = _hold_out(load_rows(), TEST_FOLDS)
    return len({r[3] for r in train} & {r[3] for r in test})


if __name__ == "__main__":
    rows = load_rows()
    train, test = _hold_out(rows, TEST_FOLDS)
    print(f"Templates: {len({r[3] for r in rows})}")
    for name, half in [("train", train), ("test", test)]:
        injected = sum(1 for r in half if r[1] == 1)
        print(f"  {name:<5} {len(half):>4} rows from "
              f"{len({r[3] for r in half}):>2} templates, "
              f"{injected} injected, {len(half) - injected} clean")
    print(f"  templates in both halves: {template_overlap()}")
    print()
    print("Test templates per attack category:")
    per_category = {}
    for _t, _l, category, template in test:
        per_category.setdefault(category, set()).add(template)
    for category, found in sorted(per_category.items()):
        print(f"  {category:<24} {len(found)}")

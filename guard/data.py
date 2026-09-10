"""Load the corpus and split it.

Every detector and the training script import the split from here. If each of
them called train_test_split on its own, a chunk could land in the training
half for one and the test half for another, and the numbers would stop being
comparable.
"""

import csv
from pathlib import Path

from sklearn.model_selection import train_test_split

DATASET_PATH = Path("data/injections/dataset.csv")

# Fixed so the split survives a rerun. Different from the seed in
# build_dataset.py, which controls a different thing: what the corpus contains.
SPLIT_SEED = 42
TEST_SIZE = 0.2


def load_rows():
    """Read the corpus as a list of (text, label, category)."""
    with DATASET_PATH.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [(row["text"], int(row["label"]), row["category"])
                for row in reader]


def load_split():
    """Return (train, test), stratified on the label.

    Stratifying keeps the ratio of injected to clean the same in both halves.
    Without it a random split can hand the test half an unrepresentative mix,
    and recall computed on it says little.
    """
    rows = load_rows()
    labels = [label for _text, label, _category in rows]
    train, test = train_test_split(
        rows,
        test_size=TEST_SIZE,
        random_state=SPLIT_SEED,
        stratify=labels,
    )
    return train, test


if __name__ == "__main__":
    train, test = load_split()
    for name, half in [("train", train), ("test", test)]:
        injected = sum(1 for _t, label, _c in half if label == 1)
        print(f"{name}: {len(half)} rows, {injected} injected, "
              f"{len(half) - injected} clean")

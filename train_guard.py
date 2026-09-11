"""Fine-tune DistilBERT with LoRA to classify chunks as clean or injected.

Follows the structure of the course notebook finetune_financial_sentiment.ipynb,
section by section. The differences, and the reason for each:

1. eval_strategy instead of evaluation_strategy. transformers 5 renamed it.
2. No 4-bit loading and no paged 8-bit optimizer. Both need a CUDA GPU and the
   bitsandbytes library; this trains on CPU in full precision.
3. A validation set carved out of the training half. The notebook scores the
   test set after every epoch and keeps the best epoch by that score, which
   tunes the model to the test set. Here the epoch is picked on validation,
   and the test set stays untouched until evaluate_guards.py. Both are split
   by template rather than by row; guard/data.py explains why.
4. Two labels instead of three, and binary precision/recall/F1 for the
   injection class instead of weighted F1 across sentiment classes.

    python train_guard.py

Takes about ten minutes on a laptop CPU. Writes the adapter to models/guard_lora
"""

import numpy as np
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from guard.data import SPLIT_SEED, load_train_validation

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "models/guard_lora"
MAX_LENGTH = 128

set_seed(SPLIT_SEED)

# ---------------------------------------------------------------------------
# 1. Data preparation
# ---------------------------------------------------------------------------

# Both halves come split by template from guard/data.py. The test half is
# never loaded in this file.
train_rows, val_rows = load_train_validation()


def to_dataset(rows):
    return Dataset.from_dict({
        "text": [text for text, _label, _category in rows],
        "label": [label for _text, label, _category in rows],
    })


dataset = {"train": to_dataset(train_rows), "validation": to_dataset(val_rows)}
print(f"Train: {len(train_rows)} rows. Validation: {len(val_rows)} rows.")

# ---------------------------------------------------------------------------
# 2. Tokenization
# ---------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


tokenized = {name: split.map(tokenize_function, batched=True)
             for name, split in dataset.items()}
for split in tokenized.values():
    split.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# ---------------------------------------------------------------------------
# 3. Model and PEFT configuration
# ---------------------------------------------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "clean", 1: "injection"},
    label2id={"clean": 0, "injection": 1},
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_lin", "v_lin"],   # query and value projections in attention
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
    # DistilBERT's two classification layers are created fresh for this task,
    # with random weights. LoRA adapts existing layers; these have nothing to
    # adapt, so they train in full.
    modules_to_save=["pre_classifier", "classifier"],
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

# ---------------------------------------------------------------------------
# 4. Training setup
# ---------------------------------------------------------------------------


def compute_metrics(prediction):
    logits, labels = prediction
    predicted = np.argmax(logits, axis=1)
    precision, recall, f1, _support = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0,
    )
    return {
        "accuracy": accuracy_score(labels, predicted),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


training_args = TrainingArguments(
    output_dir="models/checkpoints",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-4,
    num_train_epochs=3,
    eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    report_to="none",
    seed=SPLIT_SEED,
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics,
)

# ---------------------------------------------------------------------------
# 5. Run training
# ---------------------------------------------------------------------------

trainer.train()

peft_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

final = trainer.evaluate()
print(f"\nAdapter saved to {OUTPUT_DIR}")
print("Validation, best epoch:")
for key in ["eval_accuracy", "eval_precision", "eval_recall", "eval_f1"]:
    print(f"  {key[5:]:<10} {final[key]:.3f}")
print("\nTest-set numbers come from evaluate_guards.py, not from here.")

# Project proposal

**Guarded RAG — detecting prompt injection in documents retrieved by a
question-answering assistant**

Ruslan Sabitov — Student ID 99774243 — SE_AI

---

## 1. Motivation

Retrieval-augmented generation lets a language model answer questions over a
private document collection without retraining. The retrieved text is placed
into the model's prompt, where it is indistinguishable from the instructions
the developer wrote.

This is an unresolved weakness rather than an implementation mistake. Whoever
controls a document in the corpus controls part of the prompt. In a finance
setting the attack is concrete: a supplier submits an invoice whose text
instructs the reviewing model to approve it, and the model complies.

## 2. Objective

Build a working RAG assistant and add a guard stage between retrieval and
generation that classifies each retrieved chunk as clean or injected.
Injected chunks are withheld from the prompt and reported to the user rather
than silently obeyed.

Measure how well that guard works.

## 3. System design

| Stage | Technology |
|---|---|
| Document loading and chunking | LangChain, RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers |
| Vector store | ChromaDB |
| Guard | classifier (see section 4) |
| Generation | LLM over the surviving chunks |
| Interface | FastAPI endpoint `/ask` |

## 4. The classifier

Two stages, built and compared:

**Stage 1 — heuristic baseline.** Keyword and pattern matching over known
injection phrasing. Transparent, instant, and expected to fail on rephrased
attacks. The failure is the reason stage 2 exists.

**Stage 2 — DistilBERT fine-tuned with LoRA.** Parameter-efficient
fine-tuning (PEFT) on a labelled corpus of clean and injected document
chunks, following the fine-tuning approach covered in the course.

## 5. Data

Public prompt-injection datasets, extended with synthetic invoice and
expense-claim chunks written to cover the document domain. Both classes are
labelled; the split is stratified.

## 6. Evaluation

Precision, recall, F1 and a confusion matrix, reported for both stages on
the same held-out test set.

Recall is weighted above precision. A missed injection is an approved
fraudulent payment; a false positive costs a clerk one manual look. This
asymmetry is the reason the decision threshold is not left at its default.

## 7. Error analysis

Robustness against obfuscation — paraphrasing, inserted whitespace, and
language switching — reported honestly, including the cases the guard does
not catch.

## 8. Deliverables

- This repository, with commit history covering the development period
- A written report of the design, results and error analysis
- A FastAPI service exposing the guarded assistant
- A live demonstration contrasting the assistant with the guard disabled
  and enabled on the same document

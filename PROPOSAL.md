# Project proposal

**Guarded RAG: detecting prompt injection in documents retrieved by a
question-answering assistant**

Ruslan Sabitov. Student ID 99774243. SE_AI.

---

## 1. Motivation

Retrieval-augmented generation lets a language model answer questions about a
private document collection without anyone retraining it. The pipeline pastes
the retrieved text into the prompt, where it sits next to the instructions the
developer wrote and looks exactly like them.

No implementation fixes this. Whoever controls a document in the corpus
controls part of the prompt. A supplier submits an invoice whose text tells the
reviewing model to approve it, and the model approves it.

## 2. Objective

Build a working RAG assistant, then add a guard between retrieval and
generation that classifies each retrieved chunk as clean or injected. The guard
keeps injected chunks out of the prompt and names them in the answer.

Then measure how well the guard works.

## 3. System design

| Stage | Technology |
|---|---|
| Document loading and chunking | LangChain, RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers, run locally |
| Vector store | ChromaDB |
| Guard | classifier (see section 4) |
| Generation | LLM over the surviving chunks |
| Interface | FastAPI endpoint `/ask` |

## 4. The classifier

Two stages, built and measured against each other:

**Stage 1, heuristics.** Keyword and pattern matching over known injection
phrasing. The rule set fits on one screen and runs in microseconds. Rephrase an
attack outside its patterns and it misses, which is the reason stage 2 exists.

**Stage 2, DistilBERT fine-tuned with LoRA.** Parameter-efficient fine-tuning
(PEFT) on a labelled corpus of clean and injected document chunks, following
the fine-tuning approach the course covers.

**Reference point.** Meta publishes Prompt Guard 2, a general-purpose injection
classifier. Running it over the same test set answers the obvious question: why
train your own model when someone has already published one?

## 5. Data

`build_dataset.py` generates 767 labelled chunks from templates under a fixed
seed, so anyone with the file rebuilds the corpus byte for byte.

The clean class holds invoice text and hard negatives, meaning polite requests
aimed at a person, such as "Please disregard the previous version of this
document." Without those the classifier settles on "imperative verb means
attack" and blocks half the real documents it sees.

The injected class covers six attack types: instruction override, role
spoofing, output manipulation, finding suppression, prompt exfiltration and
delimiter spoofing. Most injections sit wrapped inside ordinary invoice lines,
so the classifier cannot key on a chunk looking short and strange.

## 6. Evaluation

Precision, recall, F1 and a confusion matrix on the same held-out split for
every stage.

Recall carries more weight than precision. Miss an injection and someone pays a
fraudulent invoice. Raise a false alarm and a clerk spends thirty seconds
looking at a clean document. That gap is why the decision threshold does not
stay at its default.

## 7. Error analysis

Which attack types slip through, broken down by category, and what happens when
an attacker paraphrases or pads an injection with whitespace. The report names
the cases the guard misses.

## 8. Deliverables

- This repository, with commit history covering the development period
- A written report of the design, results and error analysis
- A FastAPI service exposing the guarded assistant
- A live demonstration of the same document answered with the guard off, then
  with it on

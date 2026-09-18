# Architecture

Guarded RAG: a document question-answering assistant that classifies every
retrieved chunk for prompt injection before the chunk reaches the model prompt.

Ruslan Sabitov. Student ID 99774243. SE_AI.

---

## Component map

| File | Responsibility | Reads | Writes |
|---|---|---|---|
| `build_dataset.py` | Generates the labelled corpus from templates under a fixed seed | template lists in the file | `data/injections/dataset.csv` |
| `guard/data.py` | Loads the corpus and splits it by template | `dataset.csv` | nothing |
| `guard/heuristics.py` | Rule-based detector, 27 regular expressions | the chunk passed to it | nothing |
| `guard/promptguard.py` | Wrapper around Meta's Prompt Guard 2, with an on-disk cache | Groq API | `results/promptguard_cache.json` |
| `guard/distilbert.py` | The trained detector that the pipeline uses | `models/guard_lora` | nothing |
| `train_guard.py` | Fine-tunes DistilBERT with LoRA | corpus via `guard/data.py` | `models/guard_lora`, `results/train_log.txt` |
| `evaluate_guards.py` | Scores all three detectors on one test split | corpus, all detectors | `results/metrics.md` |
| `ingest.py` | Chunks documents, embeds them, builds the vector store | `data/documents/` | `chroma_db/` |
| `chat.py` | The assistant: retrieval, guard, generation | `chroma_db/`, `models/guard_lora` | nothing |
| `demo.py` | Runs one question in three modes and records the transcript | same as `chat.py` | `results/demo_three_modes.md` |
| `main.py` | The same assistant over HTTP | same as `chat.py` | nothing |
| `config.py` | Model names, chunk size, retrieval depth, token budget | `.env` | nothing |

`guard/heuristics.py` and `guard/promptguard.py` never run inside the
assistant. They exist so the trained detector has something to be measured
against.

## Two pipelines

The project has a build pipeline that runs once, and a request pipeline that
runs per question.

**Build**

```
build_dataset.py  ──►  data/injections/dataset.csv      1152 chunks, 75 templates
train_guard.py    ──►  models/guard_lora                3 epochs, about 5 minutes on a CPU
ingest.py         ──►  chroma_db/                       5 chunks from one invoice
evaluate_guards.py ──►  results/metrics.md              three detectors, one test split
```

**Request**

```
question
   │
   ├─► MiniLM embeds the question
   │
   ├─► Chroma returns the 4 nearest chunks
   │
   ├─► GUARD: DistilBERT scores each chunk
   │        below 0.5  ──►  the chunk enters the prompt
   │        0.5 and up ──►  the chunk is withheld and named in the answer
   │
   ├─► gpt-oss-20b answers from the surviving chunks
   │
   └─► answer, chunks used, chunks withheld with their scores
```

## What happens on one request

`chat.answer()` holds the whole path in nine lines.

1. `store.similarity_search(question, k=4)` returns the four chunks closest to
   the question. This step is identical in all three demonstration modes.
2. When a detector is passed in, `screen()` scores each chunk. The score is the
   softmax probability of the injection class.
3. Chunks at or above 0.5 go to the withheld list. The rest stay.
4. `build_prompt()` concatenates the surviving chunks into the system prompt,
   choosing the plain or the defensive template.
5. `llm.invoke()` sends the prompt to Groq.
6. `report()` prints the answer, then every withheld chunk with its score.

The guard sits between steps 1 and 4. A flagged chunk never becomes tokens the
generation model reads.

## Models

| Model | Role | Where it runs |
|---|---|---|
| `all-MiniLM-L6-v2` | Embeds chunks and questions into 384 dimensions for retrieval | locally, CPU |
| `distilbert-base-uncased` + LoRA adapter | The guard. One chunk in, one probability out | locally, CPU |
| `openai/gpt-oss-20b` | Writes the answer | Groq API |
| `llama-prompt-guard-2-86m` | Baseline in the evaluation only | Groq API |

Retrieval and the guard need no network. Only generation does.

## Design decisions

**The guard runs before the model, not inside it.** A prompt instruction
telling the model to treat context as data lives inside the model and depends
on how that model reads it on the day. The classifier produces a score that can
be logged per chunk, and it returns the same score when the generation model is
swapped. The demonstration measures both: the prompt defence held on
12 September and failed on 13 September with identical input, while the
classifier scored the same chunk 0.996 both times.

**Classification happens per chunk, not per document.** Blocking at document
level would discard the whole invoice over one line. The cost of the chunk
level shows in the transcript: line item 3 carries the injection and a
legitimate 420.00 EUR charge in the same 500 characters, and withholding the
chunk removes both. Sentence-level screening would cost the answer less.

**The corpus is split by template, not by row.** Rows generated from one
template differ only in an invoice number or a date, so they are not
independent. A row split put 191 of 231 test rows beside a twin in the training
half and produced a perfect 1.000 that meant nothing. `StratifiedGroupKFold`
groups on `template_id` and stratifies on attack category, so every attack type
reaches the test half and no phrasing appears on both sides.

**Recall is weighted above precision.** A missed injection is a fraudulent
invoice paid. A false alarm costs a clerk thirty seconds. The threshold stays
at 0.5 because the trained detector already reaches recall 1.000 there.

**Embeddings and the guard run locally.** No API key is needed to build the
vector store or to screen a chunk, which keeps the cost at zero and the guard
available without a network.

## Results

Test half: 298 chunks from 21 templates, 164 injected and 134 clean. The
fine-tuned model trained on none of those templates.

| Detector | Precision | Recall | F1 | Missed attacks | False alarms |
|---|---|---|---|---|---|
| Heuristics | 1.000 | 0.774 | 0.873 | 37 | 0 |
| Prompt Guard 2 (Meta) | 1.000 | 0.165 | 0.283 | 137 | 0 |
| DistilBERT + LoRA | 0.927 | 1.000 | 0.962 | 0 | 13 |

Misses by attack type:

| Attack type | Heuristics | Prompt Guard 2 | DistilBERT + LoRA |
|---|---|---|---|
| Role spoofing | 0% | 100% | 0% |
| Output manipulation | 50% | 100% | 0% |
| Finding suppression | 0% | 100% | 0% |
| Delimiter spoofing | 0% | 100% | 0% |
| Prompt exfiltration | 0% | 100% | 0% |
| Instruction override | 49% | 23% | 0% |

Meta trained Prompt Guard on chat jailbreaks, where an attack opens with
"ignore previous instructions". It handles that category and scores the other
five as ordinary text. An attack written as an internal note on an invoice
passes it.

`results/metrics.md` carries the confusion matrices, every false alarm, and the
caveats that belong with these numbers.

## The trained model

LoRA freezes the 67 million pretrained weights and adds two narrow matrices
beside the query and value projections in each attention block. Training
touches those, plus the two classification layers DistilBERT creates fresh for
a new task.

| Part | Trainable parameters |
|---|---|
| LoRA matrices, 6 layers x 2 projections | 147,456 |
| Classification head | 592,130 |
| **Total** | **739,586 of 67,694,596, or 1.09%** |

Three epochs on a CPU take about five minutes. Training loss falls from 0.666,
the loss of a coin flip between two classes, to 0.004. Validation loss rises
after the first epoch, 0.199 to 0.378 to 0.469, and the trainer keeps the first
epoch. That rise is the memorisation a row split had hidden.

## Interfaces

**Command line.** `python demo.py` runs one question three times and writes the
transcript. `python chat.py --guard` opens a prompt for arbitrary questions and
takes `--defensive` and `--question` as well.

**HTTP.** `uvicorn main:app --reload` exposes two routes. `GET /` reports which
models are loaded. `POST /ask` takes a question and returns the answer, the
number of chunks that reached the prompt, and every withheld chunk with its
score; `"guard": false` answers from the unguarded pipeline over the same
document. Interactive documentation sits at `/docs`. The embedding model, the
classifier and the Groq client load once at import, because loading them costs
seconds and answering costs one.

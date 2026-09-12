"""The guarded assistant as an HTTP service.

    uvicorn main:app --reload

POST /ask with {"question": "..."} returns the answer, the chunks that
reached the prompt, and the chunks the guard withheld. Interactive docs at
http://127.0.0.1:8000/docs

The guard runs on by default. Pass "guard": false to compare against the
unguarded pipeline over the same document.
"""

from fastapi import FastAPI
from langchain_chroma import Chroma
from pydantic import BaseModel

import chat
import config

app = FastAPI(
    title="Guarded RAG",
    description="Document question answering with prompt-injection screening.",
)

# Loaded once at import. The embedding model and the classifier each take a
# few seconds to load, and paying that on every request would dominate the
# response time.
store = Chroma(
    persist_directory=config.CHROMA_DIR,
    embedding_function=config.get_embeddings(),
)
llm = config.get_llm()

from guard.distilbert import DistilBertDetector  # noqa: E402

detector = DistilBertDetector()


class Query(BaseModel):
    question: str
    guard: bool = True
    defensive: bool = False


class WithheldChunk(BaseModel):
    score: float
    text: str


class Answer(BaseModel):
    answer: str
    chunks_used: int
    withheld: list[WithheldChunk]


@app.get("/")
def root():
    return {
        "service": "Guarded RAG",
        "model": config.GROQ_MODEL,
        "guard": detector.name,
        "docs": "/docs",
    }


@app.post("/ask", response_model=Answer)
def ask(query: Query):
    text, kept, withheld = chat.answer(
        query.question,
        store,
        llm,
        detector=detector if query.guard else None,
        defensive=query.defensive,
    )
    return Answer(
        answer=text,
        chunks_used=len(kept),
        withheld=[
            WithheldChunk(score=score, text=chunk.page_content)
            for chunk, score in withheld
        ],
    )

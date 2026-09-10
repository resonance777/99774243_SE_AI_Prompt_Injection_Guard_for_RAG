"""Shared configuration.

Both the embedding model and the generation model are named here so that
ingest.py and chat.py cannot drift apart. Changing the embedding model means
the vector store has to be rebuilt, because the stored vectors have a
different dimensionality.
"""

import os

from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "chroma_db"
DOCUMENTS_DIR = "data/documents"

# 500 characters with 50 of overlap, as used in the course notebook. The
# overlap exists so that a sentence split across two chunks still appears
# whole in one of them.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4

# Runs locally on the CPU. 384-dimensional vectors, no API call and no cost,
# matching the local-embedding approach in the course's RAG chatbot guide.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Groq exposes an OpenAI-compatible API, so the OpenAI client works unchanged
# once it is pointed at their base URL.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_llm(temperature=0):
    from langchain_openai import ChatOpenAI

    if not GROQ_API_KEY:
        raise SystemExit(
            "GROQ_API_KEY is not set. Copy .env.example to .env and put your "
            "key there. A free key comes from https://console.groq.com"
        )
    return ChatOpenAI(
        model=GROQ_MODEL,
        base_url=GROQ_BASE_URL,
        api_key=GROQ_API_KEY,
        temperature=temperature,
    )

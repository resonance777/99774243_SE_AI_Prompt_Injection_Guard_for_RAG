"""Build the vector store.

Reads every document in data/documents, splits it into chunks, embeds each
chunk with OpenAI, and writes the result to a local Chroma database.

Run once before chat.py, and again whenever the documents change:

    python ingest.py
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DOCUMENTS_DIR = Path("data/documents")
CHROMA_DIR = "chroma_db"

# 500 characters with 50 of overlap, as used in the course notebook. The
# overlap exists so that a sentence split across two chunks still appears
# whole in one of them.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

EMBEDDING_MODEL = "text-embedding-3-small"


def load_documents():
    """Load every .pdf and .txt file under data/documents."""
    documents = []
    for path in sorted(DOCUMENTS_DIR.iterdir()):
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        elif path.suffix.lower() == ".txt":
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            print(f"  skipped (unsupported type): {path.name}")
            continue
        loaded = loader.load()
        documents.extend(loaded)
        print(f"  loaded {path.name} ({len(loaded)} page(s))")
    return documents


def main():
    if not DOCUMENTS_DIR.exists() or not any(DOCUMENTS_DIR.iterdir()):
        raise SystemExit(
            f"No documents found in {DOCUMENTS_DIR}. "
            "Put at least one .pdf or .txt file there first."
        )

    print(f"Loading documents from {DOCUMENTS_DIR}/")
    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} document(s) into {len(chunks)} chunks")

    print(f"Embedding with {EMBEDDING_MODEL} and writing to {CHROMA_DIR}/")
    Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=CHROMA_DIR,
    )
    print("Done.")


if __name__ == "__main__":
    main()

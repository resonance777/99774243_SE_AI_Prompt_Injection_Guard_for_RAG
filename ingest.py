"""Build the vector store.

Reads every document in data/documents, splits it into chunks, embeds each
chunk locally, and writes the result to a local Chroma database.

Run once before chat.py, and again whenever the documents change:

    python ingest.py
"""

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def load_documents(directory):
    """Load every .pdf and .txt file in the given directory."""
    documents = []
    for path in sorted(Path(directory).iterdir()):
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
    directory = Path(config.DOCUMENTS_DIR)
    if not directory.exists() or not any(directory.iterdir()):
        raise SystemExit(
            f"No documents found in {directory}. "
            "Put at least one .pdf or .txt file there first."
        )

    print(f"Loading documents from {directory}/")
    documents = load_documents(directory)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} document(s) into {len(chunks)} chunks")

    print(f"Embedding locally with {config.EMBEDDING_MODEL}")
    print("(the first run downloads the model, roughly 90 MB)")
    # Chroma appends to an existing store rather than replacing it, so a second
    # run would hold every chunk twice. Start from an empty store each time.
    shutil.rmtree(config.CHROMA_DIR, ignore_errors=True)
    Chroma.from_documents(
        documents=chunks,
        embedding=config.get_embeddings(),
        persist_directory=config.CHROMA_DIR,
    )
    print(f"Written to {config.CHROMA_DIR}/. Done.")


if __name__ == "__main__":
    main()

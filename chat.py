"""Ask questions about the ingested documents.

This is the UNGUARDED baseline: every retrieved chunk goes straight into the
prompt. It exists so that the guard added later has something to be compared
against, and so the demo can show both behaviours on the same document.

    python chat.py
"""

from langchain_chroma import Chroma

import config

SYSTEM_PROMPT = """You are a finance assistant. Answer the user's question \
using only the context below.

Context:
{context}"""


def build_prompt(question, chunks):
    context = "\n\n---\n\n".join(c.page_content for c in chunks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": question},
    ]


def main():
    store = Chroma(
        persist_directory=config.CHROMA_DIR,
        embedding_function=config.get_embeddings(),
    )
    llm = config.get_llm()

    print(f"Model: {config.GROQ_MODEL}")
    print("Ask a question about the documents. Type 'exit' to quit.\n")
    while True:
        question = input("> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        chunks = store.similarity_search(question, k=config.TOP_K)
        answer = llm.invoke(build_prompt(question, chunks))

        print(f"\n{answer.content}\n")
        print(f"[retrieved {len(chunks)} chunks, none filtered]\n")


if __name__ == "__main__":
    main()

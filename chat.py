"""Ask questions about the ingested documents.

This is the UNGUARDED baseline: every retrieved chunk goes straight into the
prompt. It exists so that the guard added later has something to be compared
against, and so the demo can show both behaviours on the same document.

    python chat.py
"""

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 4

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
        persist_directory=CHROMA_DIR,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    print("Ask a question about the documents. Type 'exit' to quit.\n")
    while True:
        question = input("> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        chunks = store.similarity_search(question, k=TOP_K)
        answer = llm.invoke(build_prompt(question, chunks))

        print(f"\n{answer.content}\n")
        print(f"[retrieved {len(chunks)} chunks, none filtered]\n")


if __name__ == "__main__":
    main()

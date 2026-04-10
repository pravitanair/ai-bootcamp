import os
from dotenv import load_dotenv
from openai import OpenAI

from rag.chunking import chunk_text
from rag.embeddings import get_embedding
from rag.index import build_faiss_index
from rag.retrieve import search_index

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("data/notes.txt", "r") as f:
    full_text = f.read()

documents = chunk_text(full_text, chunk_size=80, overlap=15)
doc_embeddings = [get_embedding(doc) for doc in documents]
index = build_faiss_index(doc_embeddings)

query = input("Ask: ")
query_embedding = get_embedding(query)

results = search_index(index, query_embedding, documents, top_k=5)
context = "\n\n".join([r["text"] for r in results])

print("\nRetrieved Chunks:\n")
for r in results:
    print("-", r["text"])
    print("  distance:", r["distance"])

response = client.responses.create(
    model="gpt-4.1-mini",
    input=[
        {
            "role": "system",
            "content": (
                "Answer only from the provided context. "
                "If the answer is not in the context, say 'insufficient data'."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }
    ]
)

print("\nAnswer:\n")
print(response.output_text)
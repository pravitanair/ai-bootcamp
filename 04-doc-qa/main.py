import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


with open("data.txt", "r") as f:
    documents = [line.strip() for line in f.readlines() if line.strip()]


# Step 1: Create embeddings

doc_embeddings = []

query_embeddings = []


for doc in documents:
    response = client.embeddings.create(model="text-embedding-3-small",input=doc)
    doc_embeddings.append(response.data[0].embedding)

# Step 2: User query
query = input("Enter your query: ")

query_embedding = client.embeddings.create(model="text-embedding-3-small",input=query).data[0].embedding


# Step 3: Compare similarity

def cosine_similarity(a,b):
    return np.dot(a,b) / np.linalg.norm(a) * np.linalg.norm(b)

scores = []

for i, doc_emb in enumerate(doc_embeddings):
    score = cosine_similarity(query_embedding, doc_emb)
    scores.append((score, documents[i]))

# Step 4: Sort results
scores.sort(reverse=True)

scores.sort(reverse=True)
top_docs = [doc for _, doc in scores[:2]]

# Combine context
context = "\n".join(top_docs)

# LLM call with grounding
response = client.responses.create(
    model="gpt-4.1-mini",
    input=[
        {
            "role": "system",
            "content": "Answer ONLY using the provided context. If not found, say 'insufficient data'."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }
    ]
)

print("\nAnswer:\n", response.output_text)







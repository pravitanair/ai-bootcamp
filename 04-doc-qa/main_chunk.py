import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chunk_text(text, chunk_size=100, overlap=20):
    """
    Breaks a long string of text into smaller, manageable pieces (chunks) while 
    ensuring they stay connected through overlapping words. This is commonly 
    used in AI and data processing to keep context between fragments.

    Example (100-word text, chunk_size=50, overlap=10):
    - Chunk 1: Words 0 to 50
    - Chunk 2: Words 40 to 90 (starts at 40 because 50 - 10 = 40)
    - Chunk 3: Words 80 to 100
    """
    # Splitting: Turns the text into a list of individual words
    words = text.split()
    chunks = []
    
    # Looping with Offset: Moves through the list using a step size of 
    # chunk_size - overlap so each chunk starts before the previous one ended
    for i in range(0, len(words), chunk_size - overlap):
        # Joining: Grabs a slice of words and joins them back into a single string
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)

    # Result: Returns a list of text strings with shared words at the boundaries
    return chunks

with open("data.txt", "r") as f:
    full_text = f.read()


documents = chunk_text(full_text)


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
top_docs = [doc for _, doc in scores[:3]]

# Combine context
context = "\n".join(top_docs)

# LLM call with grounding
response = client.responses.create(
    model="gpt-4.1-mini",
    input=[
        {
                "role": "system",
                "content": """
        You are a healthcare AI assistant.

        Rules:
        - Answer ONLY using the provided context
        - Do not use external knowledge
        - If the answer is not explicitly in the context, say "insufficient data"
        - Be concise and structured
        """
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }
    ]
)

print("\nRetrieved Context:\n", context)

print("\nAnswer:\n", response.output_text)







import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate


load_dotenv()

# Load documents
documents = SimpleDirectoryReader("data").load_data()

# Create index
# This line just replaced: 
# chunking 
# embedding generation
# vector storage
# retrieval setup

index = VectorStoreIndex.from_documents(documents)

qa_template = PromptTemplate(
    "Use the context to answer the question. "
    "You may combine multiple facts. "
    "If insufficient, say so.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n"
    "Answer:"
)

# Create query engine
query_engine = index.as_query_engine(
    text_qa_template=qa_template
)
# Ask question
query = input("Ask: ")

response = query_engine.query(query)

print("\nAnswer:\n")
print(response)
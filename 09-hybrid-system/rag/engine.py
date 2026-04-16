from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate

def build_query_engine():
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)

    qa_template = PromptTemplate(
        "Use the context to answer the question. "
        "You may combine multiple facts. "
        "If insufficient, say so.\n\n"
        "Context:\n{context_str}\n\n"
        "Question: {query_str}\n"
        "Answer:"
    )

    return index.as_query_engine(text_qa_template=qa_template)
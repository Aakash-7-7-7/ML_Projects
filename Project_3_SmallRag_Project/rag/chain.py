from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_groq import ChatGroq

def rag_agent(retriever,api_key):
    llm=ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0
    )
    prompt=ChatPromptTemplate.from_template("""
    You are a professional AI assistant.

    Use ONLY the provided context to answer.
    If not found, say: "The answer is not available in the provided documents."

    Context:
    {context}

    Question:
    {input}
    """)

    document_chain=create_stuff_documents_chain(llm,prompt)
    return create_retrieval_chain(retriever,document_chain)
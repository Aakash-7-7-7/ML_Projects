from langchain_community.vectorstores import Chroma

def create_vectorstore(documents,embedding):
    return Chroma.from_documents(
        documents,
        embedding,
        persist_directory="chroma_db"
    )
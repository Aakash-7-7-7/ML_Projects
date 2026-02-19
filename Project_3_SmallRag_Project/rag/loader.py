from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_doc(path='docs'):
    loader=DirectoryLoader(
        path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True
    )

    documents=loader.load()

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    return splitter.split_documents(documents)
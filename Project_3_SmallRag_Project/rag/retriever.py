def retrieve(vectorstore):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_lwargs={"k":5}
    )
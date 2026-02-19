def retrieve(vectorstore):
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.6
        }
    )

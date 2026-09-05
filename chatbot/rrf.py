"""
Reciprocal Rank Fusion (RRF)

Combines multiple ranked retrieval lists into
a single ranking.
"""

RRF_K = 60


def reciprocal_rank_fusion(rank_lists, k=RRF_K):

    scores = {}
    documents = {}

    for ranked_list in rank_lists:

        for rank, doc in enumerate(ranked_list, start=1):

            doc_id = doc["id"]

            if doc_id not in documents:
                documents[doc_id] = doc

            scores[doc_id] = scores.get(doc_id, 0) + (
                1 / (k + rank)
            )

    fused_documents = []

    for doc_id, score in scores.items():

        document = documents[doc_id].copy()

        document["rrf_score"] = score

        fused_documents.append(document)

    fused_documents.sort(

        key=lambda x: x["rrf_score"],

        reverse=True

    )

    return fused_documents
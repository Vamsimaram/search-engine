import numpy as np

def calculate_ndcg(results, qrels, query_id, k=10):
    # Calculate nDCG@k for a single query
    relevant_docs = {doc_id: rel_score for q_id, doc_id, rel_score in qrels if q_id == query_id}
    
    # Calculate DCG
    dcg = 0
    for i, (doc_idx, _) in enumerate(results[:k]):
        doc_id = doc_idx + 1  # Assuming doc_ids start from 1
        if doc_id in relevant_docs:
            # Using relevance as gain (0-4 in Cranfield)
            rel = relevant_docs[doc_id]
            # Position is i+1 (0-indexed to 1-indexed)
            dcg += rel / np.log2(i + 2)  # log_2(1+1) = 1, so we use i+2
    
    # Calculate ideal DCG
    ideal_relevances = sorted([rel for _, rel in relevant_docs.items()], reverse=True)
    idcg = 0
    for i, rel in enumerate(ideal_relevances[:k]):
        idcg += rel / np.log2(i + 2)
    
    # Handle case where there are no relevant documents
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

def evaluate_search_engine(search_engine, dataset, k=10):
    # Evaluate search engine using nDCG@k
    ndcg_scores = []
    
    for query_id, query in zip(dataset['query_ids'], dataset['queries']):
        results = search_engine.search(query, top_k=k)
        ndcg = calculate_ndcg(results, dataset['qrels'], query_id, k=k)
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores)
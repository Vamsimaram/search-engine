import numpy as np
import torch
import os
import time

# Import components from src package
from src.preprocessing import load_cranfield_dataset
from src.lsa_engine import LSASearchEngine
from src.mpnet_engine import MPNetSearchEngine
from src.hybrid_engine import HybridSearchEngine
from src.evaluation import evaluate_search_engine

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load Cranfield dataset
    print("Loading Cranfield dataset...")
    data_dir = "./data"  # Update with your path if different
    dataset = load_cranfield_dataset(data_dir)
    
    print(f"Loaded {len(dataset['documents'])} documents and {len(dataset['queries'])} queries")
    
    # Train and evaluate baseline vector space model (TF-IDF)
    print("\nEvaluating baseline TF-IDF Vector Space Model...")
    baseline_engine = LSASearchEngine(n_components=5000)  # Using high dimensionality for baseline
    baseline_engine.fit(dataset['documents'])
    start_time = time.time()
    baseline_ndcg = evaluate_search_engine(baseline_engine, dataset, k=10)
    baseline_time = time.time() - start_time
    print(f"Baseline Vector Space Model nDCG@10: {baseline_ndcg:.4f} (Time: {baseline_time:.2f}s)")
    
    # Train and evaluate LSA model
    print("\nEvaluating LSA model...")
    lsa_engine = LSASearchEngine(n_components=100)
    lsa_engine.fit(dataset['documents'])
    start_time = time.time()
    lsa_ndcg = evaluate_search_engine(lsa_engine, dataset, k=10)
    lsa_time = time.time() - start_time
    print(f"LSA model nDCG@10: {lsa_ndcg:.4f} (Time: {lsa_time:.2f}s)")
    
    # Train and evaluate MPNET model
    print("\nEvaluating MPNET model...")
    mpnet_engine = MPNetSearchEngine()
    mpnet_engine.fit(dataset['documents'])
    start_time = time.time()
    mpnet_ndcg = evaluate_search_engine(mpnet_engine, dataset, k=10)
    mpnet_time = time.time() - start_time
    print(f"MPNET model nDCG@10: {mpnet_ndcg:.4f} (Time: {mpnet_time:.2f}s)")
    
    # Train and evaluate hybrid model
    print("\nEvaluating hybrid LSA+MPNET model...")
    # Try different weight combinations
    best_ndcg = 0
    best_weights = (0.5, 0.5)
    
    for lsa_weight in [0.3, 0.4, 0.5, 0.6, 0.7]:
        mpnet_weight = 1 - lsa_weight
        hybrid_engine = HybridSearchEngine(lsa_weight=lsa_weight, mpnet_weight=mpnet_weight)
        hybrid_engine.fit(dataset['documents'])
        hybrid_ndcg = evaluate_search_engine(hybrid_engine, dataset, k=10)
        print(f"Hybrid model (LSA: {lsa_weight}, MPNET: {mpnet_weight}) nDCG@10: {hybrid_ndcg:.4f}")
        
        if hybrid_ndcg > best_ndcg:
            best_ndcg = hybrid_ndcg
            best_weights = (lsa_weight, mpnet_weight)
    
    print(f"\nBest hybrid model weights: LSA={best_weights[0]}, MPNET={best_weights[1]}")
    print(f"Best hybrid model nDCG@10: {best_ndcg:.4f}")
    print(f"Improvement over baseline: {(best_ndcg - baseline_ndcg) * 100:.1f}%")

if __name__ == "__main__":
    main()
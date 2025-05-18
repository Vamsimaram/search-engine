from .lsa_engine import LSASearchEngine
from .mpnet_engine import MPNetSearchEngine

class HybridSearchEngine:
    def __init__(self, lsa_weight=0.4, mpnet_weight=0.6):
        self.lsa_engine = LSASearchEngine(n_components=100)
        self.mpnet_engine = MPNetSearchEngine()
        self.lsa_weight = lsa_weight
        self.mpnet_weight = mpnet_weight
        
    def fit(self, documents):
        # Train both LSA and MPNET models
        print("Training LSA model...")
        self.lsa_engine.fit(documents)
        
        print("Training MPNET model...")
        self.mpnet_engine.fit(documents)
        
        self.documents = documents
    
    def search(self, query, top_k=10):
        # Combine results from both models
        lsa_results = self.lsa_engine.search(query, top_k=top_k*2)
        mpnet_results = self.mpnet_engine.search(query, top_k=top_k*2)
        combined_scores = {}
        
        for idx, score in lsa_results:
            combined_scores[idx] = self.lsa_weight * score
            
        for idx, score in mpnet_results:
            if idx in combined_scores:
                combined_scores[idx] += self.mpnet_weight * score
            else:
                combined_scores[idx] = self.mpnet_weight * score
        
        # Sort by score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return sorted_results
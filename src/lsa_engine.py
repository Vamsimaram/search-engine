import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

class LSASearchEngine:
    def __init__(self, n_components=100):
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.svd = TruncatedSVD(n_components=n_components)
    
    def fit(self, documents):
        # Train the LSA model on documents
        # Create TF-IDF representations
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        # Apply SVD for dimensionality reduction (LSA)
        self.lsa_matrix = self.svd.fit_transform(self.tfidf_matrix)
        
        # Normalize LSA matrix for more accurate similarity calculations
        self.lsa_matrix = normalize(self.lsa_matrix)
        
        print(f"Explained variance ratio: {sum(self.svd.explained_variance_ratio_):.4f}")
    
    def transform_query(self, query):
        # Transform query to LSA space
        query_tfidf = self.vectorizer.transform([query])
        query_lsa = self.svd.transform(query_tfidf)
        query_lsa = normalize(query_lsa)
        return query_lsa
    
    def search(self, query, top_k=10):
        # Search for documents similar to query
        query_lsa = self.transform_query(query)
        
        # Calculate similarities
        similarities = cosine_similarity(query_lsa, self.lsa_matrix)[0]
        
        # Get top k results
        top_indices = np.argsort(-similarities)[:top_k]
        top_scores = similarities[top_indices]
        
        return list(zip(top_indices, top_scores))
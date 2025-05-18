import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

class MPNetSearchEngine:
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        self.model.eval()
    
    def mean_pooling(self, model_output, attention_mask):
        # Mean pool the token embeddings to get sentence embedding
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_embedding(self, text):
        # Get embeddings for a single text
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings.cpu().numpy()
    
    def get_embeddings(self, texts, batch_size=8):
        # Get embeddings for a list of texts in batches
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch_texts = texts[i:i+batch_size]
            encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            all_embeddings.append(sentence_embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def fit(self, documents):
        # Create embeddings for all documents
        self.document_embeddings = self.get_embeddings(documents)
        self.document_embeddings = normalize(self.document_embeddings)
    
    def search(self, query, top_k=10):
        # Search for documents similar to query
        query_embedding = self.get_embedding(query)
        query_embedding = normalize(query_embedding)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
        
        # Get top k results
        top_indices = np.argsort(-similarities)[:top_k]
        top_scores = similarities[top_indices]
        
        return list(zip(top_indices, top_scores))
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def preprocess_text(text):
    # Simplified preprocessing: lowercase, remove special chars, simple tokenization
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Simple tokenization by splitting on whitespace
    tokens = text.split()
    
    # Return as joined tokens - no stemming or stopword removal for now
    return ' '.join(tokens)

def load_cranfield_dataset(data_dir):
    documents = []
    doc_ids = []
    
    # Load documents
    try:
        with open(os.path.join(data_dir, 'cran.all.1400'), 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try another encoding if UTF-8 fails
        with open(os.path.join(data_dir, 'cran.all.1400'), 'r', encoding='latin-1') as f:
            content = f.read()
    
    # Parse documents - Cranfield format has specific markers
    doc_pattern = r'.I (\d+)\n.T\n(.*?)\n.A\n(.*?)\n.B\n(.*?)\n.W\n(.*?)(?=.I|\Z)'
    matches = re.findall(doc_pattern, content, re.DOTALL)
    
    for match in matches:
        doc_id, title, author, biblio, text = match
        full_text = f"{title} {text}"  # Combine title and content
        documents.append(preprocess_text(full_text))
        doc_ids.append(int(doc_id))
    
    # Load queries
    queries = []
    query_ids = []
    try:
        with open(os.path.join(data_dir, 'cran.qry'), 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(os.path.join(data_dir, 'cran.qry'), 'r', encoding='latin-1') as f:
            content = f.read()
    
    query_pattern = r'.I (\d+)\n.W\n(.*?)(?=.I|\Z)'
    matches = re.findall(query_pattern, content, re.DOTALL)
    
    for match in matches:
        query_id, text = match
        queries.append(preprocess_text(text))
        query_ids.append(int(query_id))
    
    # Load relevance judgments
    qrels = []
    try:
        with open(os.path.join(data_dir, 'cranqrel'), 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                query_id = int(parts[0])
                doc_id = int(parts[1])
                rel_score = int(parts[2])
                qrels.append((query_id, doc_id, rel_score))
    except UnicodeDecodeError:
        with open(os.path.join(data_dir, 'cranqrel'), 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split()
                query_id = int(parts[0])
                doc_id = int(parts[1])
                rel_score = int(parts[2])
                qrels.append((query_id, doc_id, rel_score))
    
    return {
        'documents': documents,
        'doc_ids': doc_ids,
        'queries': queries,
        'query_ids': query_ids,
        'qrels': qrels
    }
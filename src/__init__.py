# Import main components to make them accessible when importing the package
from .lsa_engine import LSASearchEngine
from .mpnet_engine import MPNetSearchEngine
from .hybrid_engine import HybridSearchEngine
from .preprocessing import preprocess_text, load_cranfield_dataset
from .evaluation import calculate_ndcg, evaluate_search_engine

__version__ = '0.1.0'

# Export classes and functions
__all__ = [
    'LSASearchEngine',
    'MPNetSearchEngine',
    'HybridSearchEngine',
    'preprocess_text',
    'load_cranfield_dataset',
    'calculate_ndcg',
    'evaluate_search_engine'
]
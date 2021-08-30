from abc import ABC
from typing import Dict, List

class GenericPreprocessor(ABC):
    """Generic interface for preprocessing data."""
    
    def __init__(self):
        pass
    
    def preprocess(self, documents: List[str]) -> Dict[str, List[str]]:
        """Preprocess the documents. 
        'document_tokens' field must be present in output as they are used during clustering
        """
        raise NotImplementedError
        
    def embed(self, preprocessed_documents: Dict[str, List[str]]) -> Dict[str, List[float]]:
        """Take the preprocessed documents and embed them. 
        'document_embeddings' field must be present in output as they are used during clustering 
        """
        raise NotImplementedError
from abc import ABC
from typing import Any, Dict, Generator, List


class GenericPreprocessor(ABC):
    """Generic interface for preprocessing data."""

    def __init__(self):
        pass

    def preprocess(
        self, data: List[Dict[str, Any]]
    ):
        """Preprocess the documents.
        'document_tokens' field must be present in output as they are used during clustering
        """
        raise NotImplementedError

    def embed(
        self, preprocessed_documents: Dict[str, List[str]]
    ):
        """Take the preprocessed documents and embed them.
        'document_embeddings' field must be present in output as they are used during clustering
        """
        raise NotImplementedError

    def featurize(
        self, data: List[Dict[str, Any]]
    ):
        """Featurize the data. This function is directly called by the clustering_driver.py. 
        Importantly, each featurized data point must contain the following fields:
            - `id`: a unique identifier associated with each data point 
            - `text`: the raw input text
            - `clustering_tokens`: the (preprocessed) tokens that will be input to clustering
            - `embedding`: the embedding associated with the data point.
        """
        raise NotImplementedError

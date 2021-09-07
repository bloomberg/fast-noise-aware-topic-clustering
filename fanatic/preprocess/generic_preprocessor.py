from abc import ABC
from typing import Any, Dict, Generator, List


class GenericPreprocessor(ABC):
    """Generic interface for preprocessing data."""

    def __init__(self):
        pass

    def preprocess(self, data: List[Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:
        """Preprocess the documents.
        'norm_tokens' field must be present in output if using the embedding_driver.py to train a Word2Vec model.

        Args:
            data: list of dicts containing the data.

        Returns:
            generator of the same data dict with the added required fields
        """
        raise NotImplementedError

    def featurize(self, data: List[Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:
        """Featurize the data. This function is directly called by the clustering_driver.py.
        Importantly, each featurized data point must contain the following fields:
            - `id`: a unique identifier associated with each data point
            - `text`: the raw input text
            - `clustering_tokens`: the (preprocessed) tokens that will be input to clustering
            - `embedding`: the embedding associated with the data point.

        Args:
            data: list of dicts containing the data

        Returns:
            generator of the same data dict with the added required fields
        """
        raise NotImplementedError

from fanatic.preprocess.generic_preprocessor import GenericPreprocessor
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np
from typing import Any, Dict, Generator, List
import re
from gensim.models import KeyedVectors

NUM_RE = re.compile(r"\d+[A-Za-z]{,2}")


class NLTKPreprocessor(GenericPreprocessor):
    def __init__(self, embedding_model_file, min_required_valid_tokens=3):
        # https://www.nltk.org/api/nltk.tokenize.html
        self.tokenizer = RegexpTokenizer(r"\w+")
        self.stopwords = stopwords.words("english")
        self.embedding_model = KeyedVectors.load_word2vec_format(
            embedding_model_file, binary=False
        )
        self.min_required_valid_tokens = min_required_valid_tokens
        super().__init__()

    def preprocess(
        self, data: List[Dict[str, Any]]
    ) -> Generator[Dict[str, Any], None, None]:
        for d in data:
            text = d["text"]
            d["tokens"] = [tok for tok in self.tokenizer.tokenize(text.lower())]
            d["norm_tokens"] = [
                "__NUMBER__" if NUM_RE.match(tok) else tok
                for tok in d["tokens"]
                if tok not in self.stopwords
            ]
            yield d

    def _get_averaged_embedding(self, clustering_tokens):
        vecs = [self.embedding_model[token] for token in clustering_tokens]
        averaged_vector = np.average(vecs, axis=0)
        return averaged_vector

    def embed(
        self, preprocessed_data_generator: Generator[Dict[str, Any], None, None]
    ) -> Generator[Dict[str, Any], None, None]:
        for d in preprocessed_data_generator:
            embedding_tokens = [
                tok for tok in d["norm_tokens"] if tok in self.embedding_model
            ]
            if len(embedding_tokens) >= self.min_required_valid_tokens:
                d["clustering_tokens"] = embedding_tokens
                d["embedding"] = self._get_averaged_embedding(embedding_tokens)
                yield d

    def featurize(
        self, data: List[Dict[str, Any]]
    ) -> Generator[Dict[str, Any], None, None]:
        """Combination of preprocess and embed. The required fields for downstream clustering are:
        `id`, `text`, `clustering_tokens`, `embedding`
        """
        preprocessed_data_generator = self.preprocess(data)
        embedding_data_generator = self.embed(preprocessed_data_generator)
        return embedding_data_generator

# Copyright 2021 Bloomberg L.P.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import re
from typing import Any, Dict, Generator, List, Optional, Set

import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from fanatic.preprocess.generic_preprocessor import GenericPreprocessor

logging_format = (
    "%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s"
)
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger(__name__)

NUM_RE = re.compile(r"\d+[A-Za-z]{,2}")


class NLTKPreprocessor(GenericPreprocessor):
    def __init__(
        self, embedding_model_file: Optional[str] = None, min_valid_tokens: int = 3
    ):
        # https://www.nltk.org/api/nltk.tokenize.html
        self.tokenizer = RegexpTokenizer(r"\w+")
        self.stopwords = stopwords.words("english")
        self.min_valid_tokens = min_valid_tokens
        if embedding_model_file is not None:
            self.embedding_model = KeyedVectors.load_word2vec_format(
                embedding_model_file, binary=False
            )
        else:
            self.embedding_model = None
            logger.warning(
                "embedding_model_file not provided to nltk preprocessor. Only `.preprocess` function will work."
            )
        super().__init__()

    def preprocess(
        self, data: List[Dict[str, Any]]
    ) -> Generator[Dict[str, Any], None, None]:
        """Preprocess the data.

        Args:
            data: the dataset

        Returns:
            (generator)
        """
        for d in data:
            text = d["text"]
            d["tokens"] = [tok for tok in self.tokenizer.tokenize(text.lower())]
            d["norm_tokens"] = [
                "__NUMBER__" if NUM_RE.match(tok) else tok
                for tok in d["tokens"]
                if tok not in self.stopwords
            ]
            yield d

    def _get_averaged_embedding(self, clustering_tokens: Set[str]) -> np.ndarray:
        """Get an embedding for each token, and the perform a simple average to get sentence-embedding.

        Args:
            clustering_tokens: the set of tokens to be embedded

        Returns:
            averaged_vector: the sentence-embedding.
        """
        vecs = [self.embedding_model[token] for token in clustering_tokens]
        averaged_vector = np.average(vecs, axis=0)
        return averaged_vector

    def embed(
        self, preprocessed_data_generator: Generator[Dict[str, Any], None, None]
    ) -> Generator[Dict[str, Any], None, None]:
        """Embed the preprocessed data.

        Args:
            preprocessed_data_generator: the preprocessed data generator

        Returns:
            (generator)
        """
        if self.embedding_model is None:
            raise ValueError(
                "No embedding model file was provided during init, cannot featurize data using nltk preprocessor."
            )

        for d in preprocessed_data_generator:
            embedding_tokens = set(
                [tok for tok in d["norm_tokens"] if tok in self.embedding_model]
            )
            if len(embedding_tokens) >= self.min_valid_tokens:
                d["clustering_tokens"] = embedding_tokens
                d["embedding"] = self._get_averaged_embedding(embedding_tokens)
                yield d

    def featurize(
        self, data: List[Dict[str, Any]]
    ) -> Generator[Dict[str, Any], None, None]:
        """Combination of preprocess and embed. Generates the required fields for downstream clustering:
            `id`, `text`, `clustering_tokens`, `embedding`.

        Args:
            data: the dataset

        Returns:
            (generator)
        """
        if self.embedding_model is None:
            raise ValueError(
                "No embedding model file was provided during init, cannot featurize data using nltk preprocessor."
            )

        preprocessed_data_generator = self.preprocess(data)
        embedding_data_generator = self.embed(preprocessed_data_generator)
        return embedding_data_generator

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

from abc import ABC
from typing import Any, Dict, Generator, List


class GenericPreprocessor(ABC):
    """Generic interface for preprocessing data."""

    def preprocess(
        self, data: List[Dict[str, Any]]
    ) -> Generator[Dict[str, Any], None, None]:
        """Preprocess the documents.
        'norm_tokens' field must be present in output if using the embedding_driver.py to train a Word2Vec model.

        Args:
            data: list of dicts containing the data.

        Returns:
            generator of the same data dict with the added required fields
        """
        raise NotImplementedError

    def featurize(
        self, data: List[Dict[str, Any]]
    ) -> Generator[Dict[str, Any], None, None]:
        """Featurize the data. This function is directly called by clustering_driver.py.
        Importantly, each featurized data point must contain the following fields:
            - `id`: a unique identifier associated with each data point
            - `text`: the raw input text
            - `clustering_tokens`: the final tokens that will be input to clustering
            - `embedding`: the embedding associated with the data point.

        Args:
            data: list of dicts containing the data

        Returns:
            generator of the same data dict with the added required fields
        """
        raise NotImplementedError

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

import collections
import logging
from typing import Any, Dict, FrozenSet, Generator, List

import numpy as np

logging_format = (
    "%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s"
)
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger(__name__)


class ClusterHandler:
    def __init__(self, configuration: Dict) -> None:
        """Consume the configuration and initialize the clustering model.

        Args:
            configuration: Contains the hyperparameters to init the clustering model.

        Returns:
            (nothing)
        """
        cluster_algorithm = configuration["algorithm"]
        self.clustering_model = cluster_algorithm()
        self.clustering_model.consume_config(configuration)

    def prepare_for_clustering(
        self,
        featurized_data: Generator[Dict[str, Any], None, None],
        convergence_patience: int,
        convergence_improvement_threshold: float,
    ) -> None:
        """Converts the data into Documents, which are input to the downstream clustering.

        Args:
            featurized_data: Generator pointing to featurized data.
                             Datums must contain `id`, `text`, `clustering_tokens`, `embedding`
            convergence_patience: number of successive iterations with no improvement before stopping.
            convergence_improvement_threshold: minimum improvement for an interation to reset early stopping.

        Returns:
            (nothing)
        """

        logger.info("Converting data into Documents")
        self._add_documents(featurized_data)

        logger.info("Setting document weights")
        self.clustering_model.set_document_weights()

        logger.info("Setting Convergence (Clustering) limits")
        self.clustering_model.set_convergence_limits(
            convergence_patience, convergence_improvement_threshold
        )

    def _add_documents(
        self, featurized_data: Generator[Dict[str, Any], None, None]
    ) -> None:
        """Converts featurized data into Documents, which are the required input to the clustering algorithm

        Args:
        featurized_data: Generator pointing to featurized data.
                         Datums must contain `id`, `text`, `clustering_tokens`, `embedding`

        Returns:
            (nothing)
        """
        for i, datum in enumerate(featurized_data):
            if i % 1000 == 0:
                logger.info(f"Adding document {i}...")
            self.clustering_model.add_document(datum)

    def cluster(self, seed: int) -> Dict[str, Any]:
        """Perform the actual clustering.

        Args:
            seed: used to randomly shuffle the document order.

        Returns:
            stats: contains stats acquired throughout the clustering
        """
        logger.info("Clustering")
        stats = self.clustering_model.cluster(seed)
        logger.info("Clustered")
        return stats

    def clear_results(self) -> None:
        """Reset clusters and document assignments.

        Important to do after each individual clustering run as across multiple seed-jobs the
        cluster handler is reused, removing the need to re-preprocess all documents.
        """
        # re-initialize convergence
        self.clustering_model._previous_best_metric_value = None
        self.clustering_model._patience_counter = 0

        # clear stats
        self.clustering_model.stats = {}

        # clear results
        self.clustering_model.clusters = []
        for document in self.clustering_model.documents.values():
            document.clear_cluster_assignment()


class Document:
    """Core class used by downstream clustering that encapulates the data for a document.

    Attributes:
        tokens (List of str): Tokens within the document
        document_ids (List of str): List of document ids
        raw_texts (List of str): List of raw texts associated with each document id
        vector (np.ndarray): embedding of document in embedding space
        weight (int): weight of this document when computing averages over groups of documents
        cluster (Cluster object): cluster to which this document belongs
        cluster_id (str): 32-character string uniquely identifying this cluster
        dist_to_cluster_center (float): distance fom the document to its cluster in embedding space
    """

    def __init__(self, tokens: FrozenSet, vector: np.ndarray) -> None:
        """Initialize the document.

        Args:
            tokens: The tokens associated with the document.
            vector: The embedding associated with the document.
        """
        self.tokens = tokens
        self.vector = vector
        self.document_ids: List[str] = []
        self.raw_texts: List[str] = []
        self.weight = 0
        self.cluster = None
        self.cluster_id = None
        self.dist_to_cluster_center = -1

    def clear_cluster_assignment(self) -> None:
        """Reset the document. It is no longer associated with any cluster.
        See `ClusterHandler.clear_results()` for its use.
        """
        self.cluster = None
        self.cluster_id = None
        self.dist_to_cluster_center = -1


class Cluster:
    """Class encapulating data for a cluster of documents.

    Attributes:
        cluster_id (str): 32-character string uniquely identifying this cluster
        center (np.ndarray): cluster center in embedding space
        token_probability (float): the token probability associated with this cluster.
        documents (List of Document objects): Document objects which belong to this cluster
    """

    def __init__(self, cluster_id: str, documents: List[Document]) -> None:
        """Initialize the clustering.

        Args:
            cluster_id: 32-character string uniquely identifying this cluster
            documents: Document objects which belong to this cluster
        """
        self.cluster_id = cluster_id
        self.center = None
        self.token_probability = None
        self.documents = documents

    def calculate_center(self) -> None:
        """Calculate and set the center of the cluster from the documents in the cluster using a
        weighted average of the document vectors.
        """

        if len(self.documents) == 0:
            self.center = None
            self.token_probability = {}
        else:
            self.center = np.average(
                [doc.vector for doc in self.documents],
                axis=0,
                weights=[doc.weight for doc in self.documents],
            )
            token_ctr = collections.defaultdict(float)
            for document in self.documents:
                for token in document.tokens:
                    token_ctr[token] += document.weight
            num_documents = sum(document.weight for document in self.documents)
            self.token_probability = {
                k: v / num_documents for k, v in token_ctr.items()
            }

    def get_document_ids(self) -> List[str]:
        """Return document ids of all documents in cluster.

        Args:
            (nothing)

        Returns:
            document_ids: A list of document ids which are contained in this cluster
        """

        document_ids = [
            document_id
            for document in self.documents
            for document_id in document.document_ids
        ]
        return document_ids

    def size(self) -> int:
        """Get the number of documents in the cluster.

        Returns:
            n_documents (int): The number of documents in the cluster
        """
        n_documents = sum(len(document.document_ids) for document in self.documents)
        return n_documents


class ClusteringModel:
    """Class encapulating data for a clustering model.

    Attributes:
        documents (Map of `frozenset` of str to `Document`): documents to be clustered,
                                                             keys are the tokens in the document
        clusters (List of Cluster objects): clusters in this model
    """

    def __init__(self) -> None:
        """Initialize the clustering model."""
        self.documents: Dict[FrozenSet, Document] = {}
        self.clusters: List[Cluster] = []
        self.stats: Dict[str, Any] = {}

    def add_document(self, featurized_datum: Dict[str, Any]) -> None:
        """Convert featurized datum to Document (or add to existing if tokens / embedding match existing).

        Args:
            featurized_datum: A featurized datum from the dataset.
        """
        # frozenset so that tokens can become the key
        document_tokens = frozenset(featurized_datum["clustering_tokens"])

        # add featurized_datum to document
        if document_tokens not in self.documents:
            self.documents[document_tokens] = Document(
                document_tokens, featurized_datum["embedding"]
            )
        document = self.documents[document_tokens]
        document.document_ids.append(featurized_datum["id"])
        document.raw_texts.append(featurized_datum["text"])

    def set_document_weights(self) -> None:
        """Set the weight for each document which is the number of ids."""
        for document in self.documents.values():
            document.weight = len(document.document_ids)

    def set_convergence_limits(
        self, convergence_patience: int, convergence_improvement_threshold: float
    ) -> None:
        """Set the criteria for establishing convergence and stopping the clustering algorithm.

        Args:
            convergence_patience: number of successive iterations with no improvement before stopping.
            convergence_improvement_threshold: minimum improvement for an interation to reset early stopping.

        Returns:
            (nothing)
        """
        self._patience = convergence_patience
        self._convergence_improvement_threshold = convergence_improvement_threshold
        # initialize
        self._previous_best_metric_value = None
        self._patience_counter = 0

    def check_convergence(self, metric: float) -> bool:
        """Checks for convergence.

        Assumes that the metric is trying to be minimzed, and that all metric values are positive.
        This is an equivalent principle to "early stopping", i.e. stop the algorithm if it has not
        improved in self._patience iterations.

        Args:
            metric: metric trying to be minimized

        Returns:
            has_converged: whether the algorithm has converged or not
        """
        has_converged = False
        if self._previous_best_metric_value is None:
            self._previous_best_metric_value = metric  # type: ignore
        else:
            d_metric = self._previous_best_metric_value - metric  # delta_metric
            if d_metric > self._convergence_improvement_threshold:
                # current metric is less than previous
                self._previous_best_metric_value = metric
                self._patience_counter = 0
            elif self._patience_counter > self._patience:
                has_converged = True
        self._patience_counter += 1
        return has_converged

    def consume_config(self, config: Dict[str, Any]):
        """This needs to be implemented by derived classes"""
        pass

    def cluster(self, seed: int):
        """This needs to be implemented by derived classes"""
        pass

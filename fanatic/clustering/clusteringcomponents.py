import collections
import logging
import time
from typing import Any, Dict, FrozenSet, List, Optional

import numpy as np

logging_format = (
    "%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s"
)
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger(__name__)


class ClusterHandler:
    def __init__(self, configuration: Dict) -> None:
        cluster_algorithm = configuration["algorithm"]
        self.clustering_model = cluster_algorithm()
        self.clustering_model.consume_config(configuration)

    def prepare_for_clustering(
        self, featurized_data, convergence_patience, convergence_improvement_threshold
    ):

        logger.info("Converting data into Documents")
        self._add_documents(featurized_data)

        logger.info("Setting document weights")
        self.clustering_model.set_document_weights()

        logger.info("Setting Convergence (Clustering) limits")
        self.clustering_model.set_convergence_limits(
            convergence_patience, convergence_improvement_threshold
        )

    def _add_documents(self, featurized_data: List[Dict]) -> None:
        """Create the documents to be clustered from the featurized data

        Args:
            featurized_data: A list of featurized data
                             (must contain `id`, `text`, `clustering_tokens`, `embedding`)
        """
        for i, datum in enumerate(featurized_data):
            if i % 1000 == 0:
                logger.info(f"Adding document {i}...")
            self.clustering_model.add_document(datum)

    def cluster(self, seed: int) -> Dict:
        logger.info("Clustering")
        stats = self.clustering_model.cluster(seed)
        logger.info("Clustered")
        return stats

    def clear_results(self) -> None:
        """
        Reset clusters and document assignments. Important to do after each individual
        clustering run if you are re-using cluster handler for another clustering run
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
    """Class encapulating data for a document
    Attributes:
        tokens (:obj:`list` of :obj:`str`): Tokens within the document
        document_ids (:obj:`list` of :obj:`str`): List of document ids
        raw_texts (:obj:`list` of :obj:`str`): List of raw texts associated with each document id
        vector (:obj:`np.array`): embedding of document in embedding space
        weight (int): weight of this document when computing averages over groups of documents
        cluster (:obj:`Cluster`): cluster to which this document belongs
        dist_to_cluster_center (float): distance fom the document to its cluster in embedding space
    """

    def __init__(self, tokens: FrozenSet, vector: np.ndarray) -> None:
        self.tokens = tokens
        self.vector = vector
        self.document_ids: List[str] = []
        self.raw_texts: List[str] = []
        self.weight = 0
        self.cluster = None
        self.cluster_id = None
        self.dist_to_cluster_center = -1

    def clear_cluster_assignment(self) -> None:
        self.cluster = None
        self.cluster_id = None
        self.dist_to_cluster_center = -1


class Cluster:
    """Class encapulating data for a cluster of documents
    Attributes:
        cluster_id (str): 32-characher string uniquely identifying this cluster
        center (:obj:`np.array`): cluster cener in embedding space
        documents (:obj:`list` of :obj:`Document`): Document objects which belong to this cluster
    """

    def __init__(self, cluster_id: str, documents: List[Document]):
        self.cluster_id = cluster_id
        self.center = None
        self.token_probability = None
        self.documents = documents

    def calculate_center(self):
        """Calculate and set the center of the cluster
        Calculates and sets the center of the cluster from the documents in the cluster
        using a weighted average of the document vectors.
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
        """Return document ids of all documents in cluster
        Returns:
            :obj:`list` of str: A list of document ids which are contained in this cluster
        """

        document_ids = [
            document_id
            for document in self.documents
            for document_id in document.document_ids
        ]
        return document_ids

    def size(self) -> int:
        """
        Get the number of documents in the cluster
        Returns:
            n_documents (int): The number of documents in the cluster
        """
        n_documents = sum(len(document.document_ids) for document in self.documents)
        return n_documents


class ClusteringModel:
    """Class encapulating data for a clustering model
    Attributes:
        documents (:obj:`map` from :obj:`frozenset` of str to :obj:`Document`): documents to be clustered, keys are the tokens in the document
        clusters (:obj:`list` of :obj:`Cluster`): clusters in this model
    """

    def __init__(self) -> None:
        self.documents: Dict[FrozenSet, Document] = {}
        self.clusters: List[Cluster] = []
        self.stats: Dict[str, Any] = {}

    def add_document(self, datum: Dict) -> None:
        # frozenset so that tokens can become the key
        document_tokens = frozenset(datum["clustering_tokens"])

        # add datum to document
        if document_tokens not in self.documents:
            self.documents[document_tokens] = Document(
                document_tokens, datum["embedding"]
            )
        document = self.documents[document_tokens]
        document.document_ids.append(datum["id"])
        document.raw_texts.append(datum["text"])

    def set_document_weights(self) -> None:
        """Set the weight for each document which is the number of ids."""
        for document in self.documents.values():
            document.weight = len(document.document_ids)

    def set_convergence_limits(
        self, convergence_patience: int, convergence_improvement_threshold: float
    ) -> None:
        self._patience = convergence_patience
        self._convergence_improvement_threshold = convergence_improvement_threshold
        # initialize
        self._previous_best_metric_value = None
        self._patience_counter = 0

    def check_convergence(self, metric: float) -> bool:
        """
        Checks for convergence. Assumes that the metric is trying to be minimzed, and that
        all metric values are positive. This is an equivalent principle to "early stopping",
        i.e. stop the algorithm if it has not improved in self._patience iterations
        Args:
            metric (float): metric trying to be minimized
        Return:
            has_converged (bool): whether the algorithm has converged or not
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

    def consume_config(self):
        """This needs to be implemented by derived classes"""
        pass

    def cluster(self):
        """This needs to be implemented by derived classes"""
        pass

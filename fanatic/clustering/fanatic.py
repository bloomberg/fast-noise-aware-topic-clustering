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
import random
import time
import uuid
from itertools import combinations
from typing import Any, Dict, FrozenSet, List, Tuple

import numpy as np
import scipy.spatial

from fanatic.clustering.clusteringcomponents import Cluster, ClusteringModel

logging_format = (
    "%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s"
)
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger(__name__)


class FanaticClusterModel(ClusteringModel):
    def consume_config(self, config: Dict[str, Any]) -> None:
        """Consume the configuration with the hyperparameters.

        Args:
            config: configuration containing the hyperparameters.

        Returns:
            (nothing)
        """
        self.clustering_lambda = config["clustering_lambda"]
        self.token_probability_threshold = config["token_probability_threshold"]
        self.max_num_clusters = config["max_num_clusters"]
        self.distance_metric = config["distance_metric"]
        self.merge_close_clusters_max_iterations = config[
            "merge_close_clusters_max_iterations"
        ]
        self.merge_close_clusters_lambda_fraction = config[
            "merge_close_clusters_lambda_fraction"
        ]
        self.batch_size = config["batch_size"]
        self.min_cluster_size = config["min_cluster_size"]
        self.max_clustering_time = config["max_clustering_time"]

    def reassign_documents(
        self,
        clustering_lambda: float,
        token_probability_threshold: float,
        distance_metric: str,
    ) -> None:
        """Reassign documents of filtered clusters to the clusters that remain.

        Args:
            clustering_lambda: "lambda" parameter that determines cluster size.
            token_probability_threshold: The min token probability required to add a document to existing cluster.
            distance_metric: metric used for calculating distance.

        Returns:
            (nothing)
        """
        reassigned_doc_cnt = 0
        if self.clusters:
            cluster_vectors = np.vstack(cluster.center for cluster in self.clusters)
            for document in self.documents.values():
                if not document.cluster:
                    dists = scipy.spatial.distance.cdist(
                        cluster_vectors, [document.vector], metric=distance_metric
                    )
                    all_idx = np.argsort(dists.flatten())
                    try:
                        idx = next(
                            (
                                i
                                for i in all_idx
                                if dists[i, 0] < clustering_lambda
                                and sum(
                                    self.clusters[i].token_probability.get(t, 0)
                                    for t in document.tokens
                                )
                                / float(len(document.tokens))
                                >= token_probability_threshold
                            )
                        )
                        cluster = self.clusters[idx]
                        min_dist = dists[idx, 0]
                        document.cluster = cluster
                        document.cluster_id = cluster.cluster_id
                        document.dist_to_cluster_center = min_dist
                        cluster.documents.append(document)
                        reassigned_doc_cnt += 1
                    except StopIteration:
                        pass
        logger.info(f"{str(reassigned_doc_cnt)} documents reassigned")

    def filter_small_clusters(self, min_cluster_size: int) -> None:
        """Filter out clusters with size below `min_cluster_size`.

        Args:
            min_cluster_size: the minimum required cluster size.

        Returns:
            (nothing)
        """
        for cluster in self.clusters:
            if cluster.size() < min_cluster_size:
                for document in cluster.documents:
                    document.cluster = None
                    document.cluster_id = None
                cluster.documents.clear()
        self.clusters = [
            cluster for cluster in self.clusters if len(cluster.documents) > 0
        ]

    def detect_and_merge_clusters(
        self,
        cluster_vectors: np.ndarray,
        clustering_lambda: float,
        merge_close_clusters_max_iterations: int,
        merge_close_clusters_lambda_fraction: int,
        distance_metric: str,
    ) -> np.ndarray:
        """Merge clusters that are less than
                `clustering_lambda` * `merge_close_clusters_lambda_fraction`
        apart from each other.

        Per "merge iteration", each cluster can only be involved in a single merge. See paper for additional details.
        NOTE: code has only been tested with `merge_close_clusters_max_iterations`=1:

        Args:
            cluster_vectors: array of cluster centers
            clustering_lambda: "lambda" parameter that determines cluster size
            merge_close_clusters_max_iterations: max number of iterations to run the merge algorithm
            merge_close_clusters_lambda_fraction: Sets threshold for merging clusters
            distance_metric: metric used for calculating distance

        Returns:
            cluster_vectors: updated array of cluster centers (post merging)
            n_merged_total: count of number of clusters that were merged (for stats purposes)
        """
        merge_clusters_start_time = time.time()
        clusters_merged_in_last_iteration = True
        merge_iterations = 0
        n_clusters_merged_total = 0

        while (
            clusters_merged_in_last_iteration is True
            and merge_iterations < merge_close_clusters_max_iterations
        ):
            n_clusters_merged = 0
            merge_clusters_iteration_start_time = time.time()

            # setup variables
            clusters_merged_in_last_iteration = False
            merge_iterations += 1
            n_clusters = len(cluster_vectors)
            cluster_indices_altered = (
                set()
            )  # keeps track of cluster indices that have been involved in a merge
            cluster_ids_to_remove = (
                set()
            )  # keeps track of cluster ids that will be removed

            # calculate distances
            dists = scipy.spatial.distance.pdist(
                cluster_vectors, metric=distance_metric
            )
            dists_indices = list(combinations(range(n_clusters), 2))
            all_idx = np.argsort(dists)
            for idx in all_idx:
                if (
                    dists[idx]
                    < clustering_lambda * merge_close_clusters_lambda_fraction
                ):
                    index_i, index_j = dists_indices[idx]

                    # make sure cluster has not already been altered, only one merge allowed per while loop
                    if (
                        index_i not in cluster_indices_altered
                        and index_j not in cluster_indices_altered
                    ):
                        # merge clusters - weighted average based off number of inquiries in cluster in past iteration
                        len_inqs_i = self.clusters[index_i].size()
                        len_inqs_j = self.clusters[index_j].size()
                        weight_i = len_inqs_i / (len_inqs_i + len_inqs_j)
                        weight_j = len_inqs_j / (len_inqs_i + len_inqs_j)

                        # NOTE: now cluster_vectors[i] != clusters_i.center, but this is okay since there is only one
                        # merge allowed per cluster per while loop... then they are synced up again
                        self.clusters[index_i].center = (
                            weight_i * self.clusters[index_i].center
                        ) + (weight_j * self.clusters[index_j].center)

                        # keep track of indices
                        cluster_ids_to_remove.add(self.clusters[index_j].cluster_id)
                        cluster_indices_altered.add(index_i)
                        cluster_indices_altered.add(index_j)
                        clusters_merged_in_last_iteration = True
                        n_clusters_merged += 1
                else:
                    # since distances are sorted, if current dist >= clustering_lambda then the rest are
                    break

            n_clusters_merged_total += n_clusters_merged

            # remove clusters that were merged, remake cluster_vectors array
            self.clusters = [
                cluster
                for cluster in self.clusters
                if cluster.cluster_id not in cluster_ids_to_remove
            ]
            cluster_vectors = np.vstack(cluster.center for cluster in self.clusters)
            logger.info(
                f"Merged {n_clusters_merged} clusters in iteration {merge_iterations} took "
                f"{time.time() - merge_clusters_iteration_start_time} s"
            )
        logger.info(
            f"{n_clusters_merged_total} total merged clusters time from {merge_iterations} "
            f"iterations taking {time.time() - merge_clusters_start_time} s"
        )
        return cluster_vectors

    def filter_and_recalculate_cluster_centers(self) -> Tuple[float, np.ndarray]:
        """Filter out empty clusters and recalculate cluster centers.

        Args:
            (nothing)

        Returns:
            cluster_center_change: avg. change in cluster centers weighted by number of inquiries per cluster.
            cluster_vectors: array of cluster centers
        """
        # filter out empty clusters
        self.clusters = [
            cluster for cluster in self.clusters if len(cluster.documents) > 0
        ]

        # get cluster weights
        n_inquiries_per_cluster = []
        for cluster in self.clusters:
            n_inquiries_in_cluster = 0
            for document in cluster.documents:
                n_inquiries_in_cluster += len(document.document_ids)
            n_inquiries_per_cluster.append(n_inquiries_in_cluster)
        cluster_weights = np.asarray(n_inquiries_per_cluster) / np.sum(
            n_inquiries_per_cluster
        )

        # get old/new cluster centers
        old_cluster_vectors = np.vstack(cluster.center for cluster in self.clusters)
        for cluster in self.clusters:
            cluster.calculate_center()
        cluster_vectors = np.vstack(cluster.center for cluster in self.clusters)

        # calculate weighted cluster center change
        cluster_center_change = np.sum(
            [
                cluster_weights[i]
                * np.linalg.norm(cluster_vectors[i] - old_cluster_vectors[i])
                for i in range(cluster_vectors.shape[0])
            ]
        )
        return cluster_center_change, cluster_vectors

    def assign_documents_to_fixed_clusters(
        self,
        document_keys: List[FrozenSet],
        cluster_vectors: np.ndarray,
        clustering_lambda: float,
        token_probability_threshold: float,
        distance_metric: str,
        flag_update_documents: bool,
        batch_size: int = 150000,
    ) -> None:
        """
        Assigns (the remaining) documents to static clusters, i.e. max_num_clusters has been reached and
        no new clusters can be made. Optimized to be faster than the single-document-per-loop way.

        Args:
            document_keys: The (remaining) document keys that will be assigned to the fixed clusters
            cluster_vectors: array of cluster centers
            clustering_lambda: "lambda" parameter that determines cluster size.
            token_probability_threshold: The token probability required to add a document to existing cluster.
            distance_metric: metric used for calculating distance.
            flag_update_documents: if True, update document objects too (only set to True for final document assignment)
            batch_size: How many documents to cdist at a time (more docs = more memory needed)

        Returns:
            (nothing)
        """
        # init
        cluster_idx = np.arange(len(cluster_vectors))
        n_documents = len(document_keys)
        if batch_size is None:
            batch_size = n_documents

        # go through documents in batches
        for i in range(0, n_documents, batch_size):
            document_keys_batch = document_keys[i: i + batch_size]
            document_vectors_batch = [
                self.documents[document_key].vector
                for document_key in document_keys_batch
            ]

            # find distances of all documents to clusters in batch
            dists_batch = scipy.spatial.distance.cdist(
                document_vectors_batch, cluster_vectors, metric=distance_metric
            )
            filter_idx_batch = (
                dists_batch < clustering_lambda
            )  # boolean 2D array filtering out < clustering_lambda
            for j, document_key in enumerate(document_keys_batch):
                dists_below_lamda = dists_batch[j][
                    filter_idx_batch[j]
                ]  # keep only dists < lambda
                cluster_idx_below_lamda = cluster_idx[
                    filter_idx_batch[j]
                ]  # and get corresponding cluster indices
                sorted_dummy_idx = np.argsort(
                    dists_below_lamda.flatten()
                )  # sort indices by distance, yields "dummy" indices
                all_idx = cluster_idx_below_lamda[
                    sorted_dummy_idx
                ]  # map dummy to original cluster idx again
                document = self.documents[document_key]
                try:
                    idx = next(
                        (
                            k
                            for k in all_idx
                            if sum(
                                self.clusters[k].token_probability.get(t, 0)
                                for t in document.tokens
                            )
                            / float(len(document.tokens))
                            >= token_probability_threshold
                        )
                    )
                    cluster = self.clusters[idx]
                    cluster.documents.append(document)
                    if flag_update_documents is True:
                        min_dist = dists_batch[j, idx]
                        document.cluster = cluster
                        document.cluster_id = cluster.cluster_id
                        document.dist_to_cluster_center = min_dist
                except StopIteration:
                    if flag_update_documents is True:
                        document.cluster = None
                        document.cluster_id = None
                        document.dist_to_cluster_center = -1

    def initialize_clustering(self) -> Tuple[np.ndarray, List]:
        """
        Initialize the clustering algorithm.

        Args:
            (nothing)

        Returns:
            cluster_vectors: keeps the centroid of each cluster
            document_keys: the frozenset key associated with each document
        """

        # initialize all documents to belong to same cluster
        self.clusters = []
        cluster_id = uuid.uuid4().hex
        documents = list(self.documents.values())
        cluster = Cluster(cluster_id, documents)
        cluster.calculate_center()
        self.clusters.append(cluster)

        cluster_vectors = np.vstack(cluster.center for cluster in self.clusters)

        # randomly shuffle document order
        document_keys = list(self.documents.keys())
        random.shuffle(document_keys)
        return cluster_vectors, document_keys

    # cluster
    def cluster(self, seed: int) -> Dict[str, Any]:
        """MAIN driver that performs a FANATIC clustering against a set of input documents.
        See Silburt et al. (2021) published in EMNLP for more details.

        Args:
            seed: random seed to shuffle the document order.

        Returns:
            stats: stats object containing timing of the clustering.
        """

        # initialize
        start_time = time.time()
        logger.info(f"using random seed={seed}")
        random.seed(seed)
        cluster_vectors, document_keys = self.initialize_clustering()

        # MAIN LOOP: Perform FANATIC clustering until convergence or time limit reached
        while True:
            # check for time limit
            if (time.time() - start_time) > self.max_clustering_time:
                logger.info("Reached time limit! Terminating")
                break

            logger.info("Number of clusters: {}".format(len(self.clusters)))

            # clear out document lists in clusters
            for cluster in self.clusters:
                cluster.documents.clear()

            # document loop
            for doc_i, document_key in enumerate(document_keys):
                document = self.documents[document_key]

                # find closest cluster
                dists = scipy.spatial.distance.cdist(
                    cluster_vectors, [document.vector], metric=self.distance_metric
                ).flatten()
                (filter_idx,) = np.where(
                    dists < self.clustering_lambda
                )  # filter by lambda
                all_idx = filter_idx[np.argsort(dists[filter_idx])]
                try:
                    # filter by token probability
                    idx = next(
                        (
                            i
                            for i in all_idx
                            if sum(
                                self.clusters[i].token_probability.get(t, 0)
                                for t in document.tokens
                            )
                            / float(len(document.tokens))
                            >= self.token_probability_threshold
                        )
                    )
                    cluster = self.clusters[idx]
                    cluster.documents.append(document)
                except StopIteration:
                    # create new cluster containing this document if min distance exceeds clustering threshold
                    # or token probability was too low and there are less than max_num_clusters
                    if len(self.clusters) < self.max_num_clusters:
                        cluster_id = uuid.uuid4().hex
                        documents = [document]
                        cluster = Cluster(cluster_id, documents)
                        cluster.calculate_center()
                        self.clusters.append(cluster)
                        cluster_vectors = np.vstack((cluster_vectors, cluster.center))
                    else:
                        # max clusters reached, thus document is not added to any cluster -
                        # re-assign remaining documents in a vectorized way (faster), then exit
                        remaining_document_keys = document_keys[doc_i:]
                        self.assign_documents_to_fixed_clusters(
                            document_keys=remaining_document_keys,
                            cluster_vectors=cluster_vectors,
                            clustering_lambda=self.clustering_lambda,
                            token_probability_threshold=self.token_probability_threshold,
                            distance_metric=self.distance_metric,
                            flag_update_documents=False,
                            batch_size=self.batch_size,
                        )
                        break

            # filter out empty clusters, recalculate center for each cluster
            (
                cluster_center_change,
                cluster_vectors,
            ) = self.filter_and_recalculate_cluster_centers()
            logger.info(
                "Change in cluster centers (weighted): {}".format(cluster_center_change)
            )

            # check for convergence
            if self.check_convergence(cluster_center_change) is True:
                logger.info(
                    f"Clustering metric hasnt improved by at least {self._convergence_improvement_threshold} "
                    f"in {self._patience} iterations. Terminating."
                )
                break

            # randomize order of looking at documents
            random.shuffle(document_keys)

            # find cluster pairs with distances less than lambda and merge
            if self.merge_close_clusters_max_iterations > 0:
                cluster_vectors = self.detect_and_merge_clusters(
                    cluster_vectors,
                    self.clustering_lambda,
                    self.merge_close_clusters_max_iterations,
                    self.merge_close_clusters_lambda_fraction,
                    self.distance_metric,
                )

        # clustering is over, perform final assignment of documents to clusters
        for cluster in self.clusters:
            cluster.documents.clear()

        self.assign_documents_to_fixed_clusters(
            document_keys=document_keys,
            cluster_vectors=cluster_vectors,
            clustering_lambda=self.clustering_lambda,
            token_probability_threshold=self.token_probability_threshold,
            distance_metric=self.distance_metric,
            flag_update_documents=True,
            batch_size=self.batch_size,
        )

        # filter out clusters that are too small
        n_clusters_before_filter = len(self.clusters)
        logger.info(
            "Number of clusters (pre-filtering small clusters): {}".format(
                n_clusters_before_filter
            )
        )
        self.filter_small_clusters(self.min_cluster_size)
        logger.info(
            "Number of clusters (post-filtering small clusters): {}".format(
                len(self.clusters)
            )
        )
        n_clusters_diff_filtering = n_clusters_before_filter - len(self.clusters)
        logger.info("Number of clusters Filtered: {}".format(n_clusters_diff_filtering))

        # reassign documents that were filtered out if they can be potentially assigned an existing cluster
        logger.info("Reassigning documents")
        self.reassign_documents(
            self.clustering_lambda,
            self.token_probability_threshold,
            self.distance_metric,
        )

        # cluster stats object
        self.stats["cluster_time"] = time.time() - start_time
        logger.info(f"FANATIC clustering took {self.stats['cluster_time']} seconds")

        return self.stats

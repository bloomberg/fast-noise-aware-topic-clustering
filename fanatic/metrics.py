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
from collections import Counter
from typing import Any, Dict, FrozenSet, List, Optional

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

from fanatic.clustering.clusteringcomponents import Document
from fanatic.preprocess.labels import NOISE_LABEL

logging_format = (
    "%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s"
)
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger(__name__)


NO_ASSIGNMENT = (
    "NO_ASSIGNMENT"  # label to assign to documents that did not fall into a cluster
)


def calculate_metrics(
    clustered_documents: Dict[FrozenSet, Document],
    data_labels: Dict[str, str],
    cluster_stats: Dict[str, Any],
) -> Dict[str, float]:
    """Calculate final metrics and clustering stats.

    Args:
        clustered_documents: The clustered documents
        data_labels: document_id / label mapping for the clustered documents
        cluster_stats: contains some stats aggregated during clustering and further filled here.

    Returns:
        metrics: the final calculated metrics
    """
    # main arrays
    document_assignments = []
    document_labels = []

    # stats
    number_of_tp = 0  # number of coherent docs in clusters
    number_of_fp = 0  # number of noise docs in clusters
    number_of_fn = 0  # number of coherent docs that did not fall in a cluster
    number_of_tn = 0  # number of noise docs that did not fall in a cluster
    cluster_ids: List[Optional[str]] = []

    # main
    for document in clustered_documents.values():
        # get cluster assignment
        cluster_id = document.cluster_id
        assignment = cluster_id if cluster_id is not None else NO_ASSIGNMENT

        # assign individual doc_ids
        for doc_id in document.document_ids:
            label = data_labels[doc_id]

            document_assignments.append(assignment)
            document_labels.append(label)

            # STATS - TP/FP/TN/FN
            if assignment != NO_ASSIGNMENT:
                # documents assigned to a cluster
                cluster_ids.append(cluster_id)
                if label != NOISE_LABEL:
                    number_of_tp += 1
                else:
                    number_of_fp += 1
            else:
                # documents NOT assigned to a cluster
                if label != NOISE_LABEL:
                    number_of_fn += 1
                else:
                    number_of_tn += 1

    # report stats
    calculate_cluster_stats(
        cluster_stats,
        cluster_ids,
        number_of_tp,
        number_of_fp,
        number_of_tn,
        number_of_fn,
    )

    # calculate metrics (additional metrics can be added if desired)
    ami_score = adjusted_mutual_info_score(
        document_labels, document_assignments, average_method="arithmetic"
    )
    metrics = {"ami": ami_score}

    return metrics


def calculate_cluster_stats(
    cluster_stats: Dict[str, Any],
    cluster_ids: List[Optional[str]],
    number_of_tp: int,
    number_of_fp: int,
    number_of_tn: int,
    number_of_fn: int,
) -> None:
    """Calculate the final clustering stats.

    Args:
        cluster_stats: dict to be filled with the calculated stats
        cluster_ids: list of the unique cluster ids
        number_of_tp: number of "true positives"
        number_of_fp: number of "false positives"
        number_of_tn: number of "true negatives"
        number_of_fn: number of "false negatives"
    """
    logger.info("*** Performance Cluster Stats ***")

    # cluster-size distribution stats
    clusters = Counter(cluster_ids)
    cluster_counts = list(clusters.values())
    num_clusters = len(clusters)
    cluster_stats["total_number_of_clusters"] = num_clusters

    if num_clusters > 0:
        quartiles = {
            "cluster_size_quartile_min": min(cluster_counts),
            "cluster_size_quartile_Q1": np.percentile(
                cluster_counts, 25, interpolation="nearest"
            ),
            "cluster_size_quartile_median": np.percentile(
                cluster_counts, 50, interpolation="nearest"
            ),
            "cluster_size_quartile_Q3": np.percentile(
                cluster_counts, 75, interpolation="nearest"
            ),
            "cluster_size_quartile_max": max(cluster_counts),
        }

        # cluster size stats
        logger.info(
            f"Total number of clusters: {cluster_stats['total_number_of_clusters']}"
        )
        logger.info(
            f"Cluster-size Quartile statistics: min={quartiles['cluster_size_quartile_min']}, "
            f"Q1={quartiles['cluster_size_quartile_Q1']}, "
            f"median={quartiles['cluster_size_quartile_median']} "
            f"Q3={quartiles['cluster_size_quartile_Q3']}, max={quartiles['cluster_size_quartile_max']}"
        )

        # update cluster stats with additional relevant stats
        cluster_stats.update(quartiles)

    # document clustered stats
    num_documents = number_of_tp + number_of_fp + number_of_tn + number_of_fn

    # number of docs that had a coherent subreddit label (and should have ended up in a cluster)
    number_of_coherent_labels = number_of_tp + number_of_fn

    # number of documents that actually ended up in a cluster
    number_of_documents_in_clusters = number_of_tp + number_of_fp

    # calculate pseudo precision / recall
    if (
        (number_of_tp + number_of_fp) > 0
        and (number_of_tp + number_of_tn) > 0
        and number_of_tp > 0
    ):
        pseudo_precision = number_of_tp / (number_of_tp + number_of_fp)
        pseudo_recall = number_of_tp / (number_of_tp + number_of_fn)
        pseudo_f1 = (
            2 * pseudo_precision * pseudo_recall / (pseudo_precision + pseudo_recall)
        )
    else:
        pseudo_precision = 0
        pseudo_recall = 0
        pseudo_f1 = 0

    documents_stats = {
        "num_total_documents": num_documents,
        "num_coherent_labels": number_of_coherent_labels,
        "coherent_labels_fraction": number_of_coherent_labels / num_documents,
        "num_docs_clustered": number_of_documents_in_clusters,
        "docs_clustered_fraction": number_of_documents_in_clusters / num_documents,
        "num_tp": number_of_tp,
        "tp_fraction": number_of_tp / num_documents,
        "num_fp": number_of_fp,
        "fp_fraction": number_of_fp / num_documents,
        "num_tn": number_of_tn,
        "tn_fraction": number_of_tn / num_documents,
        "num_fn": number_of_fn,
        "fn_fraction": number_of_fn / num_documents,
        "pseudo_precision": pseudo_precision,
        "pseudo_recall": pseudo_recall,
        "pseudo_f1": pseudo_f1,
    }
    for key, value in documents_stats.items():
        logger.info(f"{key}: {value}")

    # update cluster stats with additional relevant stats
    cluster_stats.update(documents_stats)


def average_metrics_stats_from_seed_runs(
    aggregate_metrics: List[Dict[str, float]], aggregate_stats: List[Dict[str, Any]]
):
    """
    Obtain the mean and standard deviation metric values from a number of seed runs

    Args:
        aggregate_metrics (list of dicts): each item is the output from calculate_metrics()
        aggregate_stats (list of dicts): each item is the cluster_stats output from clustering
    Returns:
        averaged_metrics (dict): The averaged metrics
    """
    # average metrics
    averaged_metrics = {}
    keys = list(aggregate_metrics[0].keys())
    for key in keys:
        # iterate over each individual metric, e.g. ami, ari, etc.
        key_values = []
        for run in aggregate_metrics:
            # iterate over each seed run
            try:
                key_values.append(run[key])
            except Exception:
                logger.warning(f"Couldnt find {key} in {run}")

        averaged_metrics[key] = {"mean": np.mean(key_values), "std": np.std(key_values)}
    logger.info("****** AVERAGED Metrics ******")
    logger.info(averaged_metrics)

    # average stats
    averaged_stats = {}
    keys = list(aggregate_stats[0].keys())
    for key in keys:
        key_values = []
        for run in aggregate_stats:
            try:
                key_values.append(run[key])
            except Exception:
                logger.warning(f"Couldnt find {key} in {run}")

        averaged_stats[key] = {"mean": np.mean(key_values), "std": np.std(key_values)}
    logger.info("****** AVERAGED Cluster Stats ******")
    logger.info(averaged_stats)
    return averaged_metrics, averaged_stats

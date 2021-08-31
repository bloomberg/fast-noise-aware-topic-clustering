from sklearn.metrics import adjusted_mutual_info_score
from collections import Counter
import numpy as np
import uuid
import random
from fanatic.preprocess.labels import NOISE_LABEL

import logging
logging_format = '%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger(__name__)


NO_ASSIGNMENT = 'NO_ASSIGNMENT' # label to assign to documents that did not fall into a cluster


def calculate_metrics(doc_assignments, data_labels, cluster_stats):
    '''
    Convert clustering results (and labels) into two simple flat lists of clustering assignment / label, so they
    can be easily consumed by sklearn.metrics. Additionally this function calculates a number of stats to print and
    save to file.
    Args:
        doc_assignments (dict): This is the cluster_handler.clustering_model.documents object from clusteringalgos.py
        data_labels (dict): These are the subreddit labels. They are `derived` since (if the proper flags are set)
                               their label is determined from the annotations (either the subreddit name or
                               NOISE_LABEL). See `label_generation.get_derived_clustering_label()` for more info
        cluster_stats (dict): This is filled with stats during clustering.

    Returns:
        document_assignments (list): list of cluster assignments for each document
        document_labels (list): corresponding list of derived labels for each document
    '''
    # main arrays
    document_assignments = []
    document_labels = []

    # stats
    number_of_tp = 0    # number of valid docs in clusters
    number_of_fp = 0    # number of noise docs in clusters
    number_of_fn = 0    # number of valid docs that were unassigned (i.e. in the noise cluster)
    number_of_tn = 0    # number of noise docs that were unassigned (i.e. in the noise cluster)
    cluster_ids = []

    # main
    for doc_key, document in doc_assignments.items():
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
    report_stats(cluster_stats, cluster_ids, number_of_tp, number_of_fp, number_of_tn, number_of_fn)

    # calculate metrics (additional metrics can be added if desired)
    ami_score = adjusted_mutual_info_score(document_labels, document_assignments, average_method='arithmetic')
    metrics = {
        "ami": ami_score
    }

    return metrics


def report_stats(cluster_stats, cluster_ids, number_of_tp, number_of_fp, number_of_tn, number_of_fn):
    '''
    This prints the stats in the log file (for quickly scanning) and also adds fields to `cluster_stats`
    which is eventually written to a metrics file for downstream analysis
    '''
    logger.info("*** Performance Cluster Stats ***")

    # cluster-size distribution stats
    clusters = Counter(cluster_ids)
    cluster_counts = list(clusters.values())
    n_clusters = len(clusters)
    cluster_stats['total_number_of_clusters'] = n_clusters

    if n_clusters > 0:
        quartiles = {
            'cluster_size_quartile_min': min(cluster_counts),
            'cluster_size_quartile_Q1': np.percentile(cluster_counts, 25, interpolation='nearest'),
            'cluster_size_quartile_median': np.percentile(cluster_counts, 50, interpolation='nearest'),
            'cluster_size_quartile_Q3': np.percentile(cluster_counts, 75, interpolation='nearest'),
            'cluster_size_quartile_max': max(cluster_counts)
        }

        # cluster size stats
        logger.info(f"Total number of clusters: {cluster_stats['total_number_of_clusters']}")
        logger.info(f"Cluster-size Quartile statistics: min={quartiles['cluster_size_quartile_min']}, "
                    f"Q1={quartiles['cluster_size_quartile_Q1']}, "
                    f"median={quartiles['cluster_size_quartile_median']} "
                    f"Q3={quartiles['cluster_size_quartile_Q3']}, max={quartiles['cluster_size_quartile_max']}")

        # update cluster stats with additional relevant stats
        cluster_stats.update(quartiles)

    # document clustered stats
    n_documents = number_of_tp + number_of_fp + number_of_tn + number_of_fn

    # number of docs that had a valid subreddit label (and should have ended up in a cluster)
    number_of_valid_labels = number_of_tp + number_of_fn

    # number of documents that actually ended up in a cluster
    number_of_documents_in_clusters = number_of_tp + number_of_fp

    # "precision", "recall", and "f1" (these aren't the true metrics, e.g. for TP this doesnt account for whether a TP
    # ended up in the correct cluster, only measures that it did ended up in a cluster and was supposed to)
    if (number_of_tp + number_of_fp) > 0 and (number_of_tp + number_of_tn) > 0 and number_of_tp > 0:
        coarse_precision = number_of_tp / (number_of_tp + number_of_fp)
        coarse_recall = number_of_tp / (number_of_tp + number_of_fn)
        coarse_f1 = 2 * coarse_precision * coarse_recall / (coarse_precision + coarse_recall)
    else:
        coarse_precision = 0
        coarse_recall = 0
        coarse_f1 = 0

    documents_stats = {
        'n_total_documents': n_documents,
        'n_valid_labels': number_of_valid_labels,
        'valid_labels_fraction': number_of_valid_labels / n_documents,
        'n_docs_clustered': number_of_documents_in_clusters,
        'docs_clustered_fraction': number_of_documents_in_clusters / n_documents,
        'n_tp': number_of_tp,
        'tp_fraction': number_of_tp / n_documents,
        'n_fp': number_of_fp,
        'fp_fraction': number_of_fp / n_documents,
        'n_tn': number_of_tn,
        'tn_fraction': number_of_tn / n_documents,
        'n_fn': number_of_fn,
        'fn_fraction': number_of_fn / n_documents,
        'coarse_precision': coarse_precision,
        'coarse_recall': coarse_recall,
        'coarse_f1': coarse_f1
    }
    for key, value in documents_stats.items():
        logger.info(f"{key}: {value}")

    # update cluster stats with additional relevant stats
    cluster_stats.update(documents_stats)


def average_metrics_stats_from_seed_runs(aggregate_metrics, aggregate_stats):
    '''
    Obtain the mean and standard deviation metric values from a number of seed runs

    Args:
        aggregate_metrics (list of dicts): each item is the output from calculate_metrics()
        aggregate_stats (list of dicts): each item is the cluster_stats output from clustering
    Returns:
        averaged_metrics (dict): The averaged metrics
    '''
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
            except:
                logger.warning(f'Couldnt find {key} in {run}')
        
        averaged_metrics[key] = {
            'mean': np.mean(key_values),
            'std': np.std(key_values)
        }
    print(f"****** AVERAGED Metrics ******")
    print(averaged_metrics)

    # average stats
    averaged_stats = {}
    keys = list(aggregate_stats[0].keys())
    for key in keys:
        key_values = []
        for run in aggregate_stats:
            try:
                key_values.append(run[key])
            except:
                logger.warning(f'Couldnt find {key} in {run}')
        
        averaged_stats[key] = {
            'mean': np.mean(key_values),
            'std': np.std(key_values)
        }

    # print
    print("****** AVERAGED Cluster Stats ******")
    print(averaged_stats)
    return averaged_metrics, averaged_stats

import argparse
import configparser
import json
import logging
import os
import pickle
from typing import Any, Dict

from fanatic.clustering.clusteringcomponents import ClusteringModel

logging_format = "%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger(__name__)


def _write_cluster_samples(
    write_name: str,
    clustering_model: ClusteringModel,
    data_labels: Dict[str, str],
    max_write_per_cluster: int = 10,
) -> None:
    """Write a few inquiries from each cluster to file for qualitative assessment.

    Args:
        write_name: filename to write the metrics
        clustering_model: The clustering model object that performed the clustering
        data_labels: The mapping between datum id and label
        max_write_per_cluster: the number of sample documents to write for each cluster

    Returns:
        (nothing)
    """

    logger.info("Writing cluster samples")
    with open(write_name, "w") as output:
        for i, cluster in enumerate(clustering_model.clusters):
            # write
            output.write(f"*** Cluster {i} ***\n")
            for document in cluster.documents[:max_write_per_cluster]:
                tokens = " ".join(document.tokens)
                document_labels = []
                for document_id in document.document_ids:
                    doc_label = data_labels[document_id]
                    document_labels.append(doc_label)
                document_labels_str = ",".join(document_labels)
                output.write(f"{tokens} -> {document_labels_str}\n")
            output.write("\n")


def _write_metrics(
    write_name: str,
    metrics: Dict[str, float],
    configuration: Dict[str, Any],
    cluster_stats: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Write all cluster_stats, metrics and input arguments to file via configparser.

    Args:
        write_name: filename to write the metrics
        metrics: the calculated metrics (e.g. ami)
        configuration: The hyperparameters associated with the clustering
        args: The argparser containing the job's arguments
        cluster_stats: The statistics gathered throughout the clustering job

    Returns:
        (nothing)
    """
    config = configparser.ConfigParser()
    config["HYPERS"] = configuration
    config["GENERAL_ARGS"] = {arg: str(getattr(args, arg)) for arg in vars(args)}
    config["CLUSTER_STATS"] = cluster_stats
    config["METRICS"] = metrics
    with open(write_name, "w") as configfile:
        config.write(configfile)
    logger.info("Wrote metrics, configuration, stats")


def _write_labels_and_assignments(write_name: str, clustering_model: ClusteringModel, data_labels: Dict[str, str]):
    """Dump cluster assignments and labels so one can recalculate metrics without re-running the (expensive) clustering job.

    Args:
        write_name: the name of the output json file
        clustering_model: The clustering model object that performed the clustering
        data_labels: The mapping between datum id and label

    Returns:
        (nothing)
    """
    # key is document id, value is dict containing assignment and label
    labels_and_assignments_dict = {}
    for doc in clustering_model.documents.values():
        for document_id in doc.document_ids:
            labels_and_assignments_dict[str(document_id)] = {
                "assignment": doc.cluster_id if doc.cluster_id is None else str(doc.cluster_id),
                "label": str(data_labels[document_id]),
            }

    # write to json
    with open(write_name, "w") as json_file:
        json.dump(labels_and_assignments_dict, json_file)
    logger.info("wrote assignments and labels to file")


def _get_output_basename(output_dir, clustering_run_id):
    return os.path.join(output_dir, f"fanatic_{clustering_run_id}")


def save_results(
    args: argparse.Namespace,
    data_labels: Dict[str, str],
    metrics: Dict[str, float],
    configuration: Dict[str, Any],
    cluster_stats: Dict[str, Any],
    run_index: int,
    dataset_id: str,
    clustering_model: ClusteringModel,
) -> None:
    """Primary function that dumps the input arguments and results of each seed-job to files for downstream analysis.

    Args:
        args: The argparser containing the job's arguments
        data_labels: The mapping between datum id and label
        metrics: the calculated metrics (e.g. ami)
        configuration: The hyperparameters associated with the clustering
        cluster_stats: The statistics gathered throughout the clustering job
        run_index: the seed-job number
        dataset_id: unique identifier shared by all seed-jobs
        clustering_model: The clustering model object that performed the clustering

    Returns:
        (nothing)
    """
    # create base path
    basename = _get_output_basename(args.output_dir, dataset_id)
    basename += f"_{run_index}"

    # write all document labels and assignments to file
    labels_and_assignments_file = f"{basename}_labels_and_assignments.json"
    _write_labels_and_assignments(labels_and_assignments_file, clustering_model, data_labels)

    # write the clustering summary to file - input arguments, clustering stats, clustering metrics
    results_file = f"{basename}_summary.txt"
    _write_metrics(results_file, metrics, configuration, cluster_stats, args)

    # write a few examples from each cluster to file for qualitative inspection
    clusters_sample_file = f"{basename}_sample_clusters.txt"
    _write_cluster_samples(clusters_sample_file, clustering_model, data_labels)

    # dump full pickled clustering model to file - this is expensive to do!
    if args.flag_save_clusteringmodel is True:
        clusters_file = f"{basename}_clusters.pkl"
        labels_file = f"{basename}_labels.pkl"
        with open(clusters_file, "wb") as handle:
            pickle.dump(clustering_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(labels_file, "wb") as handle2:
            pickle.dump(data_labels, handle2, protocol=pickle.HIGHEST_PROTOCOL)


def save_averaged_results(
    averaged_metrics: Dict[str, Dict[str, float]],
    averaged_stats: Dict[str, Dict[str, float]],
    configuration: Dict[str, Any],
    args: argparse.Namespace,
    dataset_id: str,
) -> None:
    """Save the input arguments and averaged results across the seed-jobs to files.

    Args:
        averaged_metrics: contains the mean and std of each metric, averaged over the seed-jobs
        averaged_stats: contains the mean and std of each cluste stat, averaged over the seed-jobs
        configuration: The hyperparameters associated with the clustering
        args: The argparser containing the job's arguments
        dataset_id: unique identifier shared by all seed-

    Returns:
        (nothing)
    """
    results_file = _get_output_basename(args.output_dir, dataset_id)
    results_file += "_summary_averaged.txt"

    _write_metrics(results_file, averaged_metrics, configuration, averaged_stats, args)

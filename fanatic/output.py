import configparser
import json
import logging
import os
import pickle
import uuid

logging_format = (
    "%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s"
)
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger(__name__)

# fields to skip writing to results file
ARGS_WRITE_SKIP_FIELDS = ["stop_words"]


def write_cluster_samples(
    clustering_model, labels, write_path, max_write_per_cluster=10
):
    """
    Simple function that writes a few inquiries from each cluster to give a birds-eye view of the clusters
    """

    logger.info("Writing cluster samples")
    with open(write_path, "w") as output:
        for i, cluster in enumerate(clustering_model.clusters):
            # write
            output.write(f"*** Cluster {i} ***\n")
            for document in cluster.documents[:max_write_per_cluster]:
                tokens = " ".join(document.tokens)
                document_labels = []
                for document_id in document.document_ids:
                    doc_label = labels[document_id]
                    document_labels.append(doc_label)
                document_labels_str = ",".join(document_labels)
                output.write(f"{tokens} -> {document_labels_str}\n")
            output.write("\n")


def write_metrics(write_name, metrics, hypers, cluster_stats, args):
    """
    Write all the `cluster_stats`, `metrics` and hyperparameters to a file for downstream analysis.

    Args:
        write_name (string): filename to write the metrics
        metrics (dict of dicts): nested dict of dicts that contains metrics calculated under different conditions
                                (tn excluded, randomized, etc.). See calculate_metrics() in performance.py
        hypers (dict): the hyperparameters
        args (argparse object): the input arguments
        cluster_stats (dict): the cluster_stats
    """
    logger.info("Writing metrics, hypers, stats")

    config = configparser.ConfigParser()
    config["HYPERS"] = hypers
    config["GENERAL_ARGS"] = {arg: getattr(args, arg) for arg in vars(args)}
    config["CLUSTER_STATS"] = cluster_stats
    config["METRICS"] = metrics
    with open(write_name, "w") as configfile:
        config.write(configfile)


def write_labels_and_assignments(
    labels_and_assignments_file, clustering_model, data_labels
):
    """
    Write the cluster assignments and labels to json so that when can recalculate metrics whenever we
    need without re-running.
    Args:
        labels_and_assignments_file (str): the name of the json file
        clustering_model (cluster obj): clustering object that contains all the docs (and clusters)
        data_labels (dict of str): (key, value) -> (document_id, label), where label = <subreddit> or "NOISE_LABEL"
    """
    # key is document id, value is dict containing assignment and label
    labels_and_assignments_dict = {}
    for doc in clustering_model.documents.values():
        for document_id in doc.document_ids:
            labels_and_assignments_dict[str(document_id)] = {
                "assignment": doc.cluster_id
                if doc.cluster_id is None
                else str(doc.cluster_id),
                "label": str(data_labels[document_id]),
            }

    # write to json
    with open(labels_and_assignments_file, "w") as json_file:
        json.dump(labels_and_assignments_dict, json_file)
    logger.info("wrote assignments and labels to file")


def _get_output_basename(output_dir, clustering_run_id):
    return os.path.join(output_dir, f"fanatic_{clustering_run_id}")


def save_results(
    args,
    data_labels,
    metrics,
    configuration,
    cluster_stats,
    run_index,
    dataset_id,
    clustering_model,
):
    """
    Main function that, per individual clustering run:
        a) writes all the results to a file, and
        b) writes a few titles from each cluster to a file.
    Also uploads to hdfs if this is running on dsp
    """
    # create base path
    basename = _get_output_basename(args.output_dir, dataset_id)
    basename += f"_{run_index}"

    # write labels and assignments to file
    labels_and_assignments_file = f"{basename}_labels_and_assignments.json"
    write_labels_and_assignments(
        labels_and_assignments_file, clustering_model, data_labels
    )

    # write cluster stats
    results_file = f"{basename}_results.txt"
    write_metrics(results_file, metrics, configuration, cluster_stats, args)

    # write cluster samples
    clusters_sample_file = f"{basename}_sample_clusters.txt"
    write_cluster_samples(clustering_model, data_labels, clusters_sample_file)

    # write pickle files - these are expensive so only write if you really want them
    if args.flag_save_clusteringmodel is True:
        clusters_file = f"{basename}_clusters.pkl"
        labels_file = f"{basename}_labels.pkl"
        with open(clusters_file, "wb") as handle:
            pickle.dump(clustering_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(labels_file, "wb") as handle2:
            pickle.dump(data_labels, handle2, protocol=pickle.HIGHEST_PROTOCOL)


def save_averaged_results(
    averaged_metrics, averaged_stats, configuration, args, dataset_id
):
    """
    Saves all the averaged results (from the individual seeded clustering runs), writes to file with the same
    consistency as save_results()
    """
    results_file = _get_output_basename(
        args.output_dir, dataset_id
    )
    results_file += f"_results_averaged.txt"

    write_metrics(results_file, averaged_metrics, configuration, averaged_stats, args)

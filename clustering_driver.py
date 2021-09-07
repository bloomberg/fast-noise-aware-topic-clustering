import argparse
import logging
import uuid
from typing import Any, Dict, List, Tuple

import numpy as np

import fanatic.metrics
import fanatic.output
from fanatic.arguments import (CONVERGENCE_IMPROVEMENT_THRESHOLD,
                               CONVERGENCE_PATIENCE, build_algorithm_config,
                               parse_args)
from fanatic.clustering.clusteringcomponents import ClusterHandler
from fanatic.preprocess import (filter_data, labels, nltk_preprocessor,
                                read_data)

logging_format = "%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger(__name__)

LARGE_INT = 100000  # arbitrary large int for picking random seeds


def process_and_write_aggregate_results(
    aggregate_metrics: List[Dict],
    aggregate_stats: List[Dict],
    configuration: Dict,
    args: argparse.Namespace,
    dataset_id: str,
) -> None:
    """Average the stats and metrics from the individual seed-jobs.

    Args:
        aggregate_metrics: list of metrics dictionaries, one per seed-job
        aggregate_stats: list of stats dictionaries, one per seed-job
        configuration: the hyperparameters for the job
        args: contains the input arguments for the job
        dataset_id: unique identifier shared by all seed-jobs

    Returns:
        (nothing)
    """
    (
        averaged_metrics,
        averaged_stats,
    ) = fanatic.metrics.average_metrics_stats_from_seed_runs(aggregate_metrics, aggregate_stats)

    fanatic.output.save_averaged_results(averaged_metrics, averaged_stats, configuration, args, dataset_id)

    final_metric = averaged_metrics["ami"]["mean"]
    logger.info(f"For dataset_id={dataset_id} final averaged ami metric={final_metric}")


def run_clustering(
    args: argparse.Namespace,
    cluster_handler: ClusterHandler,
    data_labels: Dict[str, str],
    configuration: Dict[str, Any],
    seeds_for_job: List[int],
    dataset_id: str,
) -> Tuple[List[Dict], List[Dict]]:
    """Performs the clustering on the featurized data.

    Args:
        args: contains the input arguments for the job
        cluster_handler: manages the clustering
        data_labels: Contain the labels associated with the clustering
        configuration: contains the hyperparameters for the job
        seeds_for_job: list of integers for the individual seed jobs to be run
        dataset_id: unique identifier shared by all seed-jobs

    Returns:
        aggregate_metrics: list of metrics across the seed-jobs
        aggregate_stats: list of clustering stats across the seed-jobs
    """
    # init
    aggregate_metrics = []
    aggregate_stats = []

    # perform clustering for each seed
    for run_index, seed_for_job in enumerate(seeds_for_job):
        logger.info(f"Beginning clustering job {run_index + 1}/{args.num_clustering_seed_jobs}...")

        # cluster
        cluster_stats = cluster_handler.cluster(seed_for_job)

        # aggregate results into flat lists, obtain clustering stats
        metrics = fanatic.metrics.calculate_metrics(
            cluster_handler.clustering_model.documents, data_labels, cluster_stats
        )

        # save results
        logger.info("Saving Results")
        args.job_seed = seed_for_job
        fanatic.output.save_results(
            args,
            data_labels,
            metrics,
            configuration,
            cluster_stats,
            run_index,
            dataset_id,
            cluster_handler.clustering_model,
        )

        aggregate_metrics.append(metrics)
        aggregate_stats.append(cluster_stats)

        # important: clear clustering to re-use cluster_handler and keep all pre-processed data
        cluster_handler.clear_results()

    return aggregate_metrics, aggregate_stats


def get_data_and_labels(args: argparse.Namespace) -> Tuple[List[Dict], Dict]:
    """Loads the data and generates the labels associated with said data.

    Args:
        args: the argparser containing the arguments relevant to loading the data and labels

    Returns:
        data: The loaded data
        data_labels: The mapping between datum id and label
    """

    # load annotation labels
    subreddit_labels = None
    subreddit_labels_list = None
    subreddit_noise_percentage = None
    num_docs_read_from_data_files = args.num_docs_read
    if args.subreddit_labels_file is not None:
        # restrict universe of reddit data to the list of subreddits specified in args.subreddit_labels_file
        subreddit_labels = labels.load_subreddit_labels(args.subreddit_labels_file)
        subreddit_labels_list = list(subreddit_labels.keys())

        # this argument is only relevant if subreddit labels have been provided
        subreddit_noise_percentage = args.subreddit_noise_percentage

        # read all the data from provided data files, and then filter down later according to subreddit_noise_percentage
        num_docs_read_from_data_files = None
        logger.info(
            "Reading in all data from all provided input files. "
            f"Will filter later according to subreddit_noise_percentage={subreddit_noise_percentage}"
        )

    # load data
    logger.info("Loading inquiries")
    data = read_data.read_files(
        args.data_files,
        num_docs_read=num_docs_read_from_data_files,
        min_valid_tokens=args.min_valid_tokens,
        subreddit_labels_list=subreddit_labels_list,
    )

    # if there is a desired coherent/noise subreddit percentage specified, filter here
    if subreddit_labels is not None and subreddit_noise_percentage is not None and args.num_docs_read is not None:
        logger.info(f"Filtering data according to subreddit_noise_percentage={subreddit_noise_percentage}")
        data = filter_data.filter_data_by_noise_percentage(
            data,
            args.num_docs_read,
            subreddit_noise_percentage,
            subreddit_labels,
            args.seed_data_subsample,
        )

    # return final data and labels
    return labels.prepare_final_dataset_and_labels(data, subreddit_labels)


def main():
    # parse input arguments
    args = parse_args()
    logger.info(f"arguments: {args}")

    # load configuration and init cluster handler
    configuration = build_algorithm_config(args)
    cluster_handler = ClusterHandler(configuration)
    logger.info(f"Init cluster handler with algorithm configuration: {configuration}")

    # get data and labels
    data, data_labels = get_data_and_labels(args)
    logger.info("gathered data")

    # preprocess data
    engine = nltk_preprocessor.NLTKPreprocessor(
        embedding_model_file=args.embedding_model_file,
        min_valid_tokens=args.min_valid_tokens,
    )
    featurized_data_generator = engine.featurize(data)
    cluster_handler.prepare_for_clustering(
        featurized_data_generator,
        CONVERGENCE_PATIENCE,
        CONVERGENCE_IMPROVEMENT_THRESHOLD,
    )
    logger.info("preprocessed data")

    # cluster
    dataset_id = uuid.uuid4().hex
    np.random.seed(args.clustering_seed)
    seeds_for_job = list(np.random.randint(0, LARGE_INT, args.num_clustering_seed_jobs))  # set random seeds for job
    aggregate_metrics, aggregate_stats = run_clustering(
        args, cluster_handler, data_labels, configuration, seeds_for_job, dataset_id
    )
    logger.info("clustered data")

    # write aggregate results
    args.job_seeds = seeds_for_job  # add all seeds as list attr so will be written to aggregate results
    del args.job_seed  # this attribute was only relevant for individual jobs
    process_and_write_aggregate_results(aggregate_metrics, aggregate_stats, configuration, args, dataset_id)
    logger.info("Successful completion of clustering!")


if __name__ == "__main__":
    main()

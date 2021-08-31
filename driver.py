import argparse
from fanatic.preprocess import filter_data, labels, nltk_preprocessor, read_data
from fanatic.clustering import clusteringcomponents as cc
from fanatic.metrics import performance
from fanatic.clustering.config import ALGORITHM_CONFIG, CONVERGENCE_PATIENCE, CONVERGENCE_IMPROVEMENT_THRESHOLD
import json
import numpy as np
import os
from gensim.models import KeyedVectors

import logging
logging_format = '%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger(__name__)


def run_clustering(args, cluster_handler, data_labels, configuration, seed_for_job, run_index, dataset_id):
    '''
    Runs an individual clustering job, calculates metrics, saves results. 
    Args:
        args (argparse object): contains the input arguments for the job
        cluster_handler (clusteringmodel object): contains the inputs/results of clustering
        data_labels (dict): These are the subreddit labels. They are `derived` since (if the proper flags are set)
                               their label is determined from the annotations (either the subreddit name or
                               NOISE_LABEL). See `label_generation.get_derived_clustering_label()` for more info
        configuration (dict): contains the hyperparameters for the job
        run_index (int): integer of what job number this is
        dataset_id (str or None): name shared across all seed-jobs. If None it is filled in output.save_results()
    Returns:
        metrics (dict of dicts): contains all the metrics 
        cluster_stats (dict): contains all the clustering stats (elapsed time, number of clusters, etc.)
        dataset_id (str): the dataset id
    '''

    # cluster
    cluster_stats = cluster_handler.cluster(seed_for_job)

    # aggregate results into flat lists, obtain clustering stats
    assignments, labels, assignments_exclude_tn, labels_exclude_tn = \
        performance.aggregate_results(cluster_handler.clustering_model.documents, data_labels, cluster_stats,
                                      prepare_results_excluding_tn_flag=True)

    # calculate metrics
    metrics = performance.calculate_metrics(assignments, labels, assignments_exclude_tn, labels_exclude_tn)

    # save results
    #logger.info("Saving Results")
    # dataset_id = output.save_results(cluster_handler.clustering_model, data_labels, metrics, configuration,
    #                                  cluster_stats, args, args.save_clusteringmodel_to_hdfs,
    #                                  run_index, dataset_id)
    return metrics, cluster_stats, dataset_id


def get_data_and_labels(args):
    '''
    This is the general high-level function surrounding all things related to data and labels. Specifically it includes:
        - loading annotation labels to be used to validate the clustering results (if annotation_labels_file is provided)
        - Reading in the data
        - Achieving a specific ratio of "valid" (subreddits annotated as 'yes') vs. "noise" (subreddits annotated as "no")
          subreddits (given that `subreddit_noise_percentage` is not None and `n_read` is not None)
    '''

    # load annotation labels
    subreddit_labels = None
    subreddit_labels_list = None
    subreddit_noise_percentage = None
    n_read_from_data_files = args.n_read
    if args.subreddit_labels_file is not None:
        # restrict universe of reddit data to the list of subreddits specified in args.subreddit_labels_file
        subreddit_labels = labels.load_subreddit_labels(args.subreddit_labels_file)
        subreddit_labels_list = list(subreddit_labels.keys())
        
        # this argument is only relevant if subreddit labels have been provided
        subreddit_noise_percentage = args.subreddit_noise_percentage

        # read all the data from provided data files, and then filter down later according to subreddit_noise_percentage
        n_read_from_data_files = None


    # load data
    logger.info(f'Loading inquiries')
    data = read_data.read_files(args.data_files, 
                                #n_read=n_read_from_data_files,
                                n_read=4000,  # TODO: remove
                                min_sentence_length=args.min_sentence_length,
                                subreddit_labels_list=subreddit_labels_list)

    # if there is a desired valid/noise subreddit percentage specified, filter here
    if subreddit_noise_percentage is not None and args.n_read is not None:
        logger.info(f'Filtering data according to subreddit_noise_percentage={subreddit_noise_percentage}')
        data = filter_data.filter_data_by_noise_percentage(data, 
                                                           args.n_read, 
                                                           subreddit_noise_percentage,
                                                           subreddit_labels,
                                                           args.seed_data_subsample)

    # flatten dataset
    data, data_labels = labels.prepare_titles_and_labels(data, subreddit_labels)

    return data, data_labels


# input data type for argparse
def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x


def parse_args():
    parser = argparse.ArgumentParser(description='Cluster reddit titles')                   

    # data / label arguments
    parser.add_argument('--data-files', type=str,
                        nargs='+',
                        default=[ 
                            "data/RS_2011-01.zst"
                        ],
                        help='data files to load')
    parser.add_argument('--n-read', type=int,
                        default=4000,  # TODO: change to None as default
                        help='Number of documents to read in. Default is set to `None`, which reads everything')
    parser.add_argument('--subreddit-noise-percentage', type=restricted_float,
                        default=0.5,
                        help='controls the percentage of "no"/("yes" + "no") annotated subreddits. '
                             'Requires --subreddit-labels-file argument to be specified'
                             'Leave defaul="None" to disregard this feature (and not use any specific noise ratio)')
    parser.add_argument('--seed-data-subsample', type=int,
                        default=42,
                        help='this seed is used when subsampling `n_read` from valid/noise docs')
    parser.add_argument('--subreddit-labels-file', type=str,
                        default=f'subreddit_labels.json',
                        help='location of file containing annotation labels')

    # preprocessing arguments
    parser.add_argument('--embedding-model-file', type=str,
                        default='word2vec_w5_s300_RS_2017-10_2M.txt', 
                        help='embedding model file')
    parser.add_argument('--min-sentence-length', type=int,
                        default=3,
                        help='Minimum number of words in a sentence for the sentence to be used in clustering; \
                        the first sentence in an inquiry with at least min_sentence_length words will be used.')
    
    # algorithm arguments
    parser.add_argument('--cluster-algorithm',
                        type=str,
                        default='fanatic',
                        help='algorithm to run clustering against')
    parser.add_argument('--n-seed-jobs', type=int,
                        default=3,
                        help='Number of (different-seeded) clustering jobs to run')
    parser.add_argument('--clustering-seed', type=int,
                        default=42,
                        help='Used for generating individual clustering run seeds')   


    args = parser.parse_args()
    return args


def main():
    # parse input arguments
    args = parse_args()

    # get data and labels
    data, data_labels = get_data_and_labels(args)

    # load configuration
    configuration = ALGORITHM_CONFIG[args.cluster_algorithm]
    logger.info(f"Configuration Values: {configuration}")

    # init cluster handler
    cluster_handler = cc.ClusterHandler(configuration)

    # preprocess data
    embedding_model = KeyedVectors.load_word2vec_format(args.embedding_model_file, binary=False)
    engine = nltk_preprocessor.NLTKPreprocessor()
    featurized_data_generator = engine.featurize(data)
    cluster_handler.preprocess(featurized_data_generator, embedding_model, args.stop_words, args.min_sentence_length,
                               CONVERGENCE_PATIENCE, CONVERGENCE_IMPROVEMENT_THRESHOLD)

    # cluster - run `args.n_seed` clustering jobs with different seeds
    aggregate_metrics = []
    aggregate_stats = []
    dataset_id = None   # dataset_id set once in run_clustering and re-used
    np.random.seed(args.clustering_seed)
    seeds_for_job = list(np.random.randint(0, 100000, args.n_seed_jobs)) # set random seeds for job
    for run_index, seed_for_job in enumerate(seeds_for_job):
        logger.info(f"Beginning clustering job {run_index + 1}/{args.n_seed_jobs}")

        # cluster and aggregate results
        setattr(args, 'job_seed', seed_for_job)     # add attr so it will be written with clustering results to file
        metrics, cluster_stats, dataset_id = run_clustering(args, cluster_handler, data_labels, configuration, 
                                                            seed_for_job, run_index, dataset_id)
        aggregate_metrics.append(metrics)
        aggregate_stats.append(cluster_stats)

        # important: delete clustering results from current run to re-use cluster_handler and keep all pre-processed data
        cluster_handler.clear_results()


if __name__ == '__main__':
    main()
import argparse
from typing import Any, Dict, Optional

from fanatic.clustering.fanatic import FanaticClusterModel

CONVERGENCE_PATIENCE = 1  # sets the minimum required improvement to the metric (or else increment patience),
CONVERGENCE_IMPROVEMENT_THRESHOLD = 0.02  # sets the max number of *consecutive* "no metric improvement" iterations before stopping


def _restricted_float(input_value: str) -> float:
    try:
        x = float(input_value)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (input_value,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x


def _optional_restricted_float(x: str) -> Optional[float]:
    if x == 'None':
        return None
    else:
        return _restricted_float(x)


def _optional_string(x: str) -> Optional[str]:
    if x == 'None':
        return None
    else:
        return str(x)


def _optional_int(x: str) -> Optional[int]:
    if x == 'None':
        return None
    else:
        return int(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Cluster reddit titles')                   

    # data / label arguments
    parser.add_argument('--data-files', type=str,
                        nargs='+',
                        default=[ 
                            "../data/RS_2017-11.zst",
                        ],
                        help='data files to load')
    parser.add_argument('--n-read', type=_optional_int,
                        default=50000,
                        help='Number of documents to read in. If `None` read in everything')
    parser.add_argument('--subreddit-labels-file', type=_optional_string,
                        default=f'../data/subreddit_labels.json',
                        help='location of file containing annotation labels. Set to `None` to ignore annotation labels and use the subreddit as the label')
    parser.add_argument('--subreddit-noise-percentage', type=_optional_restricted_float,
                        default=0.2,
                        help='controls the percentage of "noise"/("coherent" + "noise") documents in the data. '
                             'Requires --subreddit-labels-file argument to be specified. '
                             'If set to `None`, the noise percentage is set by the natural data distribution.')
    parser.add_argument('--seed-data-subsample', type=int,
                        default=42,
                        help='this seed is used to shuffle the order of the documents in the dataset.')

    # preprocessing arguments
    parser.add_argument('--embedding-model-file', type=str,
                        default='../data/w2v_reddit_s300_w5_sg1_RS_2017-11.txt', 
                        help='embedding model file')
    parser.add_argument('--min-valid-tokens', type=int,
                        default=3,
                        help='Minimum number of words in a sentence for the sentence to be used in clustering; \
                        the first sentence in an inquiry with at least min_sentence_length words will be used.')
    
    # fanatic algorithm arguments
    parser.add_argument('--clustering-threshold', type=float,
                        default=0.45,
                        help='Clustering lambda threshold')
    parser.add_argument('--min-term-probability', type=_restricted_float,
                        default=0.01,
                        help='Minimum average term probability required across a cluster')
    parser.add_argument('--max-num-clusters', type=int,
                        default=600,
                        help='Maximum number of clusters allowed to be created')
    parser.add_argument('--min-cluster-size',
                        type=int,
                        default=50,
                        help='Minimum cluster size')
    parser.add_argument('--distance-metric', choices=['euclidean', 'cosine'],
                        default='cosine',
                        help='Distance metric used to calculate vector distances')
    parser.add_argument('--merge-close-clusters-max-iterations', type=int,
                        default=1,
                        help='Max number of cluster merge iterations per DPMeans cluster loops.')
    parser.add_argument('--merge-close-clusters-lambda-fraction', type=_restricted_float,
                        default=0.8,
                        help='Fraction of lambda for two clusters to be merged (value between 0 and 1).')
    parser.add_argument('--max-clustering-time', type=int,
                        default=7200,
                        help='Maximum amount of time (in seconds) to spend clustering')
    parser.add_argument('--batch-size', type=int,
                        default=150000)
    parser.add_argument('--n-seed-jobs', type=int,
                        default=1,
                        help='Number of (different-seeded) clustering jobs to run. An averaged result is generated')
    parser.add_argument('--clustering-seed', type=int,
                        default=42,
                        help='Used for generating individual clustering run seeds')   

    # output arguments
    parser.add_argument('--output-dir',
                        type=str,
                        default='../output',
                        help='directory for where to dump results')
    parser.add_argument('--flag-save-clusteringmodel', action='store_true',
                        default=False,
                        help='If true, save pickled ClusteringModel results to hdfs (warning: It is a large file)')

    args = parser.parse_args()
    return args


def build_algorithm_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = {
        "clustering_threshold": args.clustering_threshold,
        "min_term_probability": args.min_term_probability,
        "max_num_clusters": args.max_num_clusters,
        "distance_metric": args.distance_metric,
        "min_cluster_size": args.min_cluster_size,
        "merge_close_clusters_max_iterations": args.merge_close_clusters_max_iterations,
        "merge_close_clusters_lambda_fraction": args.merge_close_clusters_lambda_fraction,
        "max_clustering_time": args.max_clustering_time,
        "batch_size": args.batch_size,
        "algorithm": FanaticClusterModel,
    }
    return config
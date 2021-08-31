from fanatic.clustering.fanatic import FanaticClusterModel

# CONVERGENCE_IMPROVEMENT_THRESHOLD sets the minimum required improvement to the metric (or else increment patience), 
# CONVERGENCE_PATIENCE sets the max number of *consecutive* "no metric improvement" iterations before stopping
CONVERGENCE_PATIENCE = 1
CONVERGENCE_IMPROVEMENT_THRESHOLD = 0.02

# extendible to other algorithms
ALGORITHM_CONFIG = {
    'fanatic': {
        'clustering_threshold': 1.3,
        'min_term_probability': 0,
        'max_num_clusters': 100,
        'distance_metric': 'euclidean',
        'min_cluster_size': 10,
        'merge_close_clusters_max_iterations': 0,
        'merge_close_clusters_lambda_fraction': 0.3,
        'max_clustering_time': 7200,
        'batch_size': 150000,
        'algorithm': FanaticClusterModel,
    },
}
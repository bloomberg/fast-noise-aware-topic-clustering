import numpy as np
import collections
import time

import logging
logging_format = '%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger(__name__)


def get_filtered_tokens(title, embedding_model, filter_words):
    filtered_title_tokens = [token for token in title['norm_tokens']
                             if token in embedding_model
                             and token not in filter_words]
    return filtered_title_tokens


def get_averaged_embedding(filtered_title_tokens, embedding_model):
    vecs = [embedding_model[token] for token in filtered_title_tokens]
    averaged_vector = np.average(vecs, axis=0)
    return averaged_vector


class ClusterHandler:
    
    def __init__(self, configuration):
        cluster_algorithm = configuration['algorithm']
        self.clustering_model = cluster_algorithm()
        self.clustering_model.consume_config(configuration)
   
    def add_titles(self, featurized_titles, min_required_title_tokens=3):
        ''' Add each title to clustering model
        
        Args: 
            featurized_titles: A list of featurized titles
                             (must contain id, text, norm_tokens, lemmas, and pos_tags)
            min_sentence_length: The minimum length of a sentence for it to be considered in clustering
        '''
        n_invalid_titles = 0
        for i, title in enumerate(featurized_titles):
            if i % 1000 == 0:
                logger.info('Adding sentence {}...'.format(i))

            if title.get('valid'):
                self.clustering_model.add_title(title, min_required_title_tokens)
            else:
                n_invalid_titles += 1
        logger.info(f"Filtered out {n_invalid_titles} invalid titles")


    def preprocess(self, featurized_titles, embedding_model, stop_words, min_sentence_length, 
                   convergence_patience, convergence_improvement_threshold):
        logger.info('Initializing model')
        self.clustering_model.set_embedding_model(embedding_model)
        self.clustering_model.set_filter_words(stop_words)

        logger.info('Adding titles')
        self.add_titles(featurized_titles, min_sentence_length)
        logger.info('Added titles')

        logger.info('Setting Convergence (Clustering) limits')
        self.clustering_model.set_convergence_limits(convergence_patience, convergence_improvement_threshold)


    def cluster(self, seed):
        logger.info('Clustering')
        stats = self.clustering_model.cluster(seed)
        logger.info('Clustered')
        return stats


    def clear_results(self):
        '''
        Reset clusters and document assignments. Important to do after each individual 
        clustering run if you are re-using cluster handler for another clustering run
        '''
        # re-initialize convergence
        self.clustering_model._previous_best_metric_value = None
        self.clustering_model._patience_counter = 0

        # clear stats
        self.clustering_model.stats = {}
        
        # clear results
        self.clustering_model.clusters = []
        for key, document in self.clustering_model.documents.items():
            document.clear_cluster_assignment()


class Document:
    '''Class encapulating data for a document
    Attributes:
        tokens (:obj:`list` of :obj:`str`): Tokens within the document
        document_ids (:obj:`list` of :obj:`str`): List of document ids
        raw_texts (:obj:`list` of :obj:`str`): List of raw texts associated with each document id
        vector (:obj:`np.array`): embedding of document in embedding space
        weight (int): weight of this document when computing averages over groups of documents
        cluster (:obj:`Cluster`): cluster to which this document belongs
        dist_to_cluster_center (float): distance fom the document to its cluster in embedding space
    '''

    def __init__(self, tokens):
        self.tokens = tokens
        self.document_ids = []
        self.raw_texts = []
        self.vector = []
        self.weight = 0
        self.cluster = None
        self.cluster_id = None
        self.dist_to_cluster_center = -1

    def clear_cluster_assignment(self):
        self.cluster = None
        self.cluster_id = None
        self.dist_to_cluster_center = -1


class Cluster:
    '''Class encapulating data for a cluster of documents
    Attributes:
        cluster_id (str): 32-characher string uniquely identifying this cluster
        center (:obj:`np.array`): cluster cener in embedding space
        documents (:obj:`list` of :obj:`Document`): Document objects which belong to this cluster
    '''

    def __init__(self, cluster_id, documents):
        self.cluster_id = cluster_id
        self.center = None
        self.token_probability = None
        self.documents = documents

    def calculate_center(self):
        '''Calculate and set the center of the cluster
        Calculates and sets the center of the cluster from the documents in the cluster
        using a weighted average of the document vectors.
        '''

        if len(self.documents) == 0:
            self.center = None
            self.token_probability = {}
        else:
            self.center = np.average([doc.vector for doc in self.documents],
                                     axis=0,
                                     weights=[doc.weight for doc in self.documents])
            token_ctr = collections.defaultdict(float)
            for document in self.documents:
                for token in document.tokens:
                    token_ctr[token] += document.weight
            num_documents = sum(document.weight for document in self.documents)
            self.token_probability = {k: v/num_documents for k, v in token_ctr.items()}

    def get_document_ids(self):
        '''Return document ids of all documents in cluster
        Returns:
            :obj:`list` of str: A list of document ids which are contained in this cluster
        '''

        document_ids = [document_id
                        for document in self.documents
                        for document_id in document.document_ids]
        return document_ids


    def size(self):
        '''
        Get the number of documents in the cluster
        Returns:
            n_documents (int): The number of documents in the cluster
        '''
        n_documents = sum(len(document.document_ids) for document in self.documents)
        return n_documents


class ClusteringModel:
    '''Class encapulating data for a clustering model
    Attributes:
        documents (:obj:`map` from :obj:`frozenset` of str to :obj:`Document`): documents to be clustered, keys are the tokens in the document
        clusters (:obj:`list` of :obj:`Cluster`): clusters in this model
    '''

    def __init__(self):
        self.documents = {}
        self._filter_words = []
        self.clusters = []

    def set_embedding_model(self, embedding_model):
        '''Set the word2model to be used for clustering
        Args:
            embedding_model (:obj:`map` from str to :obj:`np.array`): map from tokens to word2vc vectors
        '''

        self._embedding_model = embedding_model

    def set_filter_words(self, filter_words):
        '''Set words to be ignored when clustering
        Set the lemmas whose corresponding tokens should be removed/ignored when clustering.
        Args:
            filter_words (:obj:`list` of str): list of lemmas to ignore
        '''
        self._filter_words = set(filter_words)

    def add_title(self, title, min_required_title_tokens):

        filtered_title_tokens = get_filtered_tokens(title, self._embedding_model, self._filter_words)
        # frozenset so that tokens can become the key
        filtered_title_tokens = frozenset(filtered_title_tokens)

        # add title to document
        if len(filtered_title_tokens) >= min_required_title_tokens:
            if filtered_title_tokens not in self.documents:
                self.documents[filtered_title_tokens] = Document(filtered_title_tokens)
            document = self.documents[filtered_title_tokens]
            document.document_ids.append(title['id'])
            document.raw_texts.append(title['text'])

    def _get_vectors(self):
        '''Calculate and set the vector for each document
        Calculate and set the vector for each document.
        The vector is the average of the embedding embeddings of all valid tokens within the document.
        '''
        for document in self.documents.values():
            # get average embedding of tokens in document
            document.vector = get_averaged_embedding(document.tokens, self._embedding_model)
            document.weight = len(document.document_ids)

    def set_convergence_limits(self, convergence_patience, convergence_improvement_threshold):
        self._patience = convergence_patience
        self._convergence_improvement_threshold = convergence_improvement_threshold
        # initialize
        self._previous_best_metric_value = None
        self._patience_counter = 0

    def check_convergence(self, metric):
        '''
        Checks for convergence. Assumes that the metric is trying to be minimzed, and that
        all metric values are positive. This is an equivalent principle to "early stopping",
        i.e. stop the algorithm if it has not improved in self._patience iterations
        Args:
            metric (float): metric trying to be minimized
        Return:
            has_converged (bool): whether the algorithm has converged or not
        '''
        has_converged = False
        if self._previous_best_metric_value is None:
            self._previous_best_metric_value = metric
        else:
            d_metric = self._previous_best_metric_value - metric    #delta_metric
            if d_metric > self._convergence_improvement_threshold:
                # current metric is less than previous
                self._previous_best_metric_value = metric
                self._patience_counter = 0
            elif self._patience_counter > self._patience:
                has_converged = True
        self._patience_counter += 1
        return has_converged

    def consume_config(self):
        '''This needs to be implemented by derived classes
        '''
        pass

    def cluster(self):
        '''This needs to be implemented by derived classes
        '''
        pass


class ClusteringStats:
    def __init__(self):
        self.n_max_cluster_iterations = 0
        self.cluster_center_change = []
        self.n_docs_unassigned_to_cluster = []
        self.n_docs_assigned_to_cluster = []
        self.convergence_reached = False
        self.avg_distance_final_doc_reassignment = []
        self.n_docs_assigned_to_cluster_counter = 0
        self.n_clusters_merged_per_cluster_iteration = []
        self.document_assignment_times = []
        self.time_doc_assignment = 0

    def beginning_of_document_assignment(self):
        self.n_docs_assigned_to_cluster_counter = 0
        self.time_doc_assignment = time.time()

    def assigned_document_counter(self):
        self.n_docs_assigned_to_cluster_counter += 1

    def check_max_clusters_reached(self, max_num_clusters, logger):
        self.n_max_cluster_iterations += 1
        logger.info(f"Max number of clusters={max_num_clusters} reached this iteration")

    def end_of_document_assignment(self, logger, n_documents):
        document_assignment_time = time.time() - self.time_doc_assignment
        logger.info(f"document assignment time = {document_assignment_time}")
        self.document_assignment_times.append(document_assignment_time)
        self.n_docs_unassigned_to_cluster.append(n_documents - self.n_docs_assigned_to_cluster_counter)
        self.n_docs_assigned_to_cluster.append(self.n_docs_assigned_to_cluster_counter)

    def log_merge_count(self, n_clusters_merged):
        self.n_clusters_merged_per_cluster_iteration.append(n_clusters_merged)

    def record_cluster_center_change(self, cluster_center_change):
        self.cluster_center_change.append(cluster_center_change)


    def end_of_assignment(self, n_clusters_diff_filtering, reassigned_doc_cnt, number_of_final_clusters, n_documents):
        self.n_docs_unassigned_to_cluster.append(n_documents - self.n_docs_assigned_to_cluster_counter)
        self.n_docs_assigned_to_cluster.append(self.n_docs_assigned_to_cluster_counter)
        self.n_clusters_diff_filtering = n_clusters_diff_filtering
        self.reassigned_doc_cnt = reassigned_doc_cnt
        self.cluster_stats = {
            'number_of_cluster_iterations': len(self.cluster_center_change),
            'number_of_iterations_with_max_clusters_exceeded': self.n_max_cluster_iterations,
            'n_docs_unassigned_per_cluster_iteration': self.n_docs_unassigned_to_cluster,
            'n_docs_assigned_per_cluster_iteration': self.n_docs_assigned_to_cluster,
            'n_clusters_embed_merged_per_iteration': self.n_clusters_merged_per_cluster_iteration,
            'cluster_center_change': self.cluster_center_change,
            'number_of_final_clusters': number_of_final_clusters,
            'convergence_reached': self.convergence_reached,
            'number_of_clusters_filtered': n_clusters_diff_filtering,
            'document_assignment_iteration_times': self.document_assignment_times
        }
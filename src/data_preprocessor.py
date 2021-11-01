"""
preprocess data: build a graph from feature matrix for a dataset 
"""

# load packages 
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors

# ===========================
# ------ build graph --------
# ===========================

def nn_graph_builder(features, n=10):
    """
    build a graph from feature matrix using KNN 
    :param features: the feature matrix 
    :param n: the number of neighbors for each observation (excluding self loops)
    :return the trained adjacency matrix 
    """
    nbrs = NearestNeighbors(n_neighbors=n + 1, n_jobs=-1).fit(features)
    adjacency_matrix = nbrs.kneighbors_graph(features).toarray() 
    adjacency_matrix = adjacency_matrix - np.eye(features.shape[0])  # take away self loops
    return adjacency_matrix

def spectral_graph_builder(features, n_clusters=7, cutoff=0.95):
    """
    build a graph from spectral clustering. 
    Compared to KNN, this provides a prior knowledge of how many clusters we should expect. 
    This may be more aligned to some tasks, such as cora classification.
    :param features: the feature matrix 
    :param n_clusters: the number of clusters 
    :param cutoff: the lower threshold in percentile for considering an edge (between 0 and 1)
    :return the trained adjacency matrix 
    """
    spectral_cluster = SpectralClustering(n_clusters=n_clusters, n_jobs=-1).fit(features)
    affinity_matrix = spectral_cluster.affinity_matrix_
    affinity_matrix_off_diag = affinity_matrix - np.eye(features.shape[0])
    # cutoff 
    thr = np.quantile(affinity_matrix_off_diag.flatten(), cutoff)
    adjacency_matrix = (affinity_matrix_off_diag > thr).astype(int)
    return adjacency_matrix

# ============================
# ------ other methods -------
# ============================
def symmetrize_adjacency_matrix(adjacency_matrix):
    """ symmetrize matrix """
    adjacency_matrix_sym = adjacency_matrix + adjacency_matrix.T
    adjacency_matrix_sym[adjacency_matrix_sym > 1] = 1
    return adjacency_matrix_sym

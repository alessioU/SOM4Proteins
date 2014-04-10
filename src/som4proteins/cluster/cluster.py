import numpy as np

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster

class Cluster():
    '''Creates clusters and related stuff.
        
    :param matrix weight_mat: Each row is a unit, each column is a variable.
    '''
    def __init__(self, weight_mat):
        self.weight_mat = weight_mat
        
    def cluster_moj(self, method='complete'):
        '''Creates clusters according to the Mojena rule.

        :param string method:
            Performs hierarchical/agglomerative clustering using the
            specified method to calculate the distances('single', 'complete',
            'average', 'weighted', 'centroid', 'meidan', 'ward').
            See the Scipy documentation for the 'linkage' function.
        
        :returns: Array that assigns each unit to a cluster.
        :rtype: array of int
        '''
        linkage_matrix = linkage(pdist(self.weight_mat), method=method)
        max_num_clusters = self._calc_mojena_index(linkage_matrix)
        cl_class = fcluster(linkage_matrix, t=max_num_clusters, criterion='maxclust')
        return cl_class
    
    def calc_mojena_index(self, linkage_matrix):
        '''Calculate the number of cluster according to the Mojena rule.
        
        TODO: add k parameter, now is fixed to 2.75.
        
        * small k -> more clusters
        * big k -> less clusters
        
        
        :param linkage_matrix linkage_matrix:
                At the i-th iteration, clusters with indices Z[i, 0] and Z[i, 1]
                are combined to form cluster n + i. A cluster with an index less than n corresponds
                to one of the n original observations. The distance between clusters Z[i, 0] and Z[i, 1]
                is given by Z[i, 2]. For more information see the documentation for the linkage function
                in Scipy.
        
        :returns: Optimal number of cluster according to the Mojena rule.
        :rtype: int
                
        '''
        index = linkage_matrix[:, 2]
        m_index = np.mean(index)
        std_index = np.std(index, ddof=1)
        threshold = m_index + 2.75 * std_index
        return np.sum(index > threshold) + 1
    
    def calc_best(self, cl_class, cl_centr):
        '''For each cluster, find the closest unit to its centroid.
        
        :param array_of_int cl_class: Array that assigns for each unit a cluster.
        :param matrix_of_float cl_centr: Centroids of the clusters, dimension = num clusters x unit dimension.
            
        :returns: Array that assigns for each cluster the closest unit to its centroid.
        :rtype: array of int
        ''' 
        num_units, _ = self.weight_mat.shape
        num_clusters = len(np.unique(cl_class))
        cl_best = np.zeros(num_clusters)
        for c in range(num_clusters):        
            best_dist = -1
            for n_unit in range(num_units):
                if cl_class[n_unit] == c + 1:
                    dist_centr_unit = pdist(np.array([cl_centr[c], self.weight_mat[n_unit]]))
                    if best_dist < 0 or dist_centr_unit < best_dist:
                        best_dist = dist_centr_unit
                        cl_best[c] = n_unit
        return cl_best
    
    def calc_centroids(self, cl_class):
        '''For each cluster, find the centroid.
        
        :param array_of_int cl_class: Array that assigns for each unit a cluster
        
        :returns: TODO: add return description
        :rtype: array of int
        '''
        num_clusters = len(np.unique(cl_class))
        dim = self.weight_mat.shape[1]
        cl_centr = np.zeros((num_clusters, dim))    
        for c in range(num_clusters):
            if np.sum(cl_class == (c+1)) != 0:
                cl_centr[c] = np.sum(self.weight_mat[cl_class == (c+1), :], axis=0)
                cl_centr[c] = cl_centr[c] / np.sum(cl_class == (c+1))
            else:
                cl_centr[c] = np.zeros(dim)
        return cl_centr
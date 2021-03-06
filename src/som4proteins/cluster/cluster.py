import numpy as np

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster

class Cluster():
    '''Creates clusters given a weight matrix.
        
    :param matrix weight_mat: Each row is a neuron, each column is a variable.
    :param array_of_int hits: The number of hits in each neuron.
    
    :ivar int num_clusters: Number of clusters created.
    :ivar array_of_int _cl_class: It assigns each neuron to a cluster.
    :ivar matrix_of_float _cl_centr: Centroids of the clusters, dimension = num clusters x unit dimension.
    :ivar matrix weight_mat_no_empty_neurons: Weight matrix where empty neurons are removed.
    :cvar NO_CLUSTER: Constant assigned to an empty neuron that is not in any cluster. 
    '''
    
    NO_CLUSTER = None
    
    def __init__(self, weight_mat, hits=None):
        self.set_cl_class(np.array([]))
        self._cl_centr = []
        
        self.weight_mat = weight_mat
        self.hits = hits
        self.weight_mat_no_empty_neurons = self.weight_mat
        if hits != None:
            self.weight_mat_no_empty_neurons = self.weight_mat[hits > 0]
    
    def _add_empty_neurons(self):
        '''Add to self._cl_class the empty neurons.'''
        if self.hits == None:
            return self._cl_class
        for i, hit in enumerate(self.hits):
            if hit <= 0:
                self._cl_class = np.insert(self._cl_class, i, self.NO_CLUSTER)
    
    def cluster_moj(self, method='complete'):
        '''Creates clusters according to the Mojena rule.
        
        Empty neurons are assigned to the NO_CLUSTER cluster.

        :param string method:
            Performs hierarchical/agglomerative clustering using the
            specified method to calculate the distances('single', 'complete',
            'average', 'weighted', 'centroid', 'meidan', 'ward').
            See the Scipy documentation for the 'linkage' function.
        
        :returns: Array that assigns each unit to a cluster.
        :rtype: array of int
        '''
        linkage_matrix = linkage(pdist(self.weight_mat_no_empty_neurons),
                                    method=method)
        max_num_clusters = self._calc_mojena_index(linkage_matrix)
        self.set_cl_class(fcluster(linkage_matrix, t=max_num_clusters, criterion='maxclust'))
        self._linkage_matrix = linkage_matrix
        self._add_empty_neurons()
        return self._cl_class
    
    def _calc_mojena_index(self, linkage_matrix):
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
        self._threshold = m_index + 2.75 * std_index
        return np.sum(index > self._threshold) + 1
    
    def save_centroids_data(self, dataframe, bmus, cl_best, filename):
        '''Save the centroids' data to a file.
        
        On every line there's the number of the cluster, followed by the
        number of the neuron centroid and the label of the data that is
        closest to that neuron
        '''
        with open(filename, mode='w') as f:
            # TODO: make it more efficient
            for i, best_neuron in enumerate(cl_best):
                idx = np.arange(len(dataframe.data))[bmus == best_neuron]
                data = dataframe.data[idx]
                unit = np.tile(self.weight_mat[best_neuron], (len(data), 1))
                dist = np.sqrt(np.sum(np.square(data - unit), axis=1))
                sorted_idx = idx[np.argsort(dist)]
                labels = dataframe.row_labels[sorted_idx]
                f.write(str(i+1) + '\t' + str(best_neuron + 1)
                        + '\t' + labels[0] + '\n')
    
    def save_cluster_neuron_data(self, cl_best, filename):
        '''Save the clusters' data to a file.
        
        On every line there's the number of the cluster, followed by the
        a list of the neurons associated with that cluster sorted by
        distance with respect to the centroid
        '''
        with open(filename, mode='w') as f:
            # TODO: make it more efficient
            for i, best_neuron in enumerate(cl_best):
                idx = self._cl_class == i + 1
                data = self.weight_mat[idx]
                unit = np.tile(self.weight_mat[best_neuron], (len(data), 1))
                dist = np.sqrt(np.sum(np.square(data - unit), axis=1))
                idx = np.arange(len(idx))[idx]
                sorted_idx, _ = zip(*sorted(zip(idx, dist),
                                            key=lambda el: el[1]))
                f.write(str(i+1) + '\t' +
                        ', '.join(map(lambda n: str(n+1),
                                      sorted_idx)) + '\n')
    
    def calc_best(self):
        '''For each cluster, find the closest not empty neuron to its centroid.
        
        Before calling this function calc_centroids() has to be called.
        
        :returns: Array that assigns for each cluster the closest not empty neuron to its centroid.
        :rtype: array of int
        '''
        cl_best = np.zeros(self._num_clusters, dtype=int)
        for c, centroid in enumerate(self._cl_centr):
            best_dist = -1
            for n_unit, (neuron, cluster) in enumerate(zip(self.weight_mat, self._cl_class)):
                if cluster == c + 1:
                    dist_centr_unit = pdist(np.array([centroid,
                                                      neuron]))
                    if best_dist < 0 or dist_centr_unit < best_dist:
                        best_dist = dist_centr_unit
                        cl_best[c] = n_unit
        return cl_best
    
    def calc_centroids(self):
        '''For each cluster, find the centroid.
        
        :returns: Centroids of the clusters, dimension = num clusters x unit dimension.
        :rtype: array of int
        '''
        cl_class = self._cl_class
        dim = self.weight_mat.shape[1]
        cl_centr = np.zeros((self._num_clusters, dim))    
        for c in range(self._num_clusters):
            if np.sum(cl_class == (c+1)) != 0:
                cl_centr[c] = np.sum(self.weight_mat[cl_class == (c+1), :], axis=0)
                cl_centr[c] = cl_centr[c] / np.sum(cl_class == (c+1))
            else:
                cl_centr[c] = np.zeros(dim)
        self._cl_centr = cl_centr
        return self._cl_centr
    
    def set_cl_class(self, cl_class):
        # self._cl_class can contain None objects
        self._cl_class = cl_class.astype('O')
        self._num_clusters = len(np.unique([c for c in self._cl_class if c != self.NO_CLUSTER]))
        
    def get_linkage_matrix(self):
        return self._linkage_matrix
    
    def get_threshold(self):
        return self._threshold
    
    @property
    def num_clusters(self):
        return self._num_clusters
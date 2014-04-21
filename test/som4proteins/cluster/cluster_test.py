import unittest
import os

import numpy as np

from som4proteins.cluster.cluster import Cluster
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

class ClusterTest(unittest.TestCase):
    test_files_dir = os.path.join(os.environ["PROJECT_ABS_DIR"], "test", "testfiles")
    cl_class = np.array([2,2,2,2,2,1,1,1,1,1,2,2,2,2,2,1,1,1,1,1,2,2,2,2,2,1,1,1,1,1,2,2,2,2,1,1,1,1,1,1,2,2,2,2,1,1,1,1,1,1,2,2,2,2,1,1,1,1,1,1,2,2,2,4,4,4,4,4,4,4,2,2,2,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,4,4,4,4,4,4,4,4,3,3])
    
    def _load_file(self, filename):
        fname = os.path.join(self.test_files_dir, filename)
        return np.loadtxt(open(fname, "rb"), delimiter=",")
    
    def test_mojena(self):
        weight_mat = self._load_file("test_integration_som_protein_codebook.csv")
        c = Cluster(weight_mat)
        np.testing.assert_equal(c.cluster_moj(), self.cl_class)
        
    def test_best_units(self):
        weight_mat = self._load_file("test_integration_som_protein_codebook.csv")
        c = Cluster(weight_mat)
        c.set_cl_class(self.cl_class)
        c.calc_centroids()
        np.testing.assert_equal(c.calc_best(), [37, 22, 98, 75])
        
    def test_centroids(self):
        weight_mat = self._load_file("test_integration_som_protein_codebook.csv")
        actual_centr = self._load_file("test_mojena_centroids.csv")
        c = Cluster(weight_mat)
        c.set_cl_class(self.cl_class)
        np.testing.assert_almost_equal(c.calc_centroids(), actual_centr, decimal=5)
        
    def test_calc_mojena_index(self):
        weight_mat = self._load_file("test_integration_som_protein_codebook.csv")
        c = Cluster(weight_mat)
        linkage_matrix = linkage(pdist(weight_mat), method='complete')
        self.assertEqual(c._calc_mojena_index(linkage_matrix), 4)
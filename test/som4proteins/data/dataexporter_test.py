import unittest
import os

import numpy as np

from som4proteins.data.dataexporter import ProteinDataExporter
from som4proteins.data.dataframe import DataFrame
from som4proteins.data.dataimporter import ProteinDataImporter
from som4proteins.som.maps.enums import Lattice, Shape
from som4proteins.som.maps.map import Map
from som4proteins.som.parameters.parameters import TrainingParameters
from som4proteins.som.parameters.enums import PHASE
from som4proteins.som.trainalgorithm import TrainAlgorithm

class FirstTest(unittest.TestCase):
    test_files_dir = os.path.join(os.environ["PROJECT_ABS_DIR"], "test", "testfiles")
    output_dir = os.path.join(os.environ["PROJECT_ABS_DIR"], "test", "output")
    
    def _load_file(self, filename):
        fname = os.path.join(self.test_files_dir, filename)
        return np.loadtxt(open(fname, "rb"), delimiter=",")
    
    def _remove_all_files(self):
        fileList = os.listdir(self.output_dir)
        for fileName in fileList:
            os.remove(os.path.join(self.output_dir, fileName))
            
    def setUp(self):
        self._remove_all_files()
    
    def tearDown(self):
        self._remove_all_files()
    
    def test_save_neurons(self):
        # ahatv
        trajectory_file = os.path.join(self.test_files_dir, "trajcomb_ca_dt100_AHATV.xtc")
        structure_file = os.path.join(self.test_files_dir, "ahatv_CA_aligned.pdb")
        protein_name = "ahatv"
        p_ahatv = ProteinDataImporter(trajectory_file=trajectory_file,
                                    structure_file=structure_file,
                                    protein_name=protein_name,
                                    output_dir=self.output_dir)
        dataframe = p_ahatv.get_coord_matrix()
        # ahavf
        trajectory_file = os.path.join(self.test_files_dir, "trajcomb_ca_dt100_AHAVF.xtc")
        structure_file = os.path.join(self.test_files_dir, "ahavf_CA_aligned.pdb")
        protein_name = "ahavf"
        p_ahavf = ProteinDataImporter(trajectory_file=trajectory_file,
                                    structure_file=structure_file,
                                    protein_name=protein_name,
                                    output_dir=self.output_dir)
        dataframe.join(p_ahavf.get_coord_matrix())
        dim = dataframe.n_cols
        msize = [10, 10]
        som_map = Map(map_size=msize, weight_vector_dim=dim, lattice=Lattice.Hex, shape=Shape.Sheet)
        som_map.lininit(dataframe.data)
        trainingParameters = TrainingParameters()
        trainingParameters.defaultParameters(PHASE.Rough)
        bt = TrainAlgorithm(som_map, dataframe.data, trainingParameters)
        bt.runBatch()
        bmus,_ = som_map.calc_bmus(dataframe.data)
        pde = ProteinDataExporter(dataframe, som_map.num_units, bmus, self.output_dir)
        pde.save_neurons()
        # TODO check that file exists
import unittest
import os

import numpy as np

from som4proteins.graphics.grid import Grid
from som4proteins.som.maps.enums import Lattice

class GridTest(unittest.TestCase):
    test_files_dir = os.path.join(os.environ["PROJECT_ABS_DIR"], "test", "testfiles")
    output_dir = os.path.join(os.environ["PROJECT_ABS_DIR"], "test", "output")
    cl_class = np.array([2,2,2,2,2,1,1,1,1,1,2,2,2,2,2,1,1,1,1,1,2,2,2,2,2,1,1,1,1,1,2,2,2,2,1,1,1,1,1,1,2,2,2,2,1,1,1,1,1,1,2,2,2,2,1,1,1,1,1,1,2,2,2,4,4,4,4,4,4,4,2,2,2,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,4,4,4,4,4,4,4,4,3,3])
    
    def test_add_hits_hex(self):
        fname = os.path.join(self.test_files_dir, "test_integration_som_protein_hits.csv")
        hits = np.loadtxt(open(fname, "rb"), delimiter=",")
        g = Grid([10,10], Lattice.Hex)
        g.add_hits(hits)
        #g.show()
        g.close()
    
    def test_add_hits_rect(self):
        fname = os.path.join(self.test_files_dir, "test_integration_som_protein_hits.csv")
        hits = np.loadtxt(open(fname, "rb"), delimiter=",")
        g = Grid([10,10], Lattice.Rect)
        g.add_hits(hits)
        #g.show()
        g.close()
        
    def test_add_hits_add_numbers_hex(self):
        fname = os.path.join(self.test_files_dir, "test_integration_som_protein_hits.csv")
        hits = np.loadtxt(open(fname, "rb"), delimiter=",")
        g = Grid([10,10], Lattice.Hex)
        g.add_edges()
        g.add_hits(hits)
        g.add_numbers(hits)
        #g.show()
        g.close()
        
    def test_add_hits_add_numbers_rect(self):
        fname = os.path.join(self.test_files_dir, "test_integration_som_protein_hits.csv")
        hits = np.loadtxt(open(fname, "rb"), delimiter=",")
        g = Grid([10,10], Lattice.Rect)
        g.add_edges()
        g.add_hits(hits)
        g.add_numbers(hits)
        #g.show()
        g.close()
        
    def test_color_grid_hex(self):
        g = Grid([10,10], Lattice.Hex)
        palette = [ 'green',
                'orange',                
                'violet',
                'gray',
                'yellow',
                'brown',
                'cyan',
                'magenta' ]
        g.add_clusters(self.cl_class, 4, palette)
        #g.show()
        g.close()
    
    def test_color_grid_rect(self):
        g = Grid([10,10], Lattice.Rect)
        palette = [ 'green',
                'orange',                
                'violet',
                'gray',
                'yellow',
                'brown',
                'cyan',
                'magenta' ]
        g.add_clusters(self.cl_class, 4, palette)
        #g.show()
        g.close()
        
    def test_color_grid_add_hits_add_best_save(self):
        fname = os.path.join(self.test_files_dir, "test_integration_som_protein_hits.csv")
        hits = np.loadtxt(open(fname, "rb"), delimiter=",")
        g = Grid([10,10], Lattice.Hex)
        palette = [ 'green',
                'orange',                
                'violet',
                'gray',
                'yellow',
                'brown',
                'cyan',
                'magenta' ]
        g.add_clusters(self.cl_class, 4, palette)
        g.add_hits(hits)
        g.add_bestunits_edges(self.cl_class, [37, 22, 98, 75])
        # cl_hits_grid.png should be similar to cl_hits_grid.png in the testfiles directory
        #g.save(self.output_dir, 'cl_hits_grid')
        #g.show()
        g.close()
        
    def test_color_grid_add_hits_add_best_rect(self):
        fname = os.path.join(self.test_files_dir, "test_integration_som_protein_hits.csv")
        hits = np.loadtxt(open(fname, "rb"), delimiter=",")
        g = Grid([10,10], Lattice.Rect)
        palette = [ 'green',
                'orange',                
                'violet',
                'gray',
                'yellow',
                'brown',
                'cyan',
                'magenta' ]
        g.add_clusters(self.cl_class, 4, palette)
        g.add_hits(hits)
        g.add_bestunits_edges(self.cl_class, [37, 22, 98, 75])
        #g.show()
        g.close()
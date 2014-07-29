from matplotlib.collections import PatchCollection
from matplotlib.colors import colorConverter
from matplotlib.patches import RegularPolygon, Rectangle

import matplotlib.pyplot as plt
import numpy as np
from som4proteins.cluster.cluster import Cluster
from som4proteins.graphics.graph import Graph
from som4proteins.som.maps.enums import Lattice


class Grid(Graph):
    def __init__(self, msize, lattice):
        _, self.ax = plt.subplots()
        self.msize = msize
        self.lattice = lattice
        self.coords = self._calculate_coordinates(self.msize, self.lattice)
    
    def _calculate_coordinates(self, size, lattice):
        num_dimensions = 2
        num_units = np.prod(size)
        coordinates = np.zeros([num_units, num_dimensions])
        
        indices = np.arange(num_units)
        coordinates[:, 0] = np.floor(indices / size[0])
        coordinates[:, 1] = indices % size[0]
        
        if lattice == Lattice.Hex:
            inds_for_row = (np.cumsum(np.ones(size[1], dtype=int)) - 1) * size[0]
            for i in range(1, size[0], 2):
                coordinates[inds_for_row + i, 0] += 0.5
            coordinates[:, 1] *= np.sqrt(0.75)
        coordinates[:, 0]*=5.2
        coordinates[:, 1]*=5.2
        return coordinates     
    
    def add_hits(self, hits):
        mypatches=[]
        max_hits = np.max(hits)
        for h, (x,y) in zip(hits, self.coords):
            if self.lattice == Lattice.Hex:
                radius = 0
                if h != 0:
                    # max radius is 3, minimum is .4
                    radius = .4 + 2.6 * h/max_hits
                patch = RegularPolygon((x,-y), numVertices=6, radius=radius,
                                             linewidth=0., facecolor=colorConverter.to_rgba('black'))
            else:
                sidelen = 0
                if h != 0:
                    sidelen = .7 + 4.5 * h/max_hits
                x += (5.2 - sidelen) / 2
                y -= (5.2 - sidelen) / 2
                patch = Rectangle((x, -y), sidelen, sidelen, linewidth=0., facecolor=colorConverter.to_rgba('black'))
            mypatches.append(patch)
            
        p = PatchCollection(mypatches, match_original=True)
        self.ax.add_collection(p)
        self.ax.autoscale_view()
        plt.axes().set_aspect('equal', 'datalim')
        
    def add_numbers(self, hits):
        for i,(x,y) in enumerate(self.coords):
            if hits[i] != 0:
                if self.lattice == Lattice.Hex:
                    x = x+1
                    y = -y
                else:
                    x += 5.2 / 2 + 1
                    y = -y + 5.2 / 2 - 1
                plt.text(x, y, int(hits[i]), color='#999999',
                         size=18, horizontalalignment='center', verticalalignment='center',
                         weight='bold')
    
    def add_edges(self):
        mypatches=[]
        for x,y in self.coords:
            if self.lattice == Lattice.Hex:
                patch = RegularPolygon((x,-y), numVertices=6,
                                             radius=3, facecolor='none', linewidth=0.5, alpha=0.3)
            else:
                patch = Rectangle((x, -y), 5.2, 5.2, facecolor='none', linewidth=0.5, alpha=0.3)
            mypatches.append(patch)
  
        p = PatchCollection(mypatches, match_original=True)
        self.ax.add_collection(p)
        self.ax.autoscale_view()
        plt.axes().set_aspect('equal', 'datalim')
        
    def add_bestunits_edges(self, cl_class, cl_best):
        mypatches=[]
        for bunit in cl_best:
            x, y = self.coords[bunit]
            if self.lattice == Lattice.Hex:
                patch = RegularPolygon((x,-y), numVertices=6,
                                             radius=3, facecolor='none',
                                             linewidth=1, edgecolor='red')
            else:
                patch = Rectangle((x, -y), 5.2, 5.2, facecolor='none',
                                        linewidth=1, edgecolor='red')
            
            mypatches.append(patch)
  
        p = PatchCollection(mypatches, match_original=True)
        self.ax.add_collection(p)
        self.ax.autoscale_view()
        plt.axes().set_aspect('equal', 'datalim')
        
    def add_clusters(self, cl_class, num_clusters, cl_color):
        mypatches=[]
        for i,(x,y) in enumerate(self.coords):
            if cl_class[i] == Cluster.NO_CLUSTER:
                continue
            
            if self.lattice == Lattice.Hex:
                patch = RegularPolygon((x,-y), numVertices=6, radius=3,
                                             facecolor=cl_color[cl_class[i] - 1],
                                             edgecolor='none')
            else:
                patch = Rectangle((x, -y), 5.2, 5.2,
                                        facecolor=cl_color[cl_class[i] - 1],
                                        edgecolor='none')
            mypatches.append(patch)
         
        p = PatchCollection(mypatches, match_original=True)

        self.ax.add_collection(p)
        self.ax.autoscale_view()
        plt.axes().set_aspect('equal', 'datalim')
        # Double the size of the canvas
        current_figure = plt.gcf()
        w, h = current_figure.get_size_inches()
        current_figure.set_size_inches(w*2, h*2)
        # Shrink current axis by 20% to make space for the legend
        box = self.ax.get_position()
        self.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        
        handles = []
        for i in range(num_clusters):
            handles.append(plt.Line2D((0,1),(0,0), color=cl_color[i]))
        plt.legend(handles, [ "cluster " + str(x) for x in range(1, num_clusters + 1)],
                   loc='center left', bbox_to_anchor=(1, 0.5))
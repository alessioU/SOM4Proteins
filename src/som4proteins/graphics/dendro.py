import scipy
from scipy.cluster.hierarchy import dendrogram

from som4proteins.cluster.cluster import Cluster
from som4proteins.graphics.graph import Graph


class Dendrogram(Graph):
    def __init__(self, linkage_matrix, cl_class, color_threshold,
                    cl_color):
        labels = [str(n) for n in range(1, len(cl_class)+1)
                                        if cl_class[n-1] != Cluster.NO_CLUSTER]
        scipy.cluster.hierarchy.set_link_color_palette(cl_color)
        dendrogram(linkage_matrix, labels=labels, color_threshold=color_threshold)
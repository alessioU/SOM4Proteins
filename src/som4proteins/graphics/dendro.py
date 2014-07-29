import scipy
from scipy.cluster.hierarchy import dendrogram

import matplotlib.pyplot as plt
from som4proteins.cluster.cluster import Cluster
from som4proteins.graphics.graph import Graph


class Dendrogram(Graph):
    def __init__(self, linkage_matrix, cl_class, color_threshold,
                    num_clusters, cl_color):
        labels = [str(n) for n in range(1, len(cl_class)+1)
                                        if cl_class[n-1] != Cluster.NO_CLUSTER]
        # Double the width of the canvas
        # TODO: make the width depends on the numbers of labels
        current_figure = plt.gcf()
        w, h = current_figure.get_size_inches()
        current_figure.set_size_inches(w*2, h)
        scipy.cluster.hierarchy.set_link_color_palette(cl_color)
        dendrogram(linkage_matrix, labels=labels, color_threshold=color_threshold)
        plt.legend([ "cluster " + str(x) for x in range(1, num_clusters + 1)])
import glob
import os

import params
from som4proteins.cluster.cluster import Cluster
from som4proteins.data.dataframe import DataFrame
from som4proteins.data.dataimporter import ProteinDataImporter
from som4proteins.graphics.dendro import Dendrogram
from som4proteins.graphics.grid import Grid
from som4proteins.som.maps.enums import Lattice, Shape
from som4proteins.som.maps.map import Map
from som4proteins.som.parameters.enums import PHASE, NEIGHBORHOOD
from som4proteins.som.parameters.parameters import TrainingParameters
from som4proteins.som.trainalgorithm import TrainAlgorithm


def check_odir_empty():
    if glob.glob(os.path.join(params.output_directory, "*.pdb")) != []:
        var = input("There are already some pdb files in the output directory,\n"+
              "shall I delete them? [y/n]:")
        if var != 'y':
            print('Delete or move them and try again.')
            exit(0)
        for f in glob.glob(os.path.join(params.output_directory, "*.pdb")):
            os.remove(f)

def read_dataframe(tfs, sfs, pns):
    check_odir_empty()
    dataFrame = DataFrame()
    for tf, sf, pn in zip(tfs, sfs, pns):
        try:
            p = ProteinDataImporter(trajectory_file=tf,
                                    structure_file=sf,
                                    protein_name=pn,
                                    output_dir=params.output_directory)
            dataFrame.join(p.get_coord_matrix())
        except Exception as e:
            print(e)
            exit(0)
    return dataFrame

def output_msg(mystr, newline=True):
    if newline:
        print(mystr)
    else:
        print(mystr, end="")

def main(msize, lattice, shape):
    output_msg('Loading data ... ', False)
    dataFrame = read_dataframe(params.trajectory_files, 
                               params.structure_files, params.protein_names)
    output_msg('Done.')
    dim = dataFrame.n_cols
    som_map = Map(map_size=msize, weight_vector_dim=dim, lattice=lattice, shape=shape)
    if params.neurons_weights is None:
        if params.map_initialization == 'linear':
            som_map.lininit(dataFrame.data)
        else:
            som_map.randinit()
        trainingParameters = TrainingParameters()
        trainingParameters.defaultParameters(PHASE.Rough, msize=msize)
        bt = TrainAlgorithm(som_map, dataFrame.data, trainingParameters)
        output_msg('Rough training (1st phase) ... ')
        bt.runBatch()
        output_msg('Done.')
        trainingParameters = TrainingParameters(neigh=params.map_neighborhood,
                                                radius_ini=params.initial_radius,
                                                radius_fin=params.final_radius,
                                                trainlen=params.training_length,
                                                msize=msize)
        bt = TrainAlgorithm(som_map, dataFrame.data, trainingParameters)
        output_msg('Training (2nd phase) ... ')
        bt.runBatch()
        output_msg('Done.')
        if (params.save_neurons_weight):
            filename = 'SOM_' + str(msize[0]) + 'x' + str(msize[1]) + \
                        '_' + params.jobname 
            file = os.path.join(params.output_directory, filename)
            som_map.save_neurons_weights(file)
    else:
        som_map.load_neurons_weights(params.neurons_weights)
    hits = som_map.som_hits(dataFrame.data)
    c = Cluster(som_map.neurons_weights, hits)
    cl_class = c.cluster_moj(method=params.cluster_method)
    c.calc_centroids()
    cl_best = c.calc_best()
    # TODO: add color palette as parameter.
    # In the dendrogram the blue color is for lines above threshold
    palette = [ 'green',
                'orange',                
                'violet',
                'gray',
                'yellow',
                'brown',
                'cyan',
                'magenta' ]
    d = Dendrogram(c.get_linkage_matrix(), cl_class, c.get_threshold(),
                palette)
    if params.save_dend_png:
        filename = 'dend_' + str(msize[0]) + 'x' + str(msize[1]) + \
                '_' + params.jobname + '.png'
        file = os.path.join(params.output_directory, filename)
        d.save(file)
    else:
        d.show()
    d.close()
    g = Grid(msize, lattice)
    if params.draw_clusters:
        g.add_clusters(cl_class, palette)
    if params.draw_hits:
        g.add_hits(hits)
    if params.draw_hits_numbers:
        g.add_numbers(hits)
    if params.draw_all_edges:
        g.add_edges()
    if params.draw_best_unit_edges:
        g.add_bestunits_edges(cl_class, cl_best)
    if params.save_grid_png:
        filename = 'SOM_' + str(msize[0]) + 'x' + str(msize[1]) + \
                '_' + params.jobname + '.png'
        file = os.path.join(params.output_directory, filename)
        g.save(file)
    else:
        g.show()
    g.close()
    
def check_lattice():
    if params.map_lattice not in Lattice.all_str():
        print("Error: possible values for lattice are "+ str(Lattice.all_str()) + \
              " but " + params.map_lattice + " is not one of them")
        exit(0)
    else:
        params.map_lattice = Lattice.to_int(params.map_lattice)

def check_shape():            
    if params.map_shape not in Shape.all_str():
        print("Error: possible values for shape are "+ str(Shape.all_str()) + \
              " but " + params.map_shape + " is not one of them")
        exit(0)
    else:
        params.map_shape = Shape.to_int(params.map_shape)
        
def check_tf_st_pn():
    if not(len(params.trajectory_files) == len(params.structure_files)\
           == len(params.protein_names)):
        print("Error: trajectory_files, structure_files and protein_names must have\n"+
                "the same number of parameters.") 
        exit(0)        


def check_init():
    if params.map_initialization not in ['linear', 'random']:
        print("Error: possible values for map_initialization are 'linear'"+\
              " or 'random' but " + params.map_initialization + " is not one of them") 
        exit(0)


def check_neigh():
    if params.map_neighborhood not in NEIGHBORHOOD.all_str():
        print("Error: possible values for map_initialization are "+ str(NEIGHBORHOOD.all_str()) +\
              " but " + params.map_initialization + " is not one of them") 
        exit(0)
    params.map_neighborhood = NEIGHBORHOOD.to_int(params.map_neighborhood)

if __name__ == '__main__':
    check_tf_st_pn()
    check_lattice()
    check_shape()
    check_init()
    check_neigh()
    main(params.map_size, params.map_lattice, params.map_shape)
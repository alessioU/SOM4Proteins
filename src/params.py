# Parameters for the
trajectory_files = ['../test/testfiles/trajcomb_ca_dt100_AHATV.xtc',
                    '../test/testfiles/trajcomb_ca_dt100_AHAVF.xtc']
structure_files = ['../test/testfiles/ahatv_CA_aligned.pdb',
                   '../test/testfiles/ahavf_CA_aligned.pdb']
protein_names = ['ahatv',
                 'ahavf']

jobname = 'test1'
output_directory = './prova'

# Path to a .npy file or None
neurons_weights = './prova/SOM_10x10_test1.npy'

# just batch for now, todo sequential
algorithm = 'batch'
initial_radius = 3
final_radius = 1
training_length = 300

map_size = [10, 10]
# sheet, toroid or cylinder
map_shape = 'sheet'
# hexagonal or rectangular
map_lattice = 'rectangular'
# linear or random
map_initialization = 'linear'
# gaussian, bubble, cut-gaussian, epanechicov
map_neighborhood = 'gaussian'

cluster_method = 'complete'

# if save_*_png == True the image is saved in the output dir
# otherwise the grid is just shown
save_grid_png = True
save_dend_png = True

save_neurons_weight = True

draw_clusters = True
draw_hits = True
draw_hits_numbers = True
draw_best_unit_edges = True
draw_all_edges = True

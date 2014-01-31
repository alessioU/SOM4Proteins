import os

class ProteinDataExporter():
    def __init__(self, dataframe, num_units, bmus, output_dir):
        self.dataframe = dataframe
        self.num_units = num_units
        self.bmus = bmus
        self.output_dir = output_dir
        
    def save_neurons(self):
        for n in range(self.num_units):
            structure_names = self.dataframe.row_labels[self.bmus == n]
            with open(os.path.join(self.output_dir, "neuron" + str(n) + ".pdb"), 'w') as output_f:
                for structure_name in structure_names:
                    with open(os.path.join(self.output_dir, structure_name + ".pdb"), 'r') as input_f:
                        input_f.readline()
                        output_f.write(input_f.read())

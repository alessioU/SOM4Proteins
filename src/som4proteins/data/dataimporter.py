'''dataimporter module

'''
import os
import re
import subprocess
from string import Template

from som4proteins.data.dataframe import DataFrame
import shutil


class ProteinDataImporter():
    '''Import the protein data from files.
    
        :param string trajectory_file: Path of the .xtc file.
        :param string structure_file: Path of the .pdb file.
        :param string protein_name: Name of the protein.
        :param string output_dir: Path of the output directory.
            
        
        :ivar string trajectory_file: Path of the trajectory (xtc) file.
        :ivar string structure_file: Path of the structure (pdb) file.
        :ivar string protein_name: Name of the protein.
        :ivar string output_dit: Path of the output dir.
        :ivar int num_frames: Number of frames.
        :ivar int num_atoms: Number of atoms for every frame.
            
    '''
    
    def __init__(self, trajectory_file, structure_file, protein_name, output_dir):
        self.trajectory_file = os.path.abspath(trajectory_file)
        self.structure_file = os.path.abspath(structure_file)
        self.protein_name = protein_name
        self.output_dir = os.path.abspath(output_dir)
        self._check_files_exist()
        self._read_num_frames_atoms()
    
    def _extract_coordinates(self):
        '''Extract the coordinates (pdb files) in the output directory.'''
        old_wd = os.getcwd()
        os.chdir(self.output_dir)
        t = Template("echo 3 3 | ${command} -f ${trajectory_file} -s ${structure_file} -sep -o ${protein_name}.pdb -fit progressive > /dev/null 2>&1")
        values = {'trajectory_file': self.trajectory_file,
                  'structure_file': self.structure_file,
                  'protein_name': self.protein_name,
                  'command': 'g_trjconv'}
        if shutil.which('g_trjconv') is not None:
            os.system(t.substitute(values))
        elif shutil.which('trjconv') is not None:
            values['command'] = 'trjconv'
            os.system(t.substitute(values))
        else:
            raise Exception("\nERROR: trjconv and g_trjconv couldn't be found, install GROMACS and try again.\n")
        os.chdir(old_wd)
        
    def get_coord_matrix(self):
        '''Return a :class:`.DataFrame` with the coordinates matrix.
        
        :returns:
            Matrix of the coordinates. Every row is a frame
            and every column is either a x,y or z coordinate of
            some C-alpha atom.
        :rtype: :class:`.DataFrame` 
        '''
        self._extract_coordinates()
        row_labels = [self.protein_name + str(num) for num in range(self.num_frames)]
        col_labels = self._read_col_labels()
        data = self._read_data()
        return DataFrame(data, row_labels, col_labels)

    def _read_data(self):
        '''Return a matrix of float.'''
        float_re = "[-+]?\d*\.?\d*"
        reATOM = re.compile("ATOM\s+\d+\s+\w+\s+\w+\s+\d+\s+("+ 
                            float_re + ")\s+(" + float_re + ")\s+(" + float_re + ").*")
        data = []
        pdbfile_names = [os.path.join(self.output_dir, self.protein_name + str(num) + ".pdb")
                         for num in range(self.num_frames)]
        for i, pdbfile_name in enumerate(pdbfile_names):
            data.append([])
            with open(pdbfile_name) as f:
                for line in f.readlines():
                    matchObj = reATOM.match(line)
                    if matchObj:
                        residueX, residueY, residueZ = matchObj.groups()
                        data[i].extend([float(residueX), float(residueY), float(residueZ)])
        return data

    def _read_col_labels(self):
        '''Return a list of string with the column labels.'''
        reATOM = re.compile("ATOM\s+\d+\s+(\w+)\s+\w+\s+(\d+).*")
        firstpdb_path = os.path.join(self.output_dir, self.protein_name + "0.pdb")
        col_labels = []
        
        with open(firstpdb_path) as f:
            for line in f.readlines():
                matchObj = reATOM.match(line)
                if matchObj:
                    atomName, residueNum = matchObj.groups()
                    for coord in ['x', 'y', 'z']:
                        col_labels.append(atomName + residueNum + coord)
        return col_labels
                     
    def _check_files_exist(self):
        '''Check that the xtc and pdb files exists.'''
        for f in [self.structure_file, self.trajectory_file]:
            try:
                with open(f): pass
            except IOError:
                raise IOError(' '.join(['The file', f, 'doesn\'t exist.']))
            
    def _read_num_frames_atoms(self):
        '''Read the number of frames and atoms.'''
        if shutil.which('g_gmxcheck') is not None:
            p = subprocess.Popen(["g_gmxcheck", '-f', self.trajectory_file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        elif shutil.which('gmxcheck') is not None:
            p = subprocess.Popen(["gmxcheck", '-f', self.trajectory_file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        else:
            raise Exception("\nERROR: gmxcheck and g_gmxcheck couldn't be found, install GROMACS and try again.\n")
        str_out = p.communicate()[0]
        self.num_frames = int(re.search(b'Coords\s*(\d+)', str_out).group(1))
        self.num_atoms = int(re.search(b'# Atoms\s*(\d+)', str_out).group(1))
'''map module

'''
import os

import numpy as np
from som4proteins.som.maps.enums import Lattice, Shape


class Map:
    '''SOM Map
    
    :param list_of_int map_size: Map grid size.
    :param int weight_vector_dim: Input space dimension(i.e. dimension of a weight vector).
    :param lattice: Map lattice.
    :type lattice: :class:`.Lattice`
    :param shape: Map shape.
    :type shape: :class:`.Shape`
    
    :ivar list_of_int size: Map grid size.
    :ivar int weight_vector_dim: Input space dimension(i.e. dimension of a weight vector).
    :ivar lattice: :class:`.Lattice`, map lattice.
    :ivar shape: :class:`.Shape`, map shape.
    :ivar int m_units: Number of neurons in the map(i.e. np.prod(size)).
    :ivar matrix neurons_weights: An m_units x weight_vector_dim matrix where every row represent a neuron.
    :ivar matrix sheet_coordinates: An m_units x len(size) matrix where every row is the coordinate of a map unit.
    :ivar matrix distances:
        An m_unit x m_units matrix where every row represent the distances of a map unit
        from all the others.
    '''
    def __init__(self, map_size, weight_vector_dim,
                 lattice=Lattice.Rect, shape=Shape.Sheet):
        self.size = map_size
        self.weight_vector_dim = weight_vector_dim
        self.lattice = lattice
        self.shape = shape
        self.sheet_coordinates = self._calculate_coordinates(self.size, self.lattice, Shape.Sheet)
        self._distances = self._calculate_distances()

    def randinit(self, data):
        '''Initializes a SOM with random values.
        
        For each component (xi), the values are uniformly
        distributed in the range of [min(xi), max(xi)].
        '''
        self._neurons_weights = np.random.rand(np.prod(self.size), self.weight_vector_dim)        
        for i in range(self.weight_vector_dim):
            ma = np.max(data[:, i])
            mi = np.min(data[:, i])
            self.neurons_weights[:, i] = (ma - mi) * self.neurons_weights[:, i] + mi
    
    def lininit(self, data):
        '''Initializes a SOM linearly along its greatest eigenvectors.
        
        Initializes a SOM linearly. The initialization is made by first calculating the eigenvalues
        and eigenvectors of the training data. Then, the map is initialized
        along the mdim greatest eigenvectors of the training data, where
        mdim is the dimension of the map grid.'''
        dataset_len, data_dim = data.shape
        mdim = len(self.size)
        if dataset_len < 2:
            raise Exception("Linear map initialization requires at least 2 data points.")
        # column means
        me = np.mean(data, axis=0)
        # compute principal components
        if data_dim > 1 and sum(np.array(self.size) > 1) > 1:
            # calculate mdim largest eigenvalues and their corresponding
            # eigenvectors
            
            # covariance matrix
            cov_mat = np.cov(data, rowvar=0, bias=1)
            
            # take mdim first eigenvectors with the greatest eigenvalues
            # cov_mat is always symmetric so we can use eigh
            eigval, eigvec = np.linalg.eigh(cov_mat, UPLO='U')
            ind_desc_eigval = np.flipud(np.argsort(eigval))
            eigvec = eigvec[:, ind_desc_eigval]
            eigvec = eigvec[:, 0:mdim]
            eigval = eigval[ind_desc_eigval]
            eigval = eigval[0:mdim]
            
            # normalize eigenvectors to unit length and multiply them by 
            # corresponding (square-root-of-)eigenvalues
            for i in range(mdim):
                eigvec[:,i] = (eigvec[:,i] / np.linalg.norm(eigvec[:,i])) * np.sqrt(eigval[i])
        else:
            eigvec = np.zeros((1, data_dim))
            for i in range(data_dim): 
                eigvec[i] = np.std(data[:,i], ddof=1)
        
        # initialize the weight vectors
        if data_dim > 1:
            self._neurons_weights = np.tile(me, (self.num_units, 1))
            Coords = self._calculate_coordinates(self.size, Lattice.Rect, Shape.Sheet)
            cox = np.copy(Coords[:, 0]); Coords[:, 0] = Coords[:, 1]; Coords[:, 1] = cox
            for i in range(mdim):
                ma = np.max(Coords[:, i]); mi = np.min(Coords[:, i])
                if ma > mi:
                    Coords[:, i] = (Coords[:, i] - mi) / (ma - mi)
                else:
                    Coords[:, i] = 0.5
            Coords = (Coords - 0.5) * 2
            for n in range(self.num_units):
                for d in range(mdim):
                    self.neurons_weights[n, :] = self._neurons_weights[n, :] + np.dot(Coords[n, d], eigvec[:, d].T)
        else:
            # TODO: test this
            self._neurons_weights = np.reshape(np.arange(self.num_units), (self.num_units, 1)) \
                                        /(self.num_units - 1) * (np.max(data) - np.min(data)) + np.min(data)

    @property
    def distances(self):
        return self._distances
    
    
    @property
    def neurons_weights(self):
        return self._neurons_weights


    @property
    def num_units(self):
        return np.prod(self.size)
    
    
    def _calculate_distances(self):
        '''Calculate distances between the map units in the output space.
        
        :returns: size = num_units x num_units
        :rtype: matrix
        '''
        num_units = self.num_units
        distances = np.zeros([num_units, num_units])
        
        # width and height of the grid
        dx = np.amax(self.sheet_coordinates[:, 0]) - np.amin(self.sheet_coordinates[:, 0])
        if self.size[0] > 1:
            dx = dx * self.size[0] / (self.size[0] - 1)
        else:
            dx = dx+1
        dy = np.amax(self.sheet_coordinates[:, 1]) - np.amin(self.sheet_coordinates[:, 1])
        if self.size[1] > 1:
            dy = dy * self.size[1] / (self.size[1] - 1)
        else:
            dy = dy + 1
        if self.shape == Shape.Sheet:
            for i in range(num_units - 1):
                inds = np.arange(i + 1, num_units)
                coords_differences = (self.sheet_coordinates[inds, :] - \
                        self.sheet_coordinates[np.ones(num_units - i - 1, dtype=int) * i, :]).T
                distances[i, inds] = np.sqrt(np.sum(coords_differences ** 2, axis=0))
        elif self.shape == Shape.Cylinder:
            for i in range(num_units - 1):
                inds = np.arange(i + 1, num_units)
                coords_differences = (self.sheet_coordinates[inds, :] - \
                        self.sheet_coordinates[np.ones(num_units - i - 1, dtype=int) * i, :]).T
                dist = np.sum(coords_differences ** 2, axis=0)
                # The cylinder shape is taken into account by adding and substracting
                # the width of the map (dx) from the x-coordinate (ie. shifting the
                # map right and left).
                coords_diff_shift = np.copy(coords_differences)
                coords_diff_shift[0, :] += dx  # East (x+dx)
                dist = np.minimum(dist, np.sum(coords_diff_shift ** 2, axis=0))
                coords_diff_shift = np.copy(coords_differences)
                coords_diff_shift[0, :] -= dx  # West (x-dx)
                dist = np.minimum(dist, np.sum(coords_diff_shift ** 2, axis=0))
                distances[i, inds] = np.sqrt(dist)
        elif self.shape == Shape.Toroid:
            for i in range(num_units - 1):
                inds = np.arange(i + 1, num_units)
                coords_differences = (self.sheet_coordinates[inds, :] - \
                        self.sheet_coordinates[np.ones(num_units - i - 1, dtype=int) * i, :]).T
                dist = np.sum(coords_differences ** 2, axis=0)
                # The toroid shape is taken into account as the cylinder shape was 
                # (see above), except that the map is shifted also vertically.
                coords_diff_shift = np.copy(coords_differences)
                coords_diff_shift[0, :] += dx  # East (x+dx)
                dist = np.minimum(dist, np.sum(coords_diff_shift ** 2, axis=0))
                coords_diff_shift = np.copy(coords_differences)
                coords_diff_shift[0, :] -= dx  # West (x-dx)
                dist = np.minimum(dist, np.sum(coords_diff_shift ** 2, axis=0))
                
                coords_diff_shift = np.copy(coords_differences)
                coords_diff_shift[1, :] += dy  # South (y+dy)
                dist = np.minimum(dist, np.sum(coords_diff_shift ** 2, axis=0))
                
                coords_diff_shift = np.copy(coords_differences)
                coords_diff_shift[1, :] -= dy  # North (y-dy)
                dist = np.minimum(dist, np.sum(coords_diff_shift ** 2, axis=0))
                
                coords_diff_shift = np.copy(coords_differences)
                coords_diff_shift[0, :] += dx; coords_diff_shift[1, :] -= dy # NorthEast (x+dx, y-dy)
                dist = np.minimum(dist, np.sum(coords_diff_shift ** 2, axis=0))
                
                coords_diff_shift = np.copy(coords_differences)
                coords_diff_shift[0, :] += dx; coords_diff_shift[1, :] += dy # SouthEast (x+dx, y+dy)
                dist = np.minimum(dist, np.sum(coords_diff_shift ** 2, axis=0))
                
                coords_diff_shift = np.copy(coords_differences)
                coords_diff_shift[0, :] -= dx; coords_diff_shift[1, :] += dy # SouthWest (x-dx, y+dy)
                dist = np.minimum(dist, np.sum(coords_diff_shift ** 2, axis=0))
                
                coords_diff_shift = np.copy(coords_differences)
                coords_diff_shift[0, :] -= dx; coords_diff_shift[1, :] -= dy # NorthWest (x-dx, y-dy)
                dist = np.minimum(dist, np.sum(coords_diff_shift ** 2, axis=0))
                distances[i, inds] = np.sqrt(dist)
        else:
            raise Exception("unkown shape") 
        
        return distances + distances.T
        

    def _bend_y_and_z_axis(self, coordinates):
        coordinates[:, 1] /= np.max(coordinates[:, 1])
        coordinates[:, 1] *= 2 * np.pi * self.size[0] / (self.size[0] + 1)
        coordinates[:, 2] += -np.min(coordinates[:, 2]) + 1
        coordinates[:, [1, 2]] = coordinates[:, [2, 2]] * \
                                 np.column_stack((np.cos(coordinates[:, 1]),
                                                  np.sin(coordinates[:, 1])))
        return coordinates
    
    def _bend_x_and_z_axis(self, coordinates):
        coordinates[:, 0] /= np.max(coordinates[:, 0])
        coordinates[:, 0] *= 2 * np.pi * self.size[1] / (self.size[1] + 1)
        coordinates[:, [0, 2]] = np.column_stack((np.cos(coordinates[:, 0]),
                                                  np.sin(coordinates[:, 0])))
        return coordinates
    
    def _calculate_coordinates(self, size, lattice, shape):
        num_dimensions = len(size)
        num_units = np.prod(size)
        coordinates = np.zeros([num_units, num_dimensions])
        
        indices = np.arange(num_units)
        coordinates[:, 0] = np.floor(indices / size[0])
        coordinates[:, 1] = indices % size[0]
        
        if lattice == Lattice.Hex:
            inds_for_row = (np.cumsum(np.ones(size[1], dtype=int)) - 1) * size[0]
            for i in range(1, size[0], 2):
                coordinates[inds_for_row + i, 0] += 0.5
        
        if shape == Shape.Sheet:
            if lattice == Lattice.Hex:
                coordinates[:, 1] *= np.sqrt(0.75)
        elif shape == Shape.Cylinder:
            if num_dimensions == 2:
                coordinates = np.append(coordinates, np.ones([num_units, 1]), axis=1)
                num_dimensions = 3
                
            coordinates = self._bend_x_and_z_axis(coordinates)
        elif shape == Shape.Toroid:
            if num_dimensions == 2:
                coordinates = np.append(coordinates, np.ones([num_units, 1]), axis=1)
                num_dimensions = 3
            
            coordinates = self._bend_x_and_z_axis(coordinates)
            coordinates = self._bend_y_and_z_axis(coordinates)
        return coordinates        
    
    def som_hits(self, D):
        '''Calculate the response of the given data on the map.
        
        :returns: The number of hits in each map unit, length = munits
        :rtype: array of int
        '''
        hits = np.zeros(self.num_units)
        # calculate BMUs
        bmus, _ = self.calc_bmus(D)
        # for each unit, check how many hits it got
        for i in range(self.num_units):
            hits[i] = np.sum(bmus == i)
        return hits, bmus
        
    def calc_bmus(self, D):
        '''Finds Best-Matching Units (BMUs) for given data vector from a given map.
           
        :return: 1) The requested BMUs for each data vector.
            
            2) The corresponding quantization errors.
        :rtype: 1) array of int, length = dataset_len.
        
                2) array of float, length = dataset_len.
        '''
        M = self.neurons_weights
        dataset_len, dataset_dim = D.shape
        bmus = np.zeros(dataset_len)
        qerrors = np.zeros(dataset_len)
        # The BMU search involves calculating weighted Euclidian distances 
        # to all map units for each data vector. Basically this is done as
        #   for i in range(dataset_len): 
        #     for j in range(num_units):
        #       for k in range(dim):
        #         Dist[j,i] = Dist[j,i] + (D[i,k] - M[j,k])**2;
        # where dim is the dimension of a weight vector,
        #       Dist[j,i] is distance of the j-th map unit from the i-th data vector,
        #       D is the dataset and M the map. However, taking 
        # into account that distance between vectors m and v can be expressed as
        #   abs(m - v)**2 = sum_i ((m_i - v_i)**2) = sum_i (m_i**2 + v_i**2 - 2*m_i*v_i)
        # this can be made much faster by transforming it to a matrix operation:
        #   Dist = (M**2)*W1 + ones(m,1)*ones(dim,1)'*(D'**2) - 2*M*D'
        #
     
        # calculate distances & bmus
        
        # This is done a block of data at a time rather than in a
        # single sweep to save memory consumption. The 'Dist' matrix has 
        # size munits*blen which would be HUGE if you did it in a single-sweep
        # operation. If you _want_ to use the single-sweep version, just 
        # set blen = dlen. If you're having problems with memory, try to 
        # set the value of blen lower. 
        blen = np.amin([self.num_units, dataset_len])
        W1 = np.ones((dataset_dim, dataset_len))
        
        # constant matrices
        WD = 2 * D.T
        dconst = np.dot((D**2), np.ones(dataset_dim)); # constant term in the distances
        
        i = 0
        while i + 1 <= dataset_len: 
            # calculate distances
            inds = np.arange(i, np.amin([dataset_len, i + blen]))
            i = i + blen
            Dist = np.dot(M ** 2, W1[:, inds]) - np.dot(M, WD[:, inds]) # plus dconst for each sample
            # find the bmus and the corresponding quantization errors
            B = np.argmin(Dist, axis=0)
            Q = np.min(Dist, axis=0)
            if self.num_units == 1:
                bmus[inds] = 1
            else:
                bmus[inds] = B
            qerrors[inds] = Q.T + dconst[inds]
        return bmus, np.sqrt(np.abs(qerrors))
    
    def save_neurons_weights(self, file):
        np.save(file, self._neurons_weights)
    
    def load_neurons_weights(self, file):
        self._neurons_weights = np.load(file)
    
    def save_neuron_data(self, dataframe, bmus, file):
        ''' Save neurons' data file
        
        On every line there's the number of the neuron, followed by the
        labels of the data won by that neuron ordered by increasing distance
        '''
        with open(file, mode='w') as f:
            # TODO: make it more efficient
            for i in range(self.num_units):
                idx = np.arange(len(dataframe.data))[bmus == i]
                data = dataframe.data[idx]
                unit = np.tile(self._neurons_weights[i], (len(data), 1))
                dist = np.sqrt(np.sum(np.square(data - unit), axis=1))
                sorted_idx = idx[np.argsort(dist)]
                labels = dataframe.row_labels[sorted_idx]
                f.write(str(i+1) + '\t' + ', '.join(labels) + '\n')
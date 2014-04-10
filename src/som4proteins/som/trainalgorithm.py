"""trainalgorithm module

"""
from som4proteins.som.parameters.enums import NEIGHBORHOOD, SampleOrderType,\
    ALPHATYPE
import numpy as np
from scipy.sparse import csc_matrix

# TODO: class SequentialTrain(SOMTrain)

class TrainAlgorithm():
    """Batch som algorithm.
    
    :param som_map: Map of the SOM
    :type som_map: :class:`.Map`
    :param data: Training data
    :type data: numpy matrix
    :param parameters: Training parameters
    :type parameters: :class:`.Parameters`
    """
    
    def __init__(self, som_map, data, parameters):
        self.som_map = som_map
        self.data = data
        
        self.neigh = parameters.neigh
        self.trainlen = parameters.trainlen
        self.radius = parameters.radius
        self.sample_order_type = parameters.sample_order_type
        self.alpha_type = parameters.alpha_type
        self.alpha_ini = parameters.alpha_ini
    
    def runSequential(self):
        D = self.data
        dataset_len, dataset_dim = D.shape
        M = self.som_map.neurons_weights
        neurons_distances = self.som_map.distances
        num_units = self.som_map.num_units
        trainlen = self.trainlen * dataset_len
        if self.alpha_type == ALPHATYPE.Inv: 
            # alpha[t] = a / (t+b), where a and b are chosen suitably
            # below, they are chosen so that alpha_fin = alpha_ini/100
            b = (trainlen - 1) / (100 - 1)
            a = b * self.alpha_ini
        mu_x_0 = np.zeros(num_units, dtype=int)
        # distance between map units in the output space
        # Since in the case of gaussian and ep neighborhood functions, the 
        # equations utilize squares of the unit distances and in bubble case
        # it doesn't matter which is used, the unitdistances and neighborhood
        # radiuses are squared.
        Ud = neurons_distances ** 2
        update_step = 100
        for t in range(trainlen):
            # Every update_step iterations, new values for sample indices, neighborhood
            # radius and learning rate are calculated. This could be done
            # every step, but this way it is more efficient. Or this could 
            # be done all at once outside the loop, but it would require much
            # more memory.
            ind = t % update_step
            if ind == 0:
                steps = np.arange(t, np.minimum(trainlen, t + update_step))
                # sample order
                if self.sample_order_type == SampleOrderType.ORDERED: 
                    samples = steps % dataset_len
                elif self.sample_order_type == SampleOrderType.RANDOM:
                    samples = np.floor(dataset_len * np.random.rand(update_step))
                else:
                    raise("Error: Order type unkown, the order type must be ordered or random.")
                # neighborhood radius
                # TODO: move _calc_radius from TrainingParameters to here
                r = self.radius[steps]
                r = r ** 2  # squared radius (see notes about Ud above)
                r[r==0] = np.finfo(np.float64).eps # zero radius might cause div-by-zero error
                # learning rate
                if self.alpha_type == ALPHATYPE.Linear:
                    alpha = (1 - (steps+1)/trainlen) * self.alpha_ini
                elif self.alpha_type == ALPHATYPE.Inv:
                    alpha = a / (b + steps)
                elif self.alpha_type == ALPHATYPE.Power:
                    alpha = self.alpha_ini * (0.005 / self.alpha_ini) ** (steps / trainlen)
                else:
                    raise("Error: Alpha type unkown, the alpha type must be linear or inv or power.")
            # find BMU
            x = D[samples[ind], :] # pick one sample vector
            Dx = M - np.array([x])[mu_x_0, :] # each map unit minus the vector
            bmu = np.min(np.dot((Dx ** 2), np.ones(dataset_dim))) # minimum distance(^2) and the BMU
            # neighborhood & learning rate
            # notice that the elements Ud and radius have been squared!
            # (see notes about Ud above)
            if self.neigh == NEIGHBORHOOD.Bubble:
                H = neurons_distances[:, bmu] <= r[ind]
            elif self.neigh == NEIGHBORHOOD.Gaussian:
                H = np.exp(-neurons_distances[:, bmu] / (2*r[ind]))
            elif self.neigh == NEIGHBORHOOD.CutGaussian:
                H = np.exp(-neurons_distances[:, bmu]/(2*r[ind])) * (neurons_distances[:, bmu] <= r[ind])
            elif self.neigh == NEIGHBORHOOD.Epanechicov:
                H = (1-neurons_distances[:, bmu]/r[ind]) * (neurons_distances[:, bmu] <= r[ind])
            else:
                raise("Error: Neighborhodd type unkown.")
            H = H * alpha[ind]
            # update M
            M = M - H[:, np.zeros(dataset_dim, dtype=int)] * Dx
               
    def runBatch(self):
        """Run the batch version of the algorithm
        """
        D = self.data
        dataset_len, dataset_dim = D.shape
        M = self.som_map.neurons_weights
        neurons_distances = self.som_map.distances
        num_units = self.som_map.num_units
        
        # distance between map units in the output space
        # Since in the case of gaussian and ep neighborhood functions, the 
        # equations utilize squares of the unit distances and in bubble case
        # it doesn't matter which is used, the unitdistances and neighborhood
        # radiuses are squared.
        neurons_distances = neurons_distances ** 2
        radius = self.radius ** 2
        # zero neighborhood radius may raise a division by zero exception
        radius[radius==0] = np.finfo(np.float64).eps 
        # The training algorithm involves calculating weighted Euclidian distances 
        # to all map units for each data vector. Basically this is done as
        #   for i in range(dataset_len): 
        #     for j in range(num_units): 
        #       for k in range(dim):
        #         Dist[j,i] = Dist[j,i] + (D[i,k] - M[j,k])**2;
        # where dim is the dimension of a weight vector,
        #       Dist[j,i] is distance of the j-th map unit from the i-th data vector,
        #       D is the dataset and M the map. 
        # However, taking into account that distance between vectors m and v can be expressed as
        #   abs(m - v)**2 = sum_i ((m_i - v_i)**2) = sum_i (m_i**2 + v_i**2 - 2*m_i*v_i)
        # this can be made much faster by transforming it to a matrix operation:
        #   Dist = (M**2)*W1 + ones(m,1)*ones(dim,1)'*(D'**2) - 2*M*D'
        # Of the involved matrices, several are constant, as the mask and data do 
        # not change during training. Therefore they are calculated beforehand.
        W1 = np.ones((dataset_dim, dataset_len))
        # constant matrices
        WD = 2 * D.T
        # With the 'blen' parameter you can control the memory consumption 
        # of the algorithm, which is in practice directly proportional
        # to num_units*blen. If you're having problems with memory, try to 
        # set the value of blen lower. 
        blen = np.amin([num_units,dataset_len])
        #blen = np.amin([1500,dataset_len])
    
        # reserve some space
        bmus = np.zeros(dataset_len)
        
        ones_dlen_dim = np.ones([dataset_len, dataset_dim])
        ones_dlen = np.ones(dataset_len)
        range_dlen = range(dataset_len)
        for t in range(self.trainlen):
            if (t+1) % 100 == 0:
                print("%d / %d" % (t + 1, self.trainlen))
            # batchy train - this is done a block of data (inds) at a time
            # rather than in a single sweep to save memory consumption. 
            # The 'Dist' and 'Hw' matrices have size munits*blen
            # which - if you have a lot of data - would be HUGE if you 
            # calculated it all at once. A single-sweep version would 
            # look like this: 
            #    Dist = np.dot(M ** 2, W1) - np.dot(M, WD)
            #    bmus = np.argmin(Dist, axis=0)
            # This "batchy" version is the same as single-sweep if blen=dlen.
            if blen == dataset_len:
                Dist = np.dot(M ** 2, W1) - np.dot(M, WD)
                bmus = np.argmin(Dist, axis=0)
            else:
                i = 0
                while i + 1 <= dataset_len:
                    inds = np.arange(i, np.minimum(dataset_len, i + blen))
                    i = i + blen
                    Dist = np.dot(M**2, W1[:, inds]) - np.dot(M, WD[:, inds])
                    bmus[inds] = np.argmin(Dist, axis=0)
            # neighborhood 
            # notice that the elements neurons_distances and radius have been squared!
            if self.neigh == NEIGHBORHOOD.Bubble:
                H = neurons_distances <= radius[t]
            elif self.neigh == NEIGHBORHOOD.Gaussian:
                H = np.exp(-neurons_distances/(2*radius[t]))
            elif self.neigh == NEIGHBORHOOD.CutGaussian:
                H = np.exp(-neurons_distances/(2*radius[t])) * (neurons_distances <= radius[t])
            elif self.neigh == NEIGHBORHOOD.Epanechicov:
                H = (1-neurons_distances/radius[t]) * (neurons_distances <= radius[t])
            else:
                raise NotImplementedError
            
            # update 
            
            # In principle the updating step goes like this: replace each map unit 
            # by the average of the data vectors that were in its neighborhood.
            # The contribution, or activation, of data vectors in the mean can 
            # be varied with the neighborhood function. This activation is given 
            # by matrix H. So, for each map unit the new weight vector is
            # 
            # m = sum_i (h_i * d_i) / sum_i (h_i),
            # 
            # where i denotes the index of data vector.  Since the values of
            # neighborhood function h_i are the same for all data vectors belonging to
            # the Voronoi set of the same map unit, the calculation is actually done
            # by first calculating a partition matrix P with elements p_ij = 1 if the
            # BMU of data vector j is i.
            
            P = csc_matrix((ones_dlen, [bmus, range_dlen]),
                           shape=(num_units, dataset_len))
                           
            # Then the sum of vectors in each Voronoi set are calculated (P*D) and the
            # neighborhood is taken into account by calculating a weighted sum of the
            # Voronoi sum (H*). The "activation" matrix A is the denominator of the 
            # equation above.
            S = np.dot(H, P.dot(D)) 
            A = np.dot(H, P.dot(ones_dlen_dim))
            # only update units for which the "activation" is nonzero
            nonzero = A > 0
            M[nonzero] = S[nonzero] / A[nonzero]
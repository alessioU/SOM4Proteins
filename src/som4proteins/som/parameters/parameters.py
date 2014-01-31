"""Parameters for the som algorithm module.
"""

import numpy as np
from som4proteins.som.parameters.enums import NEIGHBORHOOD
from som4proteins.som.parameters.enums import PHASE, ALPHATYPE


class TrainingParameters:
    """Represent the parameters for the som training algorithm."""
    
    def __init__(self, neigh=None, alpha_type=None, alpha_ini=None,
                 radius_ini=None, radius_fin=None, trainlen=None, msize=None,
                 sample_order_type=None):
        self.neigh = neigh
        self.radius_ini = radius_ini
        self.radius_fin = radius_fin
        self.alpha_type = alpha_type
        self.alpha_ini = alpha_ini
        self.trainlen = trainlen
        if radius_ini != None and radius_fin != None and trainlen != None:
            self.radius = self._calc_radius(radius_ini, radius_fin, trainlen)
        else:
            self.radius = None
        self.msize = msize
        self.sample_order_type = sample_order_type
        
    def defaultParameters(self, phase, msize=[10, 10], dataset_len=None):
        """Used to give sensible values for SOM training parameters.
        
        These parameters depend on the number of training samples, 
        phase of training and map size. 
        
        Parameters
        ----------
        phase : :class: `.PHASE`
            Training phase
        msize : vector
            Map size
        dataset_len : int, optional
            Length of the training data"""
        #TODO check that values are consistent with the phase chosen
        
        # default neighborhood function
        self.neigh = NEIGHBORHOOD.Gaussian
        
        # learning rate (sequential training only)
        if phase == PHASE.Rough or phase == PHASE.Train:
            self.alpha_ini = 0.5
        elif phase == PHASE.FineTune:
            self.alpha_ini = 0.05
            
        # learning rate (alpha type) (sequential training only)
        self.alpha_type = ALPHATYPE.Inv
        
        # radius
        ms = np.max(msize)
        self.radius_ini = np.amax([1, np.ceil(ms/4)])
    
        if phase == PHASE.Rough:
            self.radius_fin = np.amax([1,self.radius_ini/4])
        else:
            self.radius_fin = 1
        
        # trainlen 
        if msize != None and dataset_len != None:
            munits = np.prod(msize) 
            mpd = munits / dataset_len 
        else:
            mpd = 0.5
        if phase == PHASE.Train: 
            self.trainlen = np.ceil(50*mpd)
        elif phase == PHASE.Rough:
            self.trainlen = np.ceil(10*mpd)
        elif phase == PHASE.FineTune:
            self.trainlen = np.ceil(40*mpd)
        
        self.trainlen = int(np.amax([1, self.trainlen]))
        
        self.radius = self._calc_radius(self.radius_ini, self.radius_fin, self.trainlen)
        
    def _calc_radius(self, r_init, r_fin, trainlen):
        return r_fin + np.arange(trainlen)[::-1]/(trainlen-1) * (r_init - r_fin)
    
    def setParameters(self, neigh=None, radius_ini=None, radius_fin=None, alpha_type=None,
                        trainlen=None, alpha_ini=None, msize=None):
        if neigh != None:
            self.neigh = neigh
        if radius_ini != None:
            self.radius_ini = radius_ini
        if radius_fin != None:
            self.radius_fin = radius_fin
        if alpha_type != None:
            self.alpha_type = alpha_type
        if alpha_ini != None:
            self.alpha_ini = alpha_ini
        if trainlen != None:
            self.trainlen = trainlen
        if radius_ini != None and radius_fin != None and trainlen != None:
            self.radius = self._calc_radius(radius_ini, radius_fin, trainlen)
        if msize != None:
            self.msize = msize
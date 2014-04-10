class PHASE:
    '''Phase of the som training algorithm.
    
    :cvar Train:
        Map training in a onepass operation, as opposed to the
        rough-finetune combination.
    :cvar Rough:
        Rough organization of the map: large neighborhood, big
        initial value for learning coefficient. Short training.
    :cvar FineTune:
        Finetuning the map after rough organization phase. Small
        neighborhood, learning coefficient is small already at 
        the beginning. Long training.'''
    
    Train, Rough, FineTune = range(3)
    
class ALPHATYPE:
    '''The learning rate (alpha) goes from the alpha_ini to 0 along the function defined by ALPHATYPE
    
    See Training Parameters http://www.cis.hut.fi/somtoolbox/documentation/somalg.shtml
    
    
    :cvar Linear:
        Linear learning rate function
    :cvar Inv:
        Inverse of time
    :cvar Power:
        Power series
    '''
    
    Linear, Inv, Power = range(3)
    
class NEIGHBORHOOD:
    '''Type of neighborhood.
        
        The four neighborhood functions: bubble, gaussian, cut gaussian and epanechicov
        
        See http://www.cis.hut.fi/somtoolbox/documentation/somalg.shtml'''        
    # TODO: complete documentation
    Bubble, Gaussian, CutGaussian, Epanechicov = range(4)
    
    @classmethod
    def all_str(self):
         return [ 'bubble', 'gaussian', 'cut-gaussian', 'epanechicov']
     
    @classmethod
    def to_int(cls, neigh):
        res = {'bubble':0,'gaussian':1, 'cut-gaussian':2, 'epanechicov':3}
        return res[neigh]
    
class SampleOrderType:
    ORDERED, RANDOM = range(2)
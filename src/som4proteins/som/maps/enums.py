class Lattice:
    Rect, Hex = range(2)
    
    @classmethod
    def all_str(self):    
        return ['rectangular', 'hexagonal']
    
    @classmethod
    def to_str(cls, lattice):
        res = ['rectangular', 'hexagonal']
        return res[lattice]
    
    @classmethod
    def to_int(cls, lattice):
        res = {'rectangular':0, 'hexagonal':1}
        return res[lattice]
    
class Shape:
    Sheet, Toroid, Cylinder = range(3)

    @classmethod
    def all_str(self):    
        return ['sheet', 'toroid', 'cylinder']
        
    @classmethod
    def to_str(cls, shape):
        res = ['sheet', 'toroid', 'cylinder']
        return res[shape]
    
    @classmethod
    def to_int(cls, lattice):
        res = {'sheet':0, 'toroid':1, 'cylinder':2}
        return res[lattice]
import os

import matplotlib.pyplot as plt


class Graph:
    def show(self):
        plt.show()
        
    def save(self, path, name):
        plt.savefig(os.path.join(path, name+'.png'))
        
    def close(self):
        plt.close()
import matplotlib.pyplot as plt


class Graph:
    def show(self):
        plt.show()
        
    def save(self, file):
        plt.savefig(file)
        
    def close(self):
        plt.close()
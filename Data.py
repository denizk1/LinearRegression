import numpy as np

class Data():
    def __init__(self,data):
        self.data= np.genfromtxt(data, delimiter = ',', skip_header=True)
    def ReturnData(self):
        self.x_data=self.data[:,:-1]
        self.y_data=self.data[:,-1]

        return self.x_data,self.y_data
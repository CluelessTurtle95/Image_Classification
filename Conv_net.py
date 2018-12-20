import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class network:
    def __init__(self , Num_layers , Element_Num_layers ):
        self.Num_layers = Num_layers
        self.l = createlayers(Num_layers , Element_Num_layers)
        self.theta = list()
        for j in range(len(self.l) - 1) :
            self.theta.append( np.zeros( len(self.l[j+1]) , len(self.l[j]) + 1 ) )
        
    def add_bias(self):
        return np.ones( self.Num_layers - 1 )
    
    def remove_bias(self):        
        for v in self.l :
            np.delete( v , 0)
    
    def Forward_Propogation(self , x):
        self.l[0] = x
        b = self.add_bias()
        for i in range(self.Num_layers - 1):
            np.insert(self.l[i] , 0 , b[i]) 
            self.l[i + 1] = self.theta[i].dot( self.l[i] )
            if i+1 != self.Num_layers - 1 : 
                self.relu(i+1)
            else:
                self.softmax(i+1)
                
    
    def relu(self , j):
        self.l[j][ self.l[j] < 0 ]  = 0 

    def softmax(self , j):
        t = np.exp(self.l[j])
        self.l[j] = t / np.sum(t)


def createlayers( Num  , layer_elements):
    a = list()
    for i in range(Num):
        a.append( np.zeros( ( layer_elements[i] , 1) ) )
    return a

    

class manager:
    pass


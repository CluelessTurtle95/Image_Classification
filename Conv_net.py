import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class network:
    def __init__(self , Element_Num_layers ): # element_num_layers like  [ 10 , 4 , 4 , 2 , 2] for network with two hidden layers of 4 each and two classes 
        self.weights = list()
        self.error = list()
        self.b = list()
        self.G_weights = list()
        self.G_b = list()
        self.Num_layers = len(Element_Num_layers)
        self.l = createlayers(Element_Num_layers)
        self.z = [None] * 3
        for index , layer in enumerate(self.l[0 : - 2] ) :  # random weights
            self.weights.append( np.random.rand( len(self.l[index + 1]) , len(layer)  ) )
        for i in Element_Num_layers[1 : -1 ] : # random biases
            self.b.append(np.random.rand( i , 1)) 
        

    def Forward_Propogation(self , x):
        self.l[0] = x
        self.l[0] = self.l[0].reshape( np.size(self.l[0]) , 1)
        for i in range(self.Num_layers - 2):
            self.z[i] = self.weights[i]  @  self.l[i]  +   self.b[i]
            self.l[i + 1] = self.relu(self.z[i])

        self.l[self.Num_layers - 1] = self.softmax(self.l[self.Num_layers - 2])

    def J_prime(self , X , result , data_index , lambd):
        self.G_b = [None] * len(self.b)
        self.G_weights = [None] * len(self.weights)
        
        for index , b in enumerate(self.b):
            self.G_b[index] = np.zeros_like(b)
            self.G_weights[index] = np.zeros_like(self.weights[index])
        
        for m in range(data_index):
            self.gradient(X[ : , m] , result[: , m]  , lambd)
        
        for index , b  in enumerate(self.b):
            self.G_b[index] = self.G_b[index] / data_index
            self.G_weights[index] = self.G_weights[index] / data_index
         
    def relu(self , t):
        t[ t < 0 ]  = 0
        return t 

    def relu_p(self , t):
        t[t < 0] = 0
        t[t > 0] = 1
        return t

    def softmax(self , t):
        t = np.exp(t)
        t = t / np.sum(t)
        return t

    def delta(self , y):
        y = y.reshape(np.size(y),1)
        self.error = list()
        self.error.insert( 0 , (self.l[-1] - y) / self.Num_layers ) 
        
        self.error.insert( 0 , self.error[0] * self.l[-1] * (1 - self.l[-1]) ) # L_sub(l) = self.l[-1]
        for i in range(2 , self.Num_layers ):
            self.error.insert(0 , np.transpose(self.weights[self.Num_layers - 1 - i]) @ (self.error[0] * self.relu_p( self.z[self.Num_layers - 1 - i])))

    def gradient(self , x , y , lambd):
        self.Forward_Propogation(x) # self.Num_layers - 1 - i = l - k (see hand calculations)
        self.delta(y)
        for i in range(self.Num_layers - 2):
            self.G_b[i] +=  self.error[i + 1] * self.relu_p(self.z[i]) 
            self.G_weights[i] += ( self.error[i + 1] * self.relu_p(self.z[i]) ) @ np.transpose(self.l[i]) + lambd * self.weights[i]

    def learn(self , X , result  , data_index , lambd , alpha , max_iter):
        for i in range(max_iter):
            self.J_prime( X , result , data_index , lambd)
            for i in range(self.Num_layers - 2):
                self.b[i]  = self.b[i] - alpha * self.G_b[i]
                self.weights[i]  = self.weights[i] - alpha * self.G_weights[i]
    
    def cost(self , X , result , data_index):
        b = 0
        for i in range(data_index):
            self.Forward_Propogation(X[: , i])
            a = self.l[self.Num_layers - 1].reshape( np.size(self.l[self.Num_layers - 1] ) , 1) - result[: , i].reshape( np.size(result[: , i]) , 1)
            a = a*a
            b += np.sum(a) / np.size(a)
        b = b / data_index
        return b
        
def createlayers(layer_elements):
    a = list()
    for i in layer_elements:
        a.append( np.zeros( (i , 1) ) )
    return a

h = createlayers([10 , 4 , 4 , 2 , 2])
#print(h[0] , np.len(h[0]))

net = network([10 , 4 , 4 , 2 , 2])
X = np.random.rand(10 , 20)
result  = np.random.rand(2 , 20)

net.learn( X , result , 8 , 0.01 , 0.01 , 10)

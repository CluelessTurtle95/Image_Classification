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
        
        # setting random feature maps
        self.fm1 = np.random.rand( 1 , 2 , 5 , 5)
        self.fm2 = np.random.rand( 2 , 2 , 3 , 3)
        self.fm3 = np.random.rand( 4 , 3 , 3 , 3)
    
    def Forward_Propogation_FC(self , x):
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

    def delta_FC(self , y):
        y = y.reshape(np.size(y),1)
        self.error = list()
        self.error.insert( 0 , (self.l[-1] - y) / self.Num_layers ) 
        
        self.error.insert( 0 , self.error[0] * self.l[-1] * (1 - self.l[-1]) ) # L_sub(l) = self.l[-1]
        for i in range(2 , self.Num_layers ):
            self.error.insert(0 , np.transpose(self.weights[self.Num_layers - 1 - i]) @ (self.error[0] * self.relu_p( self.z[self.Num_layers - 1 - i])))

    def gradient(self , x , y , lambd):
        self.Forward_Propogation_FC(x) # self.Num_layers - 1 - i = l - k (see hand calculations)
        self.delta_FC(y)
        for i in range(self.Num_layers - 2):
            self.G_b[i] +=  self.error[i + 1] * self.relu_p(self.z[i]) 
            self.G_weights[i] += ( self.error[i + 1] * self.relu_p(self.z[i]) ) @ np.transpose(self.l[i]) + lambd * self.weights[i]

    def learn(self , X , result  , data_index , lambd , alpha , max_iter):
        
        cost_saver = [None] * max_iter
        for i in range(max_iter):
            self.J_prime( X , result , data_index , lambd)
            for j in range(self.Num_layers - 2):
                self.b[j]  = self.b[j] - alpha * self.G_b[j]
                self.weights[j]  = self.weights[j] - alpha * self.G_weights[j]
            cost_saver[i] = self.cost(X , result , data_index)
            
        return cost_saver
    
    def cost(self , img , result , data_index):
        b = 0
        for i in range(data_index):
            self.Forward_Propogation_conv(img[i , : , : , :])
            a = self.l[self.Num_layers - 1].reshape( np.size(self.l[self.Num_layers - 1] ) , 1) - result[: , i].reshape( np.size(result[: , i]) , 1)
            a = a*a
            b += np.sum(a) / np.size(a)
        b = b / data_index
        return b
    
    def plot_cost(self , X , result , data_index , lambd , alpha , max_iter):
        axis_x = np.linspace( 0  , max_iter , 1 + max_iter)
        axis_y = self.learn(X ,result , data_index , lambd , alpha , max_iter)
        axis_y.append(axis_y[99])
        plt.plot(axis_x , axis_y )
        plt.xlabel( "Iterations" )
        plt.ylabel( "Cost" )
        plt.title( "Cost Vs Iterations" )
        plt.show()
    
    def convolve(self , img , feature_map): # features should be odd x odd to keep image sizes even for pooling
        Num_channel = np.size( img , axis = 0)
        Num_features = np.size(feature_map , axis = 1)
        if ( Num_channel != np.size(feature_map , axis = 0) ):
            raise Exception
        final = np.zeros( Num_channel * Num_features , np.size(img , axis = 1) - np.size(feature_map , axis = 2)  + 1 , np.size(img , axis = 2) - np.size(feature_map , axis = 3)  + 1 )
        # final image is a litle bit smaller but has many more channels. 
        for i in range(Num_channel): # i is the depth channel
            initial = img[ i , : , : ]
            for j in range( Num_features ): # j is the number of feature matrices per channel\\\
                final[ j + Num_features * i , : , : ] = self.conv( initial , feature_map[i , j , : , :] )
        return final
        
    def conv(self , img  , feature):
        result  = np.zeros(  np.size(img , axis = 0) - np.size(feature , axis = 0)  + 1 , np.size(img , axis = 1) - np.size(feature , axis = 1)  + 1 )
        for i in range(np.size(result  , axis = 0)):
            for j in range(np.size(result  , axis = 1)):
                result[ i , j ] = img[ i : i + np.size(feature  , axis = 0)  , j : j + np.size(feature  , axis = 1)  ] * feature
        return result
    
    def max_pooling(self  , img  , pool_map):
        Num_channel = np.size(img , axis = 0)
        final = np.zeros( np.size( img  , axis = 0)  , np.size( img  , axis = 1) / pool_map[0] , np.size( img  , axis = 2)  / pool_map[1] )
        for i in range(Num_channel):
            initial = img[i , : , :]
            final[i , : , :] = self.max_pool( initial  , pool_map ) 
        return final

    def max_pool(self , img  , pool_map):
        # pool map is numpy array of pooling size like [4 , 4] for 4 x 4
        final = np.zeros( np.size( img  , axis=0) / pool_map[0], np.size( img  , axis=1) / pool_map[1]) 
        for i in range( 0  , np.size(img  , axis=0) - pool_map[0], pool_map[0] ):
            for j in range( 0  , np.size(img  , axis=1) - pool_map[1], pool_map[1] ):
                final[ i / pool_map[0] , j / pool_map[1] ]  = np.max( img[ i : i + pool_map[0] ,  j : j + pool_map[1]] )
        return final

    def Forward_Propogation_conv(self , img ):
        # 3 convololutional layers
        img = self.convolve( img , self.fm1 )
        img = self.relu(img)
        img = self.max_pooling( img , np.array([2 , 2]) )

        img = self.convolve( img , self.fm2 )
        img = self.relu(img)
        img = self.max_pooling( img , np.array([2 , 2]) )

        img = self.convolve( img , self.fm3 )
        img = self.relu(img)
        img = self.max_pooling( img , np.array([4 , 4]) )

        img = img.reshape( np.size(img) , 1)

        self.Forward_Propogation_FC(img)
"""
    def newlearn(self):
        X # set of images  [ number of images  , chanels , x-size , y-size ]
        result # set of classes [  0-10 or something ]  matching X
        for i in range(data_index):
            
            X[i , : , : , :] 
"""
    


       
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
net.plot_cost( X , result , 8 , 0.1 , 0.01 , 1000)

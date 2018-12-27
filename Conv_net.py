import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import helper
from scipy import signal

class network:
    def __init__(self , fm , Element_Num_layers ): # element_num_layers like  [ 10 , 4 , 4 , 2 , 2] for network with two hidden layers of 4 each and two classes 
        
        # Initializations 
        self.weights = list()
        self.b = list()
        self.G_weights = list()
        self.G_b = list()
        self.error = list()
        
        self.Num_layers = len(Element_Num_layers)
        self.l = createlayers(Element_Num_layers)
        self.z = [None] * 3

        # random weights
        for index , layer in enumerate(self.l[0 : - 2] ) :  
            self.weights.append( np.random.rand( len(self.l[index + 1]) , len(layer)  ) )
        
        # random biases
        for i in Element_Num_layers[1 : -1 ] : 
            self.b.append(np.random.rand( i , 1)) 
        
        # setting inital feature maps
        self.fm0 = fm[0]
        self.fm1 = fm[1]
        self.fm2 = fm[2]

        # Initializing all gradients
        self.G_b = [None] * len(self.b)
        self.G_weights = [None] * len(self.weights)
        self.G_fm0 = np.zeros_like(self.fm0)
        self.G_fm1 = np.zeros_like(self.fm1)
        self.G_fm2 = np.zeros_like(self.fm2) 
    
    def Forward_Propogation_FC(self , x):
        self.l[0] = x
        self.l[0] = self.l[0].reshape( np.size(self.l[0]) , 1)
        for i in range(self.Num_layers - 2):
            self.z[i] = self.weights[i]  @  self.l[i]  +   self.b[i]
            self.l[i + 1] = self.relu(self.z[i])

        self.l[self.Num_layers - 1] = self.softmax(self.l[self.Num_layers - 2])

    def J_prime(self , X , result , data_index , lambd): 
        # J_prime is a wrapper for gradient_conv
        # Initializing all gradients of Fully Connected layer as zero
        for index , b in enumerate(self.b):
            self.G_b[index] = np.zeros_like(b)
            self.G_weights[index] = np.zeros_like(self.weights[index])
        
        for m in range(data_index):
            self.gradient_conv( X[m , : , : , :] , result[m , :]  , lambd)
        
        # gradient_conv adds all gradients so for loop to average the results.
        for index , b  in enumerate(self.b):
            self.G_b[index] = self.G_b[index] / data_index
            self.G_weights[index] = self.G_weights[index] / data_index
        
        self.G_fm0 = self.G_fm0 / data_index
        self.G_fm1 = self.G_fm1 / data_index
        self.G_fm2 = self.G_fm2 / data_index

    def gradient_conv(self , img , y , lambd):
        # Doing Fprward propogatin
        self.Forward_Propogation_conv(img)

        # Doing Back propogation

        # Fully connected errors and gradients of weights and biases
        self.gradient_FC( y , lambd) 
              
        # Calculating gradient for unrolled layer Im3
        self.G_Im3 = self.error[0].reshape( self.Im3.shape )
        
        self.G_Im3 = self.relu_p(self.G_Im3)

        # Calculating gradients for feature map fm2
        self.G_fm2 = helper.error2grad( self.G_Im3 , self.Im2)

        # Calculating gradient for layer Im2
        self.G_Im2 = helper.error_conv( self.G_Im3 , self.fm2 )

        # Calculating the gradient for pooling layer 2
        G_Im2_unpool = helper.unpool(self.G_Im2  ,  self.Im2_unpool ,  np.array([2 , 2]) )

        # apply relu_p
        G_Im2_unpool = self.relu_p(G_Im2_unpool)

        # Calculating gradients for feature map fm1
        self.G_fm1 = helper.error2grad(G_Im2_unpool , self.Im1)

        # Calculating gradient for layer Im1
        self.G_Im1 = helper.error_conv( G_Im2_unpool , self.fm1 ) # proble here , G_im1 changes sign
        
        # Calculating the gradient for pooling layer 1
        G_Im1_unpool = helper.unpool(self.G_Im1  , self.Im1_unpool, np.array([2 , 2]) )

        # apply relu_p
        G_Im1_unpool = self.relu_p(G_Im1_unpool)

        # Calculating gradients for feature map fm0
        self.G_fm0 = helper.error2grad( G_Im1_unpool , self.Im0)
        #

    def error_conv(self , G_Imk , fm_k_1 ): # fm_k_1 denotes fm(k-1)
        # function to calculate G_ImK_1
        
        
        G_Imk_1 = np.zeros(( fm_k_1.shape[0] , G_Imk.shape[1] + fm_k_1.shape[2] -1 , G_Imk.shape[2] + fm_k_1.shape[3] - 1))
        for a in range( 0 , G_Imk_1.shape[0]):
            for b in range( 0 , G_Imk_1.shape[1]):
                for c in range( 0 , G_Imk_1.shape[2]):
                    z_range = range( a * fm_k_1.shape[1] , (a+1)* fm_k_1.shape[1] )
                    # z is the image number in Imk
                    for z in z_range:
                        l = z % fm_k_1.shape[1]
                        #for i in range( max(b - fm_k_1.shape[2] , 0) , min(b+1 ,  G_Imk.shape[1]) ):
                        #    for j in range( max(c - fm_k_1.shape[3] , 0) , min(c + 1 ,  G_Imk.shape[1]) ):
                        #        G_Imk_1[ a , b , c] += G_Imk[z , i , j] * fm_k_1[a , l , b-i , c-j]
                        
                        G_Imk_1[a , b , c] = np.sum(signal.fftconvolve(G_Imk[z , : , :] , fm_k_1[a , l , : , :]))
                        #if (verify == G_Imk[a , : , :]):
                        #    raise ValueError( "Verify same")
        return G_Imk_1
        #

    def error2grad(self , G_Imk , Imk_1):
        G_fm_k_1 = np.zeros(( Imk_1.shape[0] , (int) (G_Imk.shape[0]/Imk_1.shape[0]) , Imk_1.shape[1] - G_Imk.shape[1] + 1 , Imk_1.shape[2] - G_Imk.shape[2] + 1 ))
        for m in range( G_fm_k_1.shape[0]):
            for n in range( G_fm_k_1.shape[1]):
                z = m * G_fm_k_1.shape[1] + n
                G_fm_k_1[ m , n  , : , :] = signal.fftconvolve(Imk_1[m , : , :] , G_Imk[z , : , :]  , mode = "valid" ) # dont ask for explanation it took me 3 hours to come up with this line of code.
        return G_fm_k_1

    def relu(self , t):
        t = t / 10
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

        self.error.insert( 0 , (self.l[-1] - y) / self.Num_layers ) 
        self.error.insert( 0 , self.error[0] * self.l[-1] * (1 - self.l[-1]) ) # L_sub(l) = self.l[-1]
        for i in range(2 , self.Num_layers ):
            self.error.insert(0 , np.transpose(self.weights[self.Num_layers - 1 - i]) @ (self.error[0] * self.relu_p( self.z[self.Num_layers - 1 - i])))

    def gradient_FC(self , y , lambd):
        # Forward pass assumed to be done in gradient_conv
        self.delta_FC(y)
        for i in range(self.Num_layers - 2):
            self.G_b[i] +=  self.error[i + 1] * self.relu_p(self.z[i]) 
            self.G_weights[i] += ( self.error[i + 1] * self.relu_p(self.z[i]) ) @ np.transpose(self.l[i]) + lambd * self.weights[i]

    def learn(self , X , result  , data_index , lambd , alpha , max_iter):
        
        cost_saver = [None] * max_iter
        for i in range(max_iter):
            # Calculate gradients
            self.J_prime( X , result , data_index , lambd)
            
            # Using Gradient descent
            for j in range(self.Num_layers - 2):
                self.b[j]  = self.b[j] - alpha * self.G_b[j]
                self.weights[j]  = self.weights[j] - alpha * self.G_weights[j]
            
            self.fm0 = self.fm0 - alpha * self.G_fm0
            self.fm1 = self.fm1 - alpha * self.G_fm1
            self.fm2 = self.fm2 - alpha * self.G_fm2
            
            # Saving Cost Vs Iterations for visualizations
            cost_saver[i] = self.cost(X , result , data_index)
            
        return cost_saver
    
    def cost(self , X , result , data_index):
        b = 0
        for i in range(data_index):
            self.Forward_Propogation_conv(X[i , : , : , :])
            a = self.l[self.Num_layers - 1].reshape( np.size(self.l[self.Num_layers - 1] ) , 1) - result[ i , :].reshape( np.size(result[i , :]) , 1)
            b += np.sum(a*a) / np.size(a)
        b = b / data_index
        return b
    
    def plot_cost(self , X , result , data_index , lambd , alpha , max_iter):
        axis_x = np.linspace( 0  , max_iter , 1 + max_iter)
        axis_y = self.learn(X ,result , data_index , lambd , alpha , max_iter)
        axis_y.append(axis_y[-1])
        plt.plot(axis_x , axis_y )
        plt.xlabel( "Iterations" )
        plt.ylabel( "Cost" )
        plt.title( "Cost Vs Iterations" )
        plt.show()
    
    def convolve(self , img , feature_map): # features should be odd x odd to keep image sizes even for pooling
        Num_channel = np.size( img , axis = 0)
        Num_features = np.size(feature_map , axis = 1)
        if ( Num_channel != np.size(feature_map , axis = 0) ):
            raise ValueError("Incorrect feature map")
        final = np.zeros(( Num_channel * Num_features , np.size(img , axis = 1) - np.size(feature_map , axis = 2)  + 1 , np.size(img , axis = 2) - np.size(feature_map , axis = 3)  + 1 ))
        # final image is a litle bit smaller but has many more channels. 
        for i in range(Num_channel): # i is the depth channel
            initial = img[ i , : , : ]
            for j in range( Num_features ): # j is the number of feature matrices per channel\\\
                final[ j + Num_features * i , : , : ] = signal.fftconvolve( initial , feature_map[i , j , : , :]  , mode = 'valid')
        return final
        
    def conv(self , img  , feature):
        result  = np.zeros((  np.size(img , axis = 0) - np.size(feature , axis = 0)  + 1 , np.size(img , axis = 1) - np.size(feature , axis = 1)  + 1 ))
        for i in range(np.size(result  , axis = 0)):
            for j in range(np.size(result  , axis = 1)):
                result[ i , j ] = np.sum (img[ i : i + np.size(feature  , axis = 0)  , j : j + np.size(feature  , axis = 1)  ] * feature)
        return result
    
    def max_pooling(self  , img  , pool_map):
        Num_channel = np.size(img , axis = 0)
        final = np.zeros( (np.size( img  , axis = 0)  , (int)(np.size( img  , axis = 1) / pool_map[0]) , (int) (np.size( img  , axis = 2)  / pool_map[1]) ))
        for i in range(Num_channel):
            initial = img[i , : , :]
            final[i , : , :] = helper.max_pool( initial  , pool_map ) 
        return final

    def max_pool(self , img  , pool_map):
        # pool map is numpy array of pooling size like [4 , 4] for 4 x 4
        final = np.zeros(( (int) (np.size( img  , axis=0) / pool_map[0]), (int) (np.size( img  , axis=1) / pool_map[1])) )
        for i in range( 0  , np.size(img  , axis=0) - pool_map[0], pool_map[0] ):
            for j in range( 0  , np.size(img  , axis=1) - pool_map[1], pool_map[1] ):
                final[ (int) (i / pool_map[0]) , (int) (j / pool_map[1]) ]  = np.max( img[ i : i + pool_map[0] ,  j : j + pool_map[1]] )
        return final

    def unpool(self , G_Imk , Imk_unpool , pool_map):
        final = np.zeros_like( Imk_unpool)

        for z in range(Imk_unpool.shape[0]):
            for p in range( 0 ,Imk_unpool.shape[1] - pool_map[0] + 1, pool_map[0]):
                for q in range(0 , Imk_unpool.shape[2] - pool_map[1] + 1, pool_map[1]):
                    (i , j) = np.unravel_index(np.argmax( Imk_unpool[z , p:p + pool_map[0] , q:q+pool_map[1]] , axis=None), pool_map)
                    final[z , i , j] = G_Imk[z , (int)(p / pool_map[0]) , (int)(q / pool_map[1])]
        
        return final

    def Forward_Propogation_conv(self , img ):
        # 3 convololutional layers
        self.Im0 = img

        img = self.convolve( img , self.fm0 )
        img = self.relu(img)
        
        self.Im1_unpool = img
        
        img = self.max_pooling( img , [2 , 2] )

        self.Im1 = img
        self.G_Im1 = np.zeros_like(self.Im1)

        img = self.convolve( img , self.fm1 )
        img = self.relu(img)
        
        self.Im2_unpool = img

        img = self.max_pooling( img , [2 , 2] )

        self.Im2 = img
        self.G_Im2 = np.zeros_like(self.Im2)

        img = self.convolve( img , self.fm2 )
        img = self.relu(img)

        self.Im3 = img
        self.G_Im3 = np.zeros_like(self.Im3)

        img = img.reshape( np.size(img) , 1)

        self.Forward_Propogation_FC(img)

def createlayers(layer_elements):
    a = list()
    for i in layer_elements:
        a.append( np.zeros( (i , 1) ) )
    return a

fm0 = np.random.rand( 1 , 2 , 5 , 5)
fm1 = np.random.rand( 2 , 2 , 3 , 3)
fm2 = np.random.rand( 4 , 3 , 3 , 3)

net = network( [fm0 , fm1 , fm2] , [192 , 40 , 40 , 10 , 10])

X = np.random.rand( 40 , 1 , 32 , 32)
result  = np.zeros( (40, 10) )
for i in range(40):
    j = np.random.randint(0 , high = 10)
    result[i , j] = 1

net.plot_cost( X , result , 10 , 1 , 0.001 , 10)

import numpy as np
cimport numpy as np

def max_pool( img  , pool_map):
    # pool map is numpy array of pooling size like [4 , 4] for 4 x 4
    final = np.zeros( np.size( img  , axis=0) / pool_map[0], np.size( img  , axis=1) / pool_map[1]) 
    cdef int i = 0
    cdef int j = 0
    for i in range( 0  , np.size(img  , axis=0) - pool_map[0], pool_map[0] ):
        for j in range( 0  , np.size(img  , axis=1) - pool_map[1], pool_map[1] ):
            final[ i / pool_map[0] , j / pool_map[1] ]  = np.max( img[ i : i + pool_map[0] ,  j : j + pool_map[1]] )
    return final

def conv( img  , feature):
    result  = np.zeros(  np.size(img , axis = 0) - np.size(feature , axis = 0)  + 1 , np.size(img , axis = 1) - np.size(feature , axis = 1)  + 1 )
    cdef int i = 0
    cdef int j = 0
    for i in range(np.size(result  , axis = 0)):
        for j in range(np.size(result  , axis = 1)):
            result[ i , j ] = np.sum (img[ i : i + np.size(feature  , axis = 0)  , j : j + np.size(feature  , axis = 1)  ] * feature)
    return result

def convolve(np.ndarray img , np.ndarray feature_map): # features should be odd x odd to keep image sizes even for pooling
    cdef int Num_channel = np.size( img , axis = 0)
    cdef int Num_features = np.size(feature_map , axis = 1)
    final_size = ( Num_channel*Num_features , img.shape[1]-feature_map.shape[2]+1 , img.shape[2]-feature_map.shape[3]+1)
    final = np.zeros( final_size )
    # final image is a litle bit smaller but has many more channels. 
    cdef int i = 0
    cdef int j = 0
    for i in range(Num_channel): # i is the depth channel
        initial = img[ i , : , : ]
        for j in range( Num_features ): # j is the number of feature matrices per channel\\\
            final[ j + Num_features * i , : , : ] = conv( initial , feature_map[i , j , : , :] )
    return final

def error2grad( G_Imk , Imk_1):
    G_fm_k_1 = np.zeros( Imk_1.shape[0] , G_Imk.shape[0]/Imk_1.shape[0] , Imk_1[1] - G_Imk[1] + 1 , Imk_1[2] - G_Imk[2] + 1 )
    cdef int m = 0
    cdef int n = 0
    cdef int z = 0
    for m in range( G_fm_k_1.shape[0]):
        for n in range( G_fm_k_1.shape[1]):
            z = m * G_fm_k_1.shape[1] + n
            G_fm_k_1[ m , n  , : , :] = conv(Imk_1[m , : , :] , G_Imk[z , : , :] ) # dont ask for explanation it took me 3 hours to come up with this line of code.
    return G_fm_k_1

def error_conv( G_Imk , fm_k_1 ): # fm_k_1 denotes fm(k-1)
    # function to calculate G_ImK_1
    G_Imk_1 = np.zeros( fm_k_1.shape[0] , G_Imk.shape[1] + fm_k_1.shape[2] -1 , G_Imk.shape[2] + fm_k_1.shape[3] - 1)
    cdef int a = 0
    cdef int b = 0
    cdef int c = 0
    cdef int i = 0
    cdef int j = 0
    cdef int z = 0
    cdef int l = 0
    for a in range( 0 , G_Imk[0]):
        for b in range( 0 , G_Imk[1]):
            for c in range( 0 , G_Imk[2]):
                # z is the image number in Imk
                for z in range( a * fm_k_1.shape[1] , (a+1)* fm_k_1.shape[1] ):
                    l = z % fm_k_1.shape[1]
                    for i in range(b , b - fm_k_1.shape[2] , -1 ):
                        for j in range(c , c - fm_k_1.shape[3] , -1 ):
                            G_Imk_1[ a , b , c] += G_Imk[z , i , j] * fm_k_1[a , l , b - i , c - j]
    return G_Imk_1

def unpool(G_Imk , Imk_unpool , pool_map):
    final = np.zeros_like( Imk_unpool)
    cdef int z = 0
    cdef int p = 0
    cdef int q = 0
    cdef int i = 0
    cdef int j = 0
    for z in range(Imk_unpool.shape[0]):
        for p in range( 0 ,Imk_unpool.shape[1] - pool_map[0] + 1, pool_map[0]):
            for q in range(0 , Imk_unpool.shape[2] - pool_map[1] + 1, pool_map[1]):
                (i , j) = np.unravel_index(np.argmax( Imk_unpool[z , p:p + pool_map[0] , q:q+pool_map[1]] , axis=None), pool_map)
                final[z , i , j] = G_Imk[z , p / pool_map[0] , q / pool_map[1]] 
    return final
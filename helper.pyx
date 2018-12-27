import numpy as np
cimport numpy as np
from scipy import signal

Dtype = np.float64
ctypedef np.int_t DTYPE_t

def error_conv(np.ndarray G_Imk , np.ndarray fm_k_1 ): # fm_k_1 denotes fm(k-1)
    # function to calculate G_ImK_1        
    cdef np.ndarray G_Imk_1 = np.zeros(( fm_k_1.shape[0] , G_Imk.shape[1] + fm_k_1.shape[2] -1 , G_Imk.shape[2] + fm_k_1.shape[3] - 1) , dtype = Dtype)
    assert G_Imk_1.dtype == Dtype and fm_k_1.dtype == Dtype
    cdef int a  = 0
    cdef int b  = 0
    cdef int c  = 0
    cdef int z  = 0
    cdef int l  = 0
    for a in range( 0 , G_Imk_1.shape[0]):
        for b in range( 0 , G_Imk_1.shape[1]):
            for c in range( 0 , G_Imk_1.shape[2]):
                for z in range( a * fm_k_1.shape[1] , (a+1)* fm_k_1.shape[1] ):
                    l = z % fm_k_1.shape[1]
                    G_Imk_1[a , b , c] = np.sum(signal.fftconvolve(G_Imk[z , : , :] , fm_k_1[a , l , : , :]))
    return G_Imk_1

def error2grad(np.ndarray G_Imk , np.ndarray Imk_1):
    cdef np.ndarray G_fm_k_1 = np.zeros(( Imk_1.shape[0] , (int) (G_Imk.shape[0]/Imk_1.shape[0]) , Imk_1.shape[1] - G_Imk.shape[1] + 1 , Imk_1.shape[2] - G_Imk.shape[2] + 1 ), dtype = Dtype)
    cdef int m = 0
    cdef int n = 0
    cdef int z = 0
    for m in range( G_fm_k_1.shape[0]):
        for n in range( G_fm_k_1.shape[1]):
            z = m * G_fm_k_1.shape[1] + n
            G_fm_k_1[ m , n  , : , :] = signal.fftconvolve(Imk_1[m , : , :] , G_Imk[z , : , :] , mode = "valid") # dont ask for explanation it took me 3 hours to come up with this line of code.
    return G_fm_k_1

def max_pool(np.ndarray img  , list pool_map):
    # pool map is numpy array of pooling size like [4 , 4] for 4 x 4
    cdef np.ndarray final = np.zeros(( (int) (np.size( img  , axis=0) / pool_map[0]), (int) (np.size( img  , axis=1) / pool_map[1]))  , dtype = Dtype)
    cdef int i = 0
    cdef int j = 0
    for i in range( 0  , np.size(img  , axis=0) - pool_map[0], pool_map[0] ):
        for j in range( 0  , np.size(img  , axis=1) - pool_map[1], pool_map[1] ):
            final[ (int) (i / pool_map[0]) , (int) (j / pool_map[1]) ]  = np.max( img[ i : i + pool_map[0] ,  j : j + pool_map[1]] )
    return final

def unpool(np.ndarray G_Imk , np.ndarray Imk_unpool , np.ndarray pool_map):
    cdef np.ndarray final = np.zeros_like( Imk_unpool , dtype = Dtype)
    cdef int i = 0
    cdef int j = 0
    cdef int p = 0
    cdef int q = 0
    cdef int z = 0
    for z in range(Imk_unpool.shape[0]):
        for p in range( 0 ,Imk_unpool.shape[1] - pool_map[0] + 1, pool_map[0]):
            for q in range(0 , Imk_unpool.shape[2] - pool_map[1] + 1, pool_map[1]):
                (i , j) = np.unravel_index(np.argmax( Imk_unpool[z , p:p + pool_map[0] , q:q+pool_map[1]] , axis=None), pool_map)
                final[z , i , j] = G_Imk[z , (int)(p / pool_map[0]) , (int)(q / pool_map[1])]    
    return final
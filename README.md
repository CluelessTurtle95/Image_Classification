# Image_Classification

#### Note : 
###### BackPropogation hasn't been implemented completely for all layers yet. [First two convolution layers left]
###### Image Preprocesssing not done yet

Convolutional Neural Network from scratch using only Numpy in python. 

Image Classification Program 
Using MNIST like database
Image size is 32 x 32 pixels 

## Steps : 

### 1 Image processing
       Convert Image to correct size
       Make Image Grayscale
### 2 PreProcessing
       Make Mean of each Image 0
       Normalize by standard deviation

### 3 Neural Network Tranining
####       3.1 Convolution
               18 Kernels in total across 3 convolutional + Relu Layers
               2 Pooling Layers to reduce features for fully connected part
####       3.2 Fully Connected
               192 features in first layer
               40 features in both hidden layers
               10 classes  

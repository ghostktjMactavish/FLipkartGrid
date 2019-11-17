Only the keras library has been used apart from numpy.

data_generate() function is used to to supply appropriate number of images so that memory in the GPU does not overflow.
Model was trained for 40 epochs on GTX 1050 Ti GPU
 
We have made a keras model with the following layers:

1) 2dconvolutional layer 32 filter, 3x3 kernel , stride= (1,1)
2) MaxPooling layer pool_size=(3,3), strides=(2,2)
3) 2dconvolutional layer 32 filter, 3x3 kernel , stride = (1,1)
4) MaxPooling layer pool_size=(3,3), strides=(2,2)
5) 2 2dconvolutional layers 64 filters, 3x3 kernel, stride = (1,1)
6) 2dconvolutional layer 32 filters, 3x3 kernel, stride =(1,1)
7) MaxPooling layer pool_size=(3,3), strides=(2,2)
8) Flat layer
9) Added 0.2 dropout layer
9) Dense layer 1024 neurons sigmoid activation
10) Output layer 4 neurons sigmoid activation

Loss=Mean Squared Error 
Optimizer=Adam Optimizer with learning rate = 0.000001

Usage instructions:

Store training images in the same directory in a folder named training_images
Store training.csv and test.csv in same directory as try.py
Run try.py with required files in appropriate directories to train the model.
Weights calculated are stored in "my_model_weights1.h5"
To generate result.csv run test.py
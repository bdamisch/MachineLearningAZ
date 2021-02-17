# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
tf.__version__
import sys
from PIL import Image
sys.modules['Image'] = Image
print(Image.__file__)

# Part 1 - Data Preprocessing
#train_data_dir = r'~\dataset\training_set'
train_data_dir = r'C:\Users\Ben\Desktop\Udemy_ML_Py_R\ML_python\Machine Learning A-Z (Codes and Datasets)\Part 8 - Deep Learning\Section 40 - Convolutional Neural Networks (CNN)\Python\dataset\training_set'
test_data_dir = r'C:\Users\Ben\Desktop\Udemy_ML_Py_R\ML_python\Machine Learning A-Z (Codes and Datasets)\Part 8 - Deep Learning\Section 40 - Convolutional Neural Networks (CNN)\Python\dataset\test_set'


# Preprocessing the Training set
# perform transformations on the images to prevent overfitting
# overfitting =very accurate with training set, much less accurate on test set
# apply feature scaling by dividing pixels by 255
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
# reduce the size to make it faster when feeding it into the neural network
training_set = train_datagen.flow_from_directory(train_data_dir,
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(test_data_dir,
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
# filters = number of feature detectors
# kernel_size = kernel_size x kernel_size feature detector (3x3)
# relu is rectifier
# 64x64 with 3 colors of pixels
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Step 2 - Pooling
# pool size is the size of the pool frame 2x2
# stride is the distance we move the pool frame
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer

# remove input shape parameter, that is only necessary when entering the CNN
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
#use this as the output layer for the ANN
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
# use rectifier activation fn
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
# at the output layer, use the sigmoid to create probabilties, we only need 1 neuron to encode cat or dog
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

# Part 4 - Making a single prediction

prediction_img = r'C:\Users\Ben\Desktop\Udemy_ML_Py_R\ML_python\Machine Learning A-Z (Codes and Datasets)\Part 8 - Deep Learning\Section 40 - Convolutional Neural Networks (CNN)\Python\dataset\single_prediction\Luna.jpg'

import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img(prediction_img, target_size = (64, 64))
# predict method expects an array instead of an image
test_image = image.img_to_array(test_image)
# batch #1 has 32 images, batch #2 has 32 images etc. 
# if we are going to deploy model on a single image it must be in a batch 
# for the predict method of the CNN model recognize the batch as an extra dimension
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
# just allows us to know which indices corresponds to which classes
training_set.class_indices
# first index is the batch
# second index is the element of the batch
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)
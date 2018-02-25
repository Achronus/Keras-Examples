# Convolutional Neural Network

#-------------------------
# Importing the libraries
#-------------------------
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

#-------------------------
# Custom Parameters
#-------------------------
img_width, img_height = 64, 64
input_shape = (img_width, img_height, 3)
dropout_rate = 0.2
batch_size = 32
epochs = 100

#-------------------------
# Importing the dataset
#-------------------------
# Initalising the training
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# Initalising the testing
test_datagen = ImageDataGenerator(rescale=1./255)

# Create the training set and resize the images
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(img_width, img_height),
                                                 batch_size=batch_size,
                                                 class_mode='binary')

# Create the test set and resize the images
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(img_width, img_height),
                                            batch_size=batch_size,
                                            class_mode='binary')

#-------------------------
# Create the CNN
#-------------------------
def CNN(dropout_rate, input_shape):
    '''
    A Convolutional Neural Network that takes a dropout_rate and input_shape as custom variables.
    
    dropout_rate - Used to assign the amount of neurons to drop when training, this takes a integer between 0 and 1. For example: 0.1 = 10%.
    input_shape - The size of the image dimensions and what type of image it is: (img_width, img_height, image_type). img_width & img_height = pixel dimensions; image_type = 1 (black & white images) or 3 (coloured images).
    '''
    # Initalise the CNN
    model = Sequential()
    
    # Convolution & Pooling
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Convolution & Pooling
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Convolution & Pooling
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Flattening
    model.add(Flatten())
    
    # Full Connection
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    
    # Output layer
    model.add(Dense(units=1, activation='sigmoid'))
    
    # Compiling the CNN
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Set model variable name
cnn = CNN(dropout_rate, input_shape)

#-------------------------
# Train the model
#-------------------------
cnn.fit_generator(training_set,
                  steps_per_epoch=8000,
                  epochs=epochs,
                  validation_data=test_set,
                  validation_steps=2000)

#-------------------------
# Making a prediction
#-------------------------
# Identify test image
test_image = image.load_img(path='dataset/single_prediction/cat_or_dog_1.jpg', target_size=(img_width, img_height))
# Convert image into a numpy array to have same format as input_shape
test_image = image.img_to_array(test_image)
# Add additional dimension as convolution layer has 4 dimensions (this relates to the batch dimension)
test_image = np.expand_dims(test_image, axis=0)

# Predict the result
result = cnn.predict(test_image)
# Outputs mapping for image category - this is taken from the sub directory names within the .flow_from_directory(directory)
indices = training_set.class_indices
# Identify the prediction value
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
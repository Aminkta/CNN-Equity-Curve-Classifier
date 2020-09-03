# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 10,
                         validation_data = test_set,
                         nb_val_samples = 2000)

classifier.save('classifier.h5') 




# Saving the CNN model to load in another file

'''
# In this code below we test for a an equity curve random.jpg

test_image = image.load_img('random.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] >= 0.5:
    prediction = 'YES'
else:
    prediction = 'NO'

print(prediction)
'''

# importing os module 

  
'''

# Function to rename multiple files 
#def main(): 
i = 0
path = r"C:\Users\amink\Desktop\AI_in_Finance\pair_trading_cnn_project\new_cnn_keras_algorithm\latest_code\test_loop_1"
os.chdir(path)
for filename in os.listdir(r"C:\Users\amink\Desktop\AI_in_Finance\pair_trading_cnn_project\new_cnn_keras_algorithm\latest_code\test_loop_1\\"): 
    test_image = image.load_img(filename, target_size = (64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    training_set.class_indices
    if result[0][0] >= 0.5:
        prediction = 'YES'
    else:
        prediction = 'NO'
    dst = prediction + str(i) + ".png"
    src =r"C:\Users\amink\Desktop\AI_in_Finance\pair_trading_cnn_project\new_cnn_keras_algorithm\latest_code\test_loop_1\\"+ filename 
    dst =r"C:\Users\amink\Desktop\AI_in_Finance\pair_trading_cnn_project\new_cnn_keras_algorithm\latest_code\test_loop_1\\"+ dst       
    # rename() function will 
    # rename all the files
    if not os.path.exists(dst): # we might need to delete this and unindent the two lines below
        os.rename(src, dst) 
        i += 1

'''
import numpy as np
from keras.preprocessing import image
import os 
from keras.models import load_model

path = r"C:\Users\amink\Desktop\AI_in_Finance\pair_trading_cnn_project\new_cnn_keras_algorithm\latest_code"
os.chdir(path)
classifier = load_model('classifier.h5')
# Appending the YES testing dataset into yes and no folders in the cnn_predictions folder

i = 0
path = r"C:\Users\amink\Desktop\AI_in_Finance\pair_trading_cnn_project\new_cnn_keras_algorithm\latest_code\dataset\test_set\YES"
os.chdir(path)
for filename in os.listdir(r"C:\Users\amink\Desktop\AI_in_Finance\pair_trading_cnn_project\new_cnn_keras_algorithm\latest_code\dataset\test_set\YES\\"): 
    test_image = image.load_img(filename, target_size = (64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    training_set.class_indices
    if result[0][0] >= 0.5:
        prediction = 'YES'
        dst = prediction + str(i) + ".png"
        dst =r"C:\Users\amink\Desktop\AI_in_Finance\pair_trading_cnn_project\new_cnn_keras_algorithm\latest_code\cnn_predictions\yes\\"+ dst
    else:
        prediction = 'NO'
        dst = prediction + str(i) + ".png"
        dst =r"C:\Users\amink\Desktop\AI_in_Finance\pair_trading_cnn_project\new_cnn_keras_algorithm\latest_code\cnn_predictions\no\\"+ dst
    
    src =r"C:\Users\amink\Desktop\AI_in_Finance\pair_trading_cnn_project\new_cnn_keras_algorithm\latest_code\dataset\test_set\YES\\"+ filename      
    # rename() function will 
    # rename all the files
    if not os.path.exists(dst): # we might need to delete this and unindent the two lines below
        os.rename(src, dst) 
        i += 1


# Appending the NO testing dataset into yes and no folders in the cnn_predictions folder

path = r"C:\Users\amink\Desktop\AI_in_Finance\pair_trading_cnn_project\new_cnn_keras_algorithm\latest_code\dataset\test_set\NO"
os.chdir(path)
for filename in os.listdir(r"C:\Users\amink\Desktop\AI_in_Finance\pair_trading_cnn_project\new_cnn_keras_algorithm\latest_code\dataset\test_set\NO\\"): 
    test_image = image.load_img(filename, target_size = (64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    training_set.class_indices
    if result[0][0] >= 0.5:
        prediction = 'YES'
        dst = prediction + str(i) + ".png"
        dst =r"C:\Users\amink\Desktop\AI_in_Finance\pair_trading_cnn_project\new_cnn_keras_algorithm\latest_code\cnn_predictions\yes\\"+ dst
    else:
        prediction = 'NO'
        dst = prediction + str(i) + ".png"
        dst =r"C:\Users\amink\Desktop\AI_in_Finance\pair_trading_cnn_project\new_cnn_keras_algorithm\latest_code\cnn_predictions\no\\"+ dst
    
    src =r"C:\Users\amink\Desktop\AI_in_Finance\pair_trading_cnn_project\new_cnn_keras_algorithm\latest_code\dataset\test_set\NO\\"+ filename      
    # rename() function will 
    # rename all the files
    if not os.path.exists(dst): # we might need to delete this and unindent the two lines below
        os.rename(src, dst) 
        i += 1
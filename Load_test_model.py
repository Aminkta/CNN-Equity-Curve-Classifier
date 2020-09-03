# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 11:37:03 2019

@author: amink
"""

import numpy as np
from keras.preprocessing import image
import os 
from keras.models import load_model

### VERY IMPORTANT --->> Change the path to where the script is on your computer !
path1 = r"C:\Users\amink\Desktop\CNN pair trading project"
###


os.chdir(path1)
classifier = load_model('classifier.h5')
# Appending the YES testing dataset into yes and no folders in the cnn_predictions folder

i = 0
#path = r"C:\Users\amink\Desktop\AI_in_Finance\pair_trading_cnn_project\new_cnn_keras_algorithm\latest_code\dataset\test_set\YES"
path =path1 + r"\dataset\test_set\YES"
os.chdir(path)
for filename in os.listdir(path1 + r"\dataset\test_set\YES\\"): 
    test_image = image.load_img(filename, target_size = (64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
   # training_set.class_indices
    if result[0][0] >= 0.5:
        prediction = 'YES'
        dst = prediction + str(i) + ".png"
        dst =path1 + r"\cnn_predictions\yes\\"+ dst
    else:
        prediction = 'NO'
        dst = prediction + str(i) + ".png"
        dst =path1 + r"\cnn_predictions\no\\"+ dst
    
    src =path1 + r"\dataset\test_set\YES\\"+ filename      
    # rename() function will 
    # rename all the files
    if not os.path.exists(dst):
        os.rename(src, dst) 
        i += 1


# Appending the NO testing dataset into yes and no folders in the cnn_predictions folder

path = path1 + r"\dataset\test_set\NO"
os.chdir(path)
for filename in os.listdir(path1 + r"\dataset\test_set\NO\\"): 
    test_image = image.load_img(filename, target_size = (64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
  #  training_set.class_indices
    if result[0][0] >= 0.5:
        prediction = 'YES'
        dst = prediction + str(i) + ".png"
        dst = path1 + r"\cnn_predictions\yes\\"+ dst
    else:
        prediction = 'NO'
        dst = prediction + str(i) + ".png"
        dst = path1 + r"\cnn_predictions\no\\"+ dst
    
    src = path1 + r"\dataset\test_set\NO\\"+ filename      
    # rename() function will 
    # rename all the files
    if not os.path.exists(dst): 
        os.rename(src, dst) 
        i += 1
import cv2                                                   # for capturing videos
import math                                                  # for mathematical operations
import matplotlib.pyplot as plt                              # for plotting the images
import pandas as pd
from keras.preprocessing import image                        # for preprocessing the images
import numpy as np                                           # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize                         # for resizing images
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import np_utils
from skimage.transform import resize
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout

model=load_model("modelTJ.h5")
model.load_weights("modelweights.h5")
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
print("Loaded model from disk")


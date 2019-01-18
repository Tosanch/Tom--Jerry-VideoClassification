import cv2                                                   # for capturing videos
import math                                                  # for mathematical operations
import matplotlib.pyplot as plt                              # for plotting the images
import pandas as pd
from keras.callbacks import TensorBoard
from keras.preprocessing import image                        # for preprocessing the images
import numpy as np                                           # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize                         # for resizing images
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import np_utils
from skimage.transform import resize
from keras.models import Sequential, load_model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout



count = 0
videoFile = "Tom and jerry.mp4"
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5) #frame +
x=1
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename ="frame%d.jpg" % count;count+=1
        cv2.imwrite(filename, frame)
cap.release()
print ("Done!")

img = plt.imread('frame0.jpg')        # reading image using its name
plt.imshow(img)

data = pd.read_csv('mapping.csv')     # reading the csv file
data.head()                           # printing first five rows of the file

X = [ ]                               # creating an empty array
for img_name in data.Image_ID:
    img = plt.imread('' + img_name)
    X.append(img)                     # storing each image in array X
X = np.array(X)                       # converting list to array

y = data.Class
dummy_y = np_utils.to_categorical(y)  # encoding class

image = []
for i in range(0,X.shape[0]):
    a = resize(X[i], preserve_range=True, output_shape=(224,224)).astype(int)                        # reshaping to 224*224*3(R,G,B)
    image.append(a)
X = np.array(image)

X = preprocess_input(X, mode='tf')     # preprocessing the input data

X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.3, random_state=42)    # preparing the validation set

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))                 # include_top=False to remove the top layer

X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)
X_train.shape, X_valid.shape


X_train = X_train.reshape(208, 7*7*512)      # converting to 1-D
X_valid = X_valid.reshape(90, 7*7*512)

train = X_train/X_train.max()                # centering the data
X_valid = X_valid/X_train.max()

model = Sequential()                                                                              # i. Building the model
model.add(InputLayer((7*7*512,)))                        # input layer
model.add(Dense(units=1024, activation='relu'))       # hidden layer
model.add(Dense(3, activation='softmax'))                # output layer

model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])            # ii. Compiling the model
graph = TensorBoard(log_dir='TensorGraphs/graph1')

model.fit(train, y_train, epochs  =100, validation_data=(X_valid, y_valid),callbacks=[graph])

model.save_weights("modelweights.h5")
model.save("modelTJ.h5")
print("Saved model to disk")


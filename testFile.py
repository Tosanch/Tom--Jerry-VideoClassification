import math  # for mathematical operations

import cv2  # for capturing videos
import matplotlib.pyplot as plt  # for plotting the images
import numpy as np
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, InputLayer
from keras.models import Sequential
from skimage.transform import resize

count = 0

base_model = VGG16(weights='imagenet', include_top=False,
                   input_shape=(224, 224, 3))  # include_top=False to remove the top layer

model = Sequential()  # i. Building the model
model.add(InputLayer((7 * 7 * 512,)))  # input layer
model.add(Dense(units=1024, activation='relu'))  # hidden layer
model.add(Dense(3, activation='softmax'))  # output layer

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # ii. Compiling the model

model.load_weights("modelweights.h5")

count = 0  # Testing and Calculating the screen timing
videoFile = "Tom and Jerry 3.mp4"
# cap = cv2.VideoCapture(videoFile)
# frameRate = cap.get(5)  # frame rate
# x = 1
# while (cap.isOpened()):
#     frameId = cap.get(1)  # current frame number
#     ret, frame = cap.read()
#     if (ret != True):
#         break
#     if (frameId % math.floor(frameRate) == 0):
#         filename = "test%d.jpg" % count;
#         count += 1
#         cv2.imwrite(filename, frame)
# cap.release()
# print("Done!")

test = pd.read_csv('test.csv')

test_images = []
for img_name in test.Image_ID:
    img = plt.imread('' + img_name)
    test_images.append(img)
test_img = np.array(test_images)

test_image = []
for i in range(0, test_img.shape[0]):
    a = resize(test_img[i], preserve_range=True, output_shape=(224, 224)).astype(int)
    test_image.append(a)
test_image = np.array(test_image)

# preprocessing the images
test_image = preprocess_input(test_image, mode='tf')

# extracting features from the images using pretrained model
test_image = base_model.predict(test_image)

# converting the images to 1-D form
test_image = test_image.reshape(186, 7 * 7 * 512)

# zero centered images
test_image = test_image / test_image.max()

predictions = model.predict_classes(test_image)
print(predictions)
for i in range(160, 171):

    plt.imshow(test_images[i])
    pred = predictions[i]
    title = ""
    if (pred == 1):
        title = "This is Jerry"
    elif (pred == 2):
        title = "This is Tom"
    else:
        title = "none of them appear"

    plt.title(title)
    plt.show()

# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")


print("The screen time of JERRY is", predictions[predictions == 1].shape[0], "seconds")
print("The screen time of TOM is", predictions[predictions == 2].shape[0], "seconds")



import tensorflow
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalMaxPooling2D
import numpy as np
from numpy.linalg import norm
import pickle
import cv2
from sklearn.neighbors import NearestNeighbors




feature_list =np.array(pickle.load(open("C:/Users/TRINADH/Desktop/recommendation/featurevectors.pkl","rb")))
filename = pickle.load(open("C:/Users/TRINADH/Desktop/recommendation/filenames.pkl","rb"))




model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([model,MaxPooling2D()])
model.summary()



img = cv2.imread("1636.jpg")
img = cv2.resize(img,(224,224))
img = np.array(img)
expand_img = np.expand_dims(img, axis =0)
pre_img = preprocess_input(expand_img)
result = model.predict(pre_img).flatten()
normalized = result/norm(result)



neigbhros = NearestNeighbors(n_neighbors=5, algorithm="brute",metric="euclidean")

neigbhros.fit(feature_list)

distance, indices = neigbhros.kneighbors(normalized)

for file in indices[0][0:5]:
    imgname = cv2.imread(filename[file])
    cv2.imshow("frame",cv2.resize(imgname,(640,480)))
    cv2.waitKey(0)




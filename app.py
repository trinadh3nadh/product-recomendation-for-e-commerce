

import streamlit as st
from PIL import Image
import tensorflow
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
import numpy as np
from numpy.linalg import norm
import pickle
import cv2
from sklearn.neighbors import NearestNeighbors
import os



feature_list =np.array(pickle.load(open("C:/Users/TRINADH/Desktop/recommendation/featurevectors.pkl","rb")))
filename = pickle.load(open("C:/Users/TRINADH/Desktop/recommendation/filenames.pkl","rb"))


model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([model,GlobalMaxPooling2D()])
model.summary()

st.title("Man & women Fashion Recommendation System")

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('upload',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def extract_feature(img_path,model):
    img = cv2.imread(img_path)
    img = cv2.resize(img,(224,224))
    img = np.array(img)
    expand_img = np.expand_dims(img,axis=0)
    pre_img = preprocess_input(expand_img)
    result = model.predict(pre_img).flatten()
    normalization = result/norm(result)
    return normalization

def recomended(features, feature_list):
    neigbhros = NearestNeighbors(n_neighbors=6, algorithm="brute",metric="euclidean")
    neigbhros.fit(feature_list)
    
    distances, indices = neigbhros.kneighbors([features])
    return indices


uploaded_file = st.file_uploader("Choose an image")
print(uploaded_file)

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        resize_img = display_image.resize(200,200)
        
        st.image(resize_img)
        #feature extraction
        features = extract_feature(os.path.join("upload",uploaded_file.name),model)
        #recomendation
        indices = recomended(features,feature_list)
        
        #display 
        col1,col2,col3,col4,col5 = st.columns(5)
        
        with col1:
            st.image(filename[indices[0][0]])
        with col2:
            st.image(filename[indices[0][1]])
        with col3:
            st.image(filename[indices[0][2]])
        with col4:
            st.image(filename[indices[0][3]])
        with col5:
            st.image(filename[indices[0][4]])
    else:
        print("some problem in uploading the image")



































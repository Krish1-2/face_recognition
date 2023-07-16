import os 
import tensorflow as tf
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
import pickle

loaded_model = tf.keras.models.load_model("face_recognition_model.keras")

# Load the label encoder
with open("label_encoder.pkl", 'rb') as file:
    label_encoder = pickle.load(file)

def recognize_faces(image_paths):
    for image_path in image_paths:
        # Load and process the image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (256, 256))
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_normalized = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # Perform face recognition by passing the image through the loaded model
        predictions = loaded_model.predict(np.expand_dims(img_normalized, axis=0))
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
        print(predicted_class)    



video = cv2.VideoCapture(0)
count = 0
image_paths = []
while count <= 1:
    ret, frame = video.read()
    count += 1
    cv2.imshow('windowsFrame', frame)
    image_path = 'doing/image' + str(count) + '.png'
    cv2.imwrite(image_path, frame)
    image_paths.append(image_path)

recognize_faces(image_paths)


        # Perform face recognition by passing the image through the loaded model
        # features = loaded_model.predict(np.expand_dims(img_normalized, axis=0))
        # print(features)
        #  Perform label decoding to obtain the recognized person
        # recognized_label = label_encoder.inverse_transform(np.argmax(features, axis=1))
        # print(recognized_label)
        # if features<=0.5:
        #     cv2.putText(image, 'ami', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        # elif features>0.5:
        #     cv2.putText(image, 'krish', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        # cv2.imshow('Face Recognition', image)
        # cv2.waitKey(0)
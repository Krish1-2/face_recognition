import os 
import tensorflow as tf
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
import pickle
import time

loaded_model = tf.keras.models.load_model("face_recognition_model.keras")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the label encoder
with open("label_encoder.pkl", 'rb') as file:
    label_encoder = pickle.load(file)

def recognize_faces(image_paths):
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (200, 200))
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
while count <= 10:
    ret, frame = video.read()
    count += 1
    cv2.imshow('windowsFrame', frame)
    image_path = 'doing/image' + str(count) + '.png'
    cv2.imwrite(image_path,frame)
    img=cv2.imread(image_path)
    faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    for x,y,w,h in faces_detected:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
        #removing background
        cv2.imwrite(image_path, img[y:y+h, x:x+w])
    image_paths.append(image_path)
video.release()

recognize_faces(image_paths)
cv2.destroyAllWindows()

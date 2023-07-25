from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.preprocessing.image import img_to_array
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt

train_images = []
train_labels = []

directory = 'dataset'
label_encoder_path = "label_encoder.pkl"
tensorboard_callbacks=tf.keras.callbacks.TensorBoard(log_dir='callbacks')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

for root, dirs, files in os.walk(directory):
    for file in files:
        file_name= os.path.join(root, file)
        img = cv2.imread(file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        for x,y,w,h in faces_detected:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            #removing background
            cv2.imwrite(file_name, img[y:y+h, x:x+w])
        img_normalized = img/255
        resized_img = cv2.resize(img_normalized, (200, 200))
        
        print("image shape=",resized_img.shape)
        image = img_to_array(resized_img)
        train_images.append(image)    
        # Extract the label from the directory name
        label = os.path.basename(root)
        train_labels.append(label)

train_images = np.array(train_images) 
train_labels = np.array(train_labels) 
print(train_images.shape,train_labels.shape)

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
print(train_labels)
train_labels = tf.keras.utils.to_categorical(train_labels,3)
with open(label_encoder_path, 'wb') as file:
    pickle.dump(label_encoder, file)


# X_train, X_test, y_train, y_test = train_test_split(X, y)
train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)


model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu',strides=(1, 1), input_shape=(200,200, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu',strides=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (5, 5), activation='relu',strides=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist=model.fit(train_images,train_labels, epochs=12,validation_data=(test_images, test_labels),callbacks=[tensorboard_callbacks])
model.save('face_recognition_model.keras')

# Plot training and validation accuracy
plt.plot(hist.history['accuracy'], label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
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

for root, dirs, files in os.walk(directory):
    for file in files:
        file_name= os.path.join(root, file)
        img = cv2.imread(file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_normalized = img/255

        # #removing background
        # _, alpha = cv2.threshold(img_normalized, 0, 255, cv2.THRESH_BINARY)
        # b, g, r = cv2.split(img_normalized)
        # rgba = [b, g, r, alpha]
        # img_normalized = cv2.merge(rgba, 4)
  
        print("image shape=",img_normalized.shape)

        # image = load_img(os.path.join(root, file), target_size=(480,640))
        resized_img = cv2.resize(img_normalized, (256, 256))
        image = img_to_array(resized_img)
        train_images.append(image)    
        # Extract the label from the directory name
        label = os.path.basename(root)
        train_labels.append(label)

train_images = np.array(train_images)  # Convert the list to a numpy array
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
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256,256, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist=model.fit(train_images,train_labels, epochs=3,callbacks=[tensorboard_callbacks])
model.summary()
model.save('face_recognition_model.keras')


fig=plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()
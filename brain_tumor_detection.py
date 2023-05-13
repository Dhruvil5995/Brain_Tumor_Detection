import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.utils.np_utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D, Activation, Dropout, Flatten, Dense

data_path =  'dataset/'
no_brain_tumor = os.listdir(data_path + 'no/')
yes_brain_tumor = os.listdir(data_path + 'yes/')

def processing(no, yes):
    dataset = []
    label = []


    for i, img_name in enumerate(no_brain_tumor):
        if (img_name.split('.')[1]=='jpg'):
            image= cv2.imread(data_path+'no/'+img_name)
            image=Image.fromarray(image,'RGB')
            image=image.resize((64,64))
            dataset.append(np.array(image))
            label.append(0)

    for i, img_name in enumerate(yes_brain_tumor):
        if (img_name.split('.')[1]=='jpg'):
            image= cv2.imread(data_path+'yes/'+img_name)
            image=Image.fromarray(image,'RGB')
            image=image.resize((64,64))
            dataset.append(np.array(image))
            label.append(1)

    #print('total_img',len(dataset))
    #print('total_label',len(label))

    data = np.array(dataset)
    labels = np.array(label)

    print(len(data))
    print(len(labels))

    return data, labels

def train_test(data, labels):
    #processing(no_brain_tumor, yes_brain_tumor)
    x_train,  x_test, y_train, y_test = train_test_split(data, labels, test_size= 0.25,random_state=0)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    #x_train = x_train.reshape(-1, 64, 64, 3)
    #x_test = x_test.reshape(-1, 64, 64, 3)

    x_train =normalize(x_train,axis=1)
    x_test =normalize(x_test,axis=1)

    return x_train,y_train,x_test,y_test

def train_model(x_train,y_train,x_test, y_test):

    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=16,
              verbose=1, epochs=25,
              validation_data=(x_test, y_test),
              shuffle=False)

    return model


data, labels = processing(no_brain_tumor, yes_brain_tumor)
x_train, y_train, x_test, y_test = train_test(data, labels)

train = train_model(x_train,y_train,x_test, y_test)
train.save('brain_model.h5')

loss, accuracy = train.evaluate(x_test, y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

# Make predictions on the test set
predictions = train.predict(x_test)
predicted_labels = np.round(predictions).flatten()

import matplotlib.pyplot as plt

# Plotting the first 10 test images with their predicted labels


plt.figure(figsize=(15, 10))
for i in range(20,36):
    plt.subplot(4, 4, i -19)
    plt.imshow(x_test[i])
    if predicted_labels[i] == 0:
        label = 'No Tumor'
    else:
        label = 'Tumor'
    plt.title(label)
    plt.axis('off')

plt.tight_layout()
plt.show()































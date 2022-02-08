from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.activations import sigmoid
import matplotlib.pyplot as plt

Shape = 200

Data_Generator = ImageDataGenerator(rotation_range=30,width_shift_range=0.1,
                                    height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,fill_mode='nearest'
                                    ,horizontal_flip=True,rescale=1/255)

train_data = Data_Generator.flow_from_directory(directory="C:\\Users\\dell\\PycharmProjects\\Covid-19_Classification\\Data\\Train",
                                             target_size=(Shape,Shape),batch_size=16,class_mode = 'binary')

test_data = Data_Generator.flow_from_directory(directory="C:\\Users\\dell\\PycharmProjects\\Covid-19_Classification\\Data\\Test",
                                               target_size=(Shape,Shape),batch_size=16,class_mode = 'binary')


model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(Shape,Shape,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# model.fit(train_data,epochs=100,validation_data=test_data)
model = load_model("Covid-CT-v2")
##
plt.plot(model.history.history["accuracy"])
plt.plot(model.history.history["val_accuracy"])
##
plt.plot(model.history.history["loss"])
plt.plot(model.history.history["val_loss"])
##
Data_Generator_test = ImageDataGenerator(rotation_range=0,width_shift_range=0.0,
                                    height_shift_range=0.0,shear_range=0.0,zoom_range=0.0,fill_mode='nearest'
                                    ,horizontal_flip=True,rescale=1/255)

test_data = Data_Generator_test.flow_from_directory(directory="C:\\Users\\dell\\PycharmProjects\\Covid-19_Classification\\Data\\Test",
                                                target_size=(200,200),batch_size=16,class_mode = 'binary',shuffle=False)
##
i = 0
plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    y = model.predict(test_data[0][0][i].reshape(1,200,200,3))
    plt.imshow(test_data[0][0][i])
    plt.title(f"{test_data.classes[i]} {y[0]}")
    plt.axis('off')

##
from sklearn.metrics import confusion_matrix

y_true = test_data.classes
y_pred = model.predict(test_data)
y_pred = y_pred.reshape(200)
y_pred = y_pred.round(decimals=0)

cm = confusion_matrix(y_pred,y_true)
print(cm)

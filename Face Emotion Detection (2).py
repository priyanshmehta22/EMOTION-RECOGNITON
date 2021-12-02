#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import progressbar  


# In[2]:


df = pd.read_csv("fer2013.csv")       #dataset used, train and test both.


# In[3]:


df.shape               #shape of the csv file(dataset)


# In[4]:


f = open("fer2013.csv")   #read path
f.__next__()
trainImage, trainLables = [],[]     #train dataset, labels' variables
valImage, valLables = [],[]         
testImage, testLables = [],[]       #test dataset, labels' variables


# In[5]:


widgets = ["Image Processing", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=35887, widgets=widgets).start()

for n, i in enumerate(f):
    lable, image, uses = i.strip().split(",")
    image = np.array(image.split(" "), dtype = "uint8")     #8 bit unsigned integer
    image = image.reshape((48,48, 1)) #reshape the array 
    #The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length.
    if uses == "Training":
        trainImage.append(image)
        trainLables.append(lable)
    elif uses == "PrivateTest":
        valImage.append(image)
        valLables.append(lable)
    else:
        testImage.append(image)
        testLables.append(lable)
    pbar.update(n)            #process bar ka function, visualizing the progression of a computer operation
pbar.finish()


# In[6]:


trainImage = np.array(trainImage)      #numpy array(changed object type)
valImage = np.array(valImage)
testImage = np.array(testImage)


# In[7]:


from tensorflow.keras.models import Sequential      #ML LIBRARY(pre programed model)
from tensorflow.keras.layers import  Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from imutils import paths
import numpy as np
import cv2
import os
import progressbar


# In[8]:


def LeNet(width, height, depth, classes):        #Convolutional Neural Network (CNN) using Python and the Keras deep learning package.
    model = Sequential()

    model.add(Conv2D(20, (5,5), activation = "relu", input_shape = (width, height, depth)))   #relu negative index ko 0 kar deta hai
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))


    model.add(Conv2D(50, (5,5), activation = "relu"))   #keras conv2d is  2D Convolution Layer, this layer creates a convolution 
                                                                                  #kernel that winds with layers input which helps produce a tensor of outputs.
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))         #A tensor is a vector or matrix of n-dimensions that represents all types of data. 

    model.add(Flatten())             #multi dimension to 1D
    model.add(Dense(500, activation = "relu"))      #adds output layer

    model.add(Dense(classes, activation = "softmax"))

    return model


# In[9]:


augmentation = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, 
                                  height_shift_range=0.2, zoom_range=0.2,horizontal_flip=True,
                                  fill_mode="nearest")   #image capture depiction i.e. deatils of image captured


# In[10]:


from sklearn.preprocessing import LabelBinarizer   #binarize labels in "1 vs all" fashion

trainLabels = LabelBinarizer().fit_transform(trainLables)
valLabels = LabelBinarizer().fit_transform(valLables)
testLabels = LabelBinarizer().fit_transform(testLables)


# In[11]:


# CODE TO TRAIN MODEL
model = LeNet(48, 48, 1, len(trainLabels[0]))
model.compile(loss = "binary_crossentropy", optimizer="adam", metrics=["accuracy"])     #Used as a loss function for binary classification model.
print("Model is Going to Train.....")
H = model.fit(augmentation.flow(trainImage, trainLabels, batch_size = 32), validation_data=(valImage, valLabels), steps_per_epoch=len(trainImage)//32, epochs=10, verbose=1)             
#model.fit feeding data to model


# In[12]:


#### Live Emotion Prediction
Emotion = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]   


# In[13]:


from tensorflow.keras.models import load_model    #tensorflow helps in training model and interference of deep neural networks
from tensorflow.keras.preprocessing.image import img_to_array, load_img   #turn the image into numpy array
import cv2
import numpy


# In[14]:


loaded_model = load_model("facial_emaotion_model.hdf5", compile=False)


# In[ ]:


import cv2             #image capture
face_class = cv2.CascadeClassifier(r"C:\Users\Priyansh Mehta\Desktop\TechSim_Intern\DL\CNN\Emotion detection\haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
while True:                 #capturing for each frame and converting into grayscale
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_class.detectMultiScale(gray, 1.3, 5) #scale factor is 1.03, it means we're using a small step for resizing, i.e. reduce size by 3 %

    for x,y,w,h in faces: 
        roi = gray[y:y+h, x:x+w]           #box in the video capture window
        roi = cv2.resize(roi, (64,64))    #region of intrest
        roi = roi.astype("float")/255  # array is divided by 255(the values inside the image will lie between 0 and 255),
                                                    #we use float as value will be between 0 and 1
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis = 0)
        
        pred = loaded_model.predict(roi)[0]
        prob = np.max(pred)
        label = Emotion[pred.argmax()]
        Status = label + "  " + str(prob)
        color = ""
        if label == "Happy":    
            color = (0,255,0)         #green box
        elif label=="Angry":
            color = (0,0,255)          #red box
        elif label=="Fear":
            color = (255,0,0)           #blue box
        elif label=="Disgust":
            color = (42,42,165)    #brown box
        elif label=="Sad":
            color = (160,160,160)  #grey box
        elif label=="Surprise":
            color = (255,255,51)   #light blue box
        else:              #neutral emotion
            color = (0,0,0)      #black box
            
        print(Status)  #printing the output emotion
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)       #box
        cv2.putText(img, Status, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.50, color, 2)  
        

    cv2.imshow("My Face", img)
    if cv2.waitKey(10) == 13:    #ascii value for enter
        break

cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





import numpy as np

def my_label(image_name):
    name = image_name.split('.')[-3] 


    if name=="Shree":
        return np.array([1,0,0])
    elif name=="priya":
        return np.array([0,1,0])
    elif name=="Gill":
        return np.array([0,0,1])

import os
import cv2
from random import shuffle
from tqdm import tqdm

def my_data():
    data = []
    for img in tqdm(os.listdir("data")):
        path = os.path.join("data", img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (50, 50))
        data.append([np.array(img_data), my_label(img)])
    shuffle(data)  
    return data

data = my_data()
train = data[:2400]  
test = data[2400:]

X_train = np.array([i[0] for i in train]).reshape(-1, 50, 50, 1)
print(X_train.shape)
y_train = [i[1] for i in train]

X_test = np.array([i[0] for i in test]).reshape(-1, 50, 50, 1)
print(X_test.shape)
y_test = [i[1] for i in test]

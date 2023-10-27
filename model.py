import numpy as np
import os
import cv2
from random import shuffle
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Define a label encoding dictionary
label_dict = {
    "Charishma": np.array([1, 0, 0]),
    "Sameera": np.array([0, 1, 0]),
    "Shreeraj": np.array([0, 0, 1]),
}

def my_label(image_name):
    name = image_name.split('.')[-3]
    return label_dict.get(name, np.array([0, 0, 0]))  # Default to [0, 0, 0] for unknown labels

def my_data():
    data = []
    for img in tqdm(os.listdir("data")):
        path = os.path.join("data", img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (60, 60))
        data.append([np.array(img_data), my_label(img)])
    shuffle(data)
    return data

data = my_data()
train = data[:2400]
test = data[2400:]

X_train = np.array([i[0] for i in train]).reshape(-1, 60, 60, 1)
y_train = np.array([i[1] for i in train])  # Keep the same shape as your model's output
X_test = np.array([i[0] for i in test]).reshape(-1, 60, 60, 1)
y_test = np.array([i[1] for i in test])  # Keep the same shape as your model's output

model = Sequential()

model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(60, 60, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation data should not be augmented
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Apply data augmentation to training data
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with data augmentation and early stopping
history = model.fit(
    train_generator,
    epochs=12,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

model.summary()

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

model.save("modified_model.h5")

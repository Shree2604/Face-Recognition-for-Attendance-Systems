import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from random import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Define a label encoding dictionary for 18 persons
label_dict = {
    "alluarjun": 0,
    "Ram": 1,
    "Mahesh": 2,
    "Rashi": 3,
    "Dhanush": 4,
    "Jiaxu": 5,
    "Hruthik": 6,
    "Sindhu": 7,
    "Mary": 8,
    "Saina": 9,
    "Sunil": 10,
    "Sundar": 11,
    "Chiranjeevi": 12,
    "Balakrishna": 13,
    "Shree": 14,
    "Suriya": 15,
    "KamalHaasan": 16,
    "Cherry": 17
}

# Handle unknown labels and filter out data with unknown labels
def my_label(image_name):
    name = image_name.split('.')[-3]
    label = label_dict.get(name, -1)  # Return -1 if not found
    return label


def load_and_preprocess_data(data_directory):
    data = []
    for img in tqdm(os.listdir(data_directory)):
        path = os.path.join(data_directory, img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (50, 50))  # Resize to a consistent size
        img_data = cv2.equalizeHist(img_data)  # Apply histogram equalization
        mean = np.mean(img_data)
        std = np.std(img_data)
        img_data = (img_data - mean) / std
        label = my_label(img)
        if label != -1:
            data.append([np.array(img_data), label])
    shuffle(data)
    return data

data_directory = "data"
data = load_and_preprocess_data(data_directory)

# Split data and labels
X_data = [item[0] for item in data]
y_labels = [item[1] for item in data]

# Split data into training and test sets
X_train = X_data[:32400]
X_test = X_data[32400:]

# Split labels into training and test sets
y_train = to_categorical(y_labels[:32400], num_classes=18)
y_test = to_categorical(y_labels[32400:], num_classes=18)


# Create and compile the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))

# Add Dropout layers for regularization
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

# Increase the number of output units in the final layer
model.add(Dense(18, activation='softmax'))

# Modify the learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Set up early stopping with increased patience
early_stopping = EarlyStopping(monitor='accuracy', patience=5)  # Adjust the patience as needed

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

# Reshape data to have rank 4 for the ImageDataGenerator
X_train = np.array(X_train).reshape(-1, 50, 50, 1)
y_train = np.array(y_train)

X_test = np.array(X_test).reshape(-1, 50, 50, 1)
y_test = np.array(y_test)

# Apply data augmentation to training data
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

# Train the model with data augmentation and early stopping
history = model.fit(train_generator, epochs=100, callbacks=[early_stopping])

model.summary()

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Save the model
model.save("modified_model.h5")


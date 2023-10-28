#  Face Recognition for Attendance Systems using Haar Cascade Algorithm

This repository hosts a face recognition project designed for automating attendance management systems using computer vision techniques.

## Table of Contents:

- [Face Recognition for Attendance Systems](#project-name)
  - [Description](#description)
  - [Getting Started](#Getting-Started)
  - [Prerequisites](#Prerequisites)
  - [Installation](#installation)
  - [Required Libraries](#Required-Libraries)


## Description

A Python project for face recognition using the Haar Cascade algorithm. This project is in its initial stages and serves as a foundation for building a face recognition system. It includes the following key components:

- `generateimages.py`: A script to capture and save images of known individuals for training the recognition model.
- `createdataandlabel.py`: A script to preprocess the captured images, create labels, and organize the training dataset.
- `model.py`: Contains the code for training a face recognition model using the preprocessed data.
- `testing.py`: A script to test the trained model on new images and perform face recognition.

## Getting Started

These instructions will help you set up the project on your local machine for development and testing purposes.

### Prerequisites

- Python 3.x
- Required Python libraries -- OpenCV , Numpy, Tensorflow, tqdm

### Installation
This section in your README file will inform users about the prerequisites and provide them with commands to install the required libraries. Users can copy and paste these commands into their terminal or command prompt to install the necessary dependencies for your project.

1. Clone the repository to your local machine.

   ```bash
   git clone https://github.com/your-username/face-recognition-haar-cascade.git
   cd face-recognition-haar-cascade
### Required Libraries

2. This project relies on several Python libraries. You can install them using `pip`:

- **OpenCV**: For image processing.
   ```bash
   pip install opencv-python
- NumPy: For numerical operations and array handling
   ```bash
   pip install numpy
- TensorFlow: For machine learning and deep learning tasks.
   ```bash
   pip install tensorflow
- tqdm: For displaying progress bars during time-consuming operations.
   ```bash
   pip install tqdm

### Usage
1. Capture images of known individuals using generateimages.py. Organize these images into folders with each person's name.
   ```bash
   python generateimages.py
2. Run createdataandlabel.py to preprocess the captured images and create training data and will store them in the data folder.
   ```bash
   python createdataandlabel.py
3. Train the face recognition model using model.py.
   ```bash
   python model.py
4. Test the trained model on new images for face recognition using testing.py.
   ```bash
   python testing.py


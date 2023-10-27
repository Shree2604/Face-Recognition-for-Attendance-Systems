import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained Keras model
model = load_model("modified_model.h5")

# Load an input image
input_image = cv2.imread("hi.jpg")  # Replace "input_image.jpg" with the path to your image
if input_image is None:
    print("Error: Could not load the input image.")
else:
    # Preprocess the loaded image to match the model's input shape (60x60 pixels, grayscale)
    img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (60, 60))
    img = img.reshape(1, 60, 60, 1)  # Reshape for model input

    # Make a prediction
    predictions = model.predict(img)

    # Get the class label with the highest probability
    class_label = np.argmax(predictions)

    # Define class labels (modify these according to your model's classes)
    class_labels = ["Panthulu", "Venkky", "Shree"]

    # Display the predicted class label
    cv2.putText(input_image, f"Prediction: {class_labels[class_label]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Create a larger window for displaying the image
    cv2.namedWindow("Input Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Input Image", input_image)

    # Wait for a key press to close the window
    cv2.waitKey(0)

    # Close the OpenCV window
    cv2.destroyAllWindows()

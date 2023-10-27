import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained Keras model
model = load_model("modified_model.h5") 

cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the image to match the model's input shape (60x60 pixels, grayscale)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (60, 60))
    img = img.reshape(1, 60, 60, 1)  # Reshape for model input

    # Make a prediction
    predictions = model.predict(img)

    # Get the class label with the highest probability
    class_label = np.argmax(predictions)

    # Define class labels (modify these according to your model's classes)
    class_labels = ["Karthik", "Sameera", "Shree"]

    # Display the predicted class label on the image
    cv2.putText(frame, f"Prediction: {class_labels[class_label]}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the image
    cv2.imshow("Image", frame)

    # Print the predicted class label in the terminal
    print(f"Prediction: {class_labels[class_label]}")

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

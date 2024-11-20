import tensorflow as tf
import numpy as np
import cv2

def preprocess_image(image):
    # Resize to model input size (416x416 for YOLO)
    image = cv2.resize(image, (416, 416))
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return np.expand_dims(image, axis=0).astype(np.float32)  # Add batch dimension

def draw_bounding_box(frame, prediction, confidence_threshold=0.5):
    h, w, _ = frame.shape
    # YOLO model predicts bounding box: [xmin, ymin, xmax, ymax]
    xmin, ymin, xmax, ymax = prediction
    # Convert normalized coordinates to pixel coordinates
    xmin = int(xmin * w)
    ymin = int(ymin * h)
    xmax = int(xmax * w)
    ymax = int(ymax * h)
    # Draw the bounding box and label
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    label = "Weed"
    cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)

def main(image_path):
    # Load the image from the provided path
    print(f"Loading image from {image_path}...")
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load the image.")
        return

    # Preprocess the image for the model
    input_image = preprocess_image(image)

    # Perform TensorFlow model inference
    print("Performing inference...")
    predictions = model.predict(input_image)[0]  # Get first prediction in the batch

    # Process and draw bounding boxes
    draw_bounding_box(image, predictions)

    # Display the result
    cv2.imshow("Weed Detection - Image", image)
    print("Press 'q' to close the window.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cv2.destroyAllWindows()

# Load the YOLO model
print("Loading YOLO model...")
model = tf.keras.models.load_model('../models/yolo_weed_detection_model.h5', compile=False)
model.compile(optimizer='adam', loss='mse') 
image_path = "images/test_image.jpg"
main(image_path)

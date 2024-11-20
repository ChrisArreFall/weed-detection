import tensorflow as tf
import numpy as np
import cv2

def preprocess_image(image, target_size):
    # Resize the image to the target size
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return np.expand_dims(image, axis=0).astype(np.float32)  # Add batch dimension

def draw_bounding_box(frame, prediction, target_size):
    h, w, _ = frame.shape
    input_w, input_h = target_size
    # Unpack prediction: [xmin, ymin, xmax, ymax]
    xmin, ymin, xmax, ymax = prediction
    # Convert normalized coordinates to pixel coordinates
    xmin = int(xmin * w / input_w)
    ymin = int(ymin * h / input_h)
    xmax = int(xmax * w / input_w)
    ymax = int(ymax * h / input_h)
    # Draw the bounding box and label
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    label = "Weed"
    cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)

def run_tflite_inference(tflite_model_path, image_path, target_size):
    # Load the TFLite model
    print(f"Loading TFLite model from {tflite_model_path}...")
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Load and preprocess the image
    print(f"Loading image from {image_path}...")
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load the image.")
        return
    input_image = preprocess_image(image, target_size)
    # Perform inference
    print("Performing inference...")
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]  # Get first prediction
    # Draw bounding box on the original image
    draw_bounding_box(image, prediction, target_size)
    # Display the result
    cv2.imshow("Weed Detection - YOLO TFLite", image)
    print("Press 'q' to close the window.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release resources
    cv2.destroyAllWindows()

tflite_model_path = "../models/yolo_weed_detection_model.tflite"
image_path = "images/test_image.jpg"
target_size = (416, 416)
run_tflite_inference(tflite_model_path, image_path, target_size)

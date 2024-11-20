import tensorflow as tf
import numpy as np
import cv2

def preprocess_image(image):
    image = cv2.resize(image, (640, 640))
    image = image / 255.0 
    return np.expand_dims(image, axis=0).astype(np.float32) 

def draw_bounding_box(frame, prediction):
    h, w, _ = frame.shape
    xmin, ymin, xmax, ymax = prediction
    xmin = int(xmin * w)
    ymin = int(ymin * h)
    xmax = int(xmax * w)
    ymax = int(ymax * h)
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    label = "Weed"
    cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)

def main(image_path):
    print(f"Loading image from {image_path}...")
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load the image.")
        return
    input_image = preprocess_image(image)
    print("Performing inference...")
    predictions = model.predict(input_image)[0]
    draw_bounding_box(image, predictions)
    cv2.imshow("Weed Detection - Image", image)
    print("Press 'q' to close the window.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cv2.destroyAllWindows()

print("Loading custom model...")
model = tf.keras.models.load_model('weed_detection_model.h5', compile=False)
model.compile(optimizer='adam', loss='mse') 
image_path = "images/test_image.jpg" 
main(image_path)

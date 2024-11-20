import tensorflow as tf
# Convert the models to TensorFlow Lite
print("Converting YOLO model to TensorFlow Lite...")
yolo_model = tf.keras.models.load_model("models/yolo_weed_detection_model.h5", compile=False)
print("Converting custom model to TensorFlow Lite...")
custom_model = tf.keras.models.load_model("models/weed_detection_model.h5", compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(yolo_model)
tflite_yolo_model = converter.convert()
with open("models/yolo_weed_detection_model.tflite", "wb") as f:
    f.write(tflite_yolo_model)
print("YOLO model saved as 'yolo_weed_detection_model.tflite'")


converter = tf.lite.TFLiteConverter.from_keras_model(custom_model)
tflite_custom_model = converter.convert()
with open("models/weed_detection_model.tflite", "wb") as f:
    f.write(tflite_custom_model)
print("Custom model saved as 'weed_detection_model.tflite'")

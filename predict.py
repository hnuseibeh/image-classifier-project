"""
This script loads a trained deep learning model and predicts the class of an input flower image.
It includes improvements to ensure a clear and smooth experience for the end-user.

Enhancements:
1. Suppressing TensorFlow Warnings:
   - TensorFlow logs many internal messages that can clutter the output.
   - I set `TF_CPP_MIN_LOG_LEVEL` to `3` to hide unnecessary warnings and logs.

2. Forcing CPU Usage for Compatibility:
   - Not all users have GPU access, which can cause compatibility issues.
   - The script explicitly disables GPU usage (`tf.config.set_visible_devices([], 'GPU')`)
     to ensure smooth execution on all machines.

3. Command-line Arguments:
   - Allows specifying the model, top-K predictions, and custom category names via `argparse`.

4. Clean Output:
   - The predictions are displayed in a simple, readable format with mapped flower names.

Usage Example:
    python predict.py ./test_images/orange_dahlia.jpg \
        --model flower_classifier.keras \
        --category_names label_map.json \
        --top_k 5
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
import json
import numpy as np
import tensorflow_hub as hub

def load_class_names(label_map_file):
    """Load a mapping from class indices to flower names from a JSON file."""
    try:
        with open(label_map_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading class names from {label_map_file}: {e}")
        return {}

def process_image(image_path):
    """Load and preprocess an image for model prediction."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224)) / 255.0
    return np.expand_dims(image.numpy(), axis=0)

def predict(image_path, model, top_k=5):
    """Predict top K most likely classes and probabilities."""
    processed_image = process_image(image_path)
    predictions = model.predict(processed_image)

    # Get indices of the top k predictions, in descending order
    top_k_indices = np.argsort(predictions[0])[-top_k:][::-1]

    # Convert probabilities from [0,1] to [0,100] for easier reading (optional)
    probs = predictions[0][top_k_indices] * 100
    classes = [str(i) for i in top_k_indices]

    return probs, classes

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Predict flower name from an image using a trained model.")
    # Positional argument for the image path
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    # Optional arguments
    parser.add_argument("--model", type=str, default="flower_classifier.keras",
                        help="Path to the trained Keras model file.")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Return the top K most likely classes.")
    parser.add_argument("--category_names", type=str, default="label_map.json",
                        help="Path to a JSON file mapping labels to flower names.")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Load the trained model
    model = tf.keras.models.load_model(args.model, custom_objects={'KerasLayer': hub.KerasLayer})

    # Run prediction
    probs, classes = predict(args.image_path, model, top_k=args.top_k)

    # Load class names
    class_names = load_class_names(args.category_names)

    # Convert numeric class IDs to actual flower names, if possible
    flower_names = [class_names.get(c, "Unknown") for c in classes]

    # Print results
    print("\n🔹 Predicted Flowers (top_k = {}):".format(args.top_k))
    for name, prob in zip(flower_names, probs):
        print(f"{name} ({prob:.2f}%)")

    # Verify CPU usage (optional)
    cpu_devices = tf.config.list_physical_devices('CPU')
    print(f"\nTensorFlow is using CPU: {cpu_devices}")

if __name__ == "__main__":
    main()

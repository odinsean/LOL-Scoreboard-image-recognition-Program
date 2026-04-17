import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pickle
import tensorflow as tf
from config import MODEL_NAME, ENCODER_PATH, IMAGES_PATH, LABELS_PATH, NAME_MAP

model = tf.keras.models.load_model(MODEL_NAME)

with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

# Load a small sample
images = np.load(IMAGES_PATH)
labels = np.load(LABELS_PATH)

# Check what the model predicts for first 10 images
predictions = model.predict(images[:10], verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

print("Sample predictions vs actual:")
for i in range(10):
    actual = le.inverse_transform([labels[i]])[0]
    actual = NAME_MAP.get(actual, actual)
    predicted = le.inverse_transform([predicted_classes[i]])[0]
    predicted = NAME_MAP.get(predicted, predicted)
    match = "✅" if actual == predicted else "❌"
    print(f"  {match} Actual: {actual} | Predicted: {predicted}")
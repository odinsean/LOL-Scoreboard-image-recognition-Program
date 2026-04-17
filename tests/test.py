import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pickle
import tensorflow as tf
from config import IMAGE_SIZE_WH, MODEL_NAME, ENCODER_PATH, NAME_MAP

# ── Load Model ───────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_NAME)

with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

print("Model loaded!\n")

# ── Test ─────────────────────────────────────────────────
print("Example inputs:")
print("  Same folder:  Jinx.png")
print("  Full path:    C:\\Users\\Username\\Desktop\\Jinx.png")
print()

while True:
    path = input("Enter image path (or Q to quit): ").strip()
    
    if path.upper() == "Q":
        break
        
    if not os.path.exists(path):
        print("File not found, try again\n")
        continue

    # What champion does the user expect?
    expected = input("What champion is this? (press Enter to skip): ").strip()

    # Load and preprocess
    img = cv2.imread(path)
    img = cv2.resize(img, IMAGE_SIZE_WH)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict top 3
    prediction = model.predict(img, verbose=0)[0]
    top3_idx = np.argsort(prediction)[::-1][:3]

    print("\n=== RESULTS ===")
    for i, idx in enumerate(top3_idx):
        champ = le.inverse_transform([idx])[0]
        champ = NAME_MAP.get(champ, champ)
        confidence = prediction[idx] * 100
        print(f"  #{i+1}: {champ} ({confidence:.1f}%)")

    if expected:
        predicted_names = [NAME_MAP.get(le.inverse_transform([idx])[0], le.inverse_transform([idx])[0]).lower() for idx in top3_idx]
        correct = expected.lower() in predicted_names
        print(f"\n  Expected: {expected}")
        print(f"  Correct: {'✅ YES' if correct else '❌ NO'}")

    print()
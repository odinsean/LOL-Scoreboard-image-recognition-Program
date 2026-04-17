import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from config import IMAGES_PATH, LABELS_PATH

labels = np.load(LABELS_PATH)
unique, counts = np.unique(labels, return_counts=True)
print(f"Unique classes: {len(unique)}")
print(f"Min count: {counts.min()}")
print(f"Max count: {counts.max()}")
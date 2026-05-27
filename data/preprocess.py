import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import albumentations as A
import pickle
from config import IMAGE_SIZE_WH, AUGMENTS_PER_IMAGE, ICON_FOLDER, IMAGES_PATH, LABELS_PATH, ENCODER_PATH

def load_icons(icon_folder, output_size):
    """
    Discovers all PNG champion icons in the given directory, standardizes 
    their shapes by resizing, normalizes pixel intensities to [0, 1], 
    and parses out the baseline champion name from the file names.
    """
    images, labels = [], []
    
    for img_path in Path(icon_folder).glob("*.png"):
        # Strip suffix to get base champion name (e.g., "Swain_sb1" -> "Swain")
        champion_name = img_path.stem.split("_sb")[0]
        
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skipping {img_path.name} - could not read")
            continue

        img = cv2.resize(img, output_size)
        img = img.astype(np.float32) / 255.0

        images.append(img)
        labels.append(champion_name)

    print(f"Loaded {len(images)} icons")
    return np.array(images), labels


def encode_labels(labels):
    """
    Maps text-based champion names to unique integers. 
    Serializes the fitted LabelEncoder object to disk via pickle 
    so it can be re-used later for inference decoding.
    """
    le = LabelEncoder()
    encoded = le.fit_transform(labels)

    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    print(f"Found {len(le.classes_)} champions")
    return encoded, le


def augment_dataset(images, labels):
    """
    Maps text-based champion names to unique integers. 
    Serializes the fitted LabelEncoder object to disk via pickle 
    so it can be re-used later for inference decoding.
    """
    # Robust transformation pipeline representing artifacts seen on game streams
    augmenter = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.3),
        A.GaussNoise(p=0.3),
        A.Rotate(limit=10, p=0.3),
        A.CoarseDropout(p=0.2),

        A.Downscale(scale_range=(0.3, 0.7), p=0.5),  # Quality loss simulation
        A.ImageCompression(quality_range=(30, 70), p=0.5), # JPEG stream compression
        A.GaussianBlur(blur_limit=3, p=0.3),               # Slight blurring
        A.Sharpen(p=0.3),                                   # Over-sharpening artifacts
        A.Perspective(scale=(0.05, 0.1), p=0.2),           # Slight angle warp
    ])

    aug_images, aug_labels = [], []

    for img, label in zip(images, labels):
       # Convert float back to uint8 because albumentations expects uint8 pixel ranges
        img_uint8 = (img * 255).astype(np.uint8)

        
        for _ in range(AUGMENTS_PER_IMAGE):
            # Re-normalize back to float32 [0, 1] for model consumption
            augmented = augmenter(image=img_uint8)["image"]
            aug_images.append(augmented.astype(np.float32) / 255.0)
            aug_labels.append(label)

    print(f"Augmented dataset: {len(aug_images)} total images")
    return np.array(aug_images), np.array(aug_labels)

def save_dataset(images, labels):
    """
    Saves the completely processed image matrices and label arrays 
    as binary numpy files (.npy) for fast disk I/O loading during training.
    """
    np.save(IMAGES_PATH, images)
    np.save(LABELS_PATH, labels)
    print("Saved images.npy and labels.npy")

# Run Program
images, labels = load_icons(ICON_FOLDER, IMAGE_SIZE_WH)
encoded_labels, le = encode_labels(labels)
aug_images, aug_labels = augment_dataset(images, encoded_labels)
save_dataset(aug_images, aug_labels)

print("Preprocessing complete!")

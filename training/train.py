import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
from config import IMAGE_SIZE, BATCH_SIZE, EPOCHS, MODEL_NAME, IMAGES_PATH, LABELS_PATH, ENCODER_PATH

def load_data():
    """
    Loads the preprocessed image and label arrays from disk,
    unpickles the LabelEncoder to determine the total class count,
    and displays primary dataset metrics.
    """
    print("Loading data...")
    images = np.load(IMAGES_PATH)
    labels = np.load(LABELS_PATH)

    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

    # Print dataset statistics
    num_classes = len(le.classes_)
    print(f"Classes: {num_classes} champions")
    print(f"Dataset size: {len(images)} images")

    return images, labels, num_classes


def build_model(num_classes):
    """
    Constructs a 4-block Convolutional Neural Network (CNN) 
    interleaved with Batch Normalization and Max Pooling, feeding 
    into a Dense classifier layer with Dropout regularizer.
    
    Compiles the architecture using Adam and Sparse Categorical Crossentropy.
    """
    print("Building model...")
    
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=IMAGE_SIZE),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Block 4
        layers.Conv2D(256, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Dense Classifier Classifier
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model

def plot_history(history):
    """
    Generates side-by-side subplot tracking metric progress 
    (Accuracy and Loss) across training epochs for both training 
    and validation subsets. Saves the final layout to disk.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"], label="Train")
    ax1.plot(history.history["val_accuracy"], label="Val")
    ax1.set_title("Accuracy")
    ax1.legend()

    ax2.plot(history.history["loss"], label="Train")
    ax2.plot(history.history["val_loss"], label="Val")
    ax2.set_title("Loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_results.png")
    plt.show()

# Load Data 
images, labels, num_classes = load_data()

# Train/Val Split (with stratification to preserve class distributions)
X_train, X_val, y_train, y_val = train_test_split(
    images, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print(f"Training: {len(X_train)} | Validation: {len(X_val)}")

# Build Model Architecture
model = build_model(num_classes)

# Training Constraints & Checkpoints
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_NAME, save_best_only=True)
]

print("\nTraining...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# Output evaluation graphs
plot_history(history)

# Final Diagnostics Evaluator
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"\nFinal Validation Accuracy: {accuracy * 100:.2f}%")
print(f"Model saved as {MODEL_NAME}")

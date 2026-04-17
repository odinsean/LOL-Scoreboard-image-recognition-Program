import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# config.py
IMAGE_SIZE_WH = (64, 64) # width, height
IMAGE_SIZE = (64, 64, 3)  # for model input
AUGMENTS_PER_IMAGE = 50
BATCH_SIZE = 32
EPOCHS = 50  
ICON_SIZE_RATIO = 0.048
ICON_FOLDER = os.path.join(ROOT, "champion_icons")
MODEL_NAME = os.path.join(ROOT, "model.keras")
IMAGES_PATH = os.path.join(ROOT, "images.npy")
LABELS_PATH = os.path.join(ROOT, "labels.npy")
ENCODER_PATH = os.path.join(ROOT, "label_encoder.pkl")
RESULTS_FILE = os.path.join(ROOT, "results.xlsx")
NAME_MAP = {
    "MonkeyKing": "Wukong",
    "JarvanIV": "Jarvan IV"
}
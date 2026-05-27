# League of Legends Scoreboard Image Recognition Program

An end-to-end Machine Learning pipeline designed to recognize League of Legends champions directly from in-game match scoreboard screenshots. By implementing an elegant spatial-interpolation mechanism, the system minimizes user friction, requiring only **two structural mouse clicks** to locate, crop, and identify all five champions within a team composition layout. 

---

## 1. Directory Structure Blueprint
Before executing the scripts, ensure your local workspace matches the directory framework below:

```text
project_root/
│
├── config.py              # Path definitions & dynamic scaling ratios
├── calibration.json       # Structural interface configuration matrix
├── model.keras            # Serialized deep learning model weights
│
├── champion_icons/        # Primary directory for raw and cropped champion icons
│
├── data/
│   ├── download_icons.py
│   ├── collect_scoreboard_icons.py
│   ├── preprocess.py
│   └── scoreboard_testing/# Input folder for multi-champion test screenshots
│
├── predict/
│   └── predict.py         # Production inference and Excel logging pipeline
│
└── training/
    └── train.py           # Neural network construction and fitting script
```
## 2. Installation & Environment Setup
Open your terminal environment and install the required data science, computer vision, and deep learning libraries:

```bash
pip install opencv-python numpy albumentations scikit-learn tensorflow openpyxl matplotlib
```
## 3. Operational Workflow

To initialize, train, and deploy the vision pipeline, execute the primary modules in their strict chronological progression.

### Phase 1: Data Acquisition (`data/collect_scoreboard_icons.py`)
This script populates your asset catalog with context-accurate champion icon crops isolated from raw screenshots.
```bash
python data/collect_scoreboard_icons.py
```

Save an in-game scoreboard screenshot into your data/scoreboard_testing/ folder.

Crucial: Name the file using a dash-separated naming convention that perfectly lists the 5 champions in the exact order they appear from top to bottom (e.g., Swain-Vi-Ambessa-Velkoz-Morgana.png).

Run the script and input your screenshot path into the terminal prompt.

An interactive image window will open. Click precisely on the center of the TOP lane champion icon.

Next, click precisely on the center of the SUPPORT lane champion icon.

Press Q on your keyboard to calculate the linear spacing, automatically slice all 5 frames, and save them as labeled images to your data/champion_icons/ folder.

### Phase 2: Compilation & Augmentation (`data/preprocess.py`)
Compile individual champion crops into a highly variable, tensor-ready training dataset.
```bash
python data/preprocess.py
```
What it does: The script automatically resizes images to uniform bounding scales and normalizes pixel spaces. It then runs an intensive data augmentation pipeline (applying compression artifacts, motion blur, random contrast shifts, and perspective warping) to simulate stream lag and varying video qualities.

Output: Generates images.npy, labels.npy, and a serialized LabelEncoder object map inside your data configuration bindings.

### Phase 3: Model Training (`training/train.py`)
Train your convolutional neural network to learn and distinguish champion visual signatures.
```bash
python training/train.py
```

What it does: This module handles data partitioning by setting up an 80/20 train-test split stratified by class distributions. It builds and trains a 4-block deep 2D Convolutional Neural Network (CNN) complete with Batch Normalization and Dropout layers.

Output: Saves a plot of training performance to training_results.png and overwrites/saves your top-performing weights to model.keras.

### Phase 4: Production Inference (`predict/predict.py`)
Run real-time predictions on unseen scoreboard captures and automatically catalog your match history.
```bash
python predict/predict.py
```
Supply a path to any raw, unstructured scoreboard screenshot when prompted by the CLI interface.

Complete the same 2-click sequence (Click 1: Top icon center, Click 2: Support icon center) and press Q to finalize.

The neural graph identifies each layout role position, prints the predicted champion composition directly to the console, and automatically appends a structured record row into a tracking spreadsheet named results.xlsx.

## Strategic Core Engineering Notes
Anchor Alignment: The mathematical spatial interpolation operates under the assumption of a fixed, linear structural layout standard to the League of Legends scoreboard UI. Always center your anchor clicks as perfectly as possible on the Top and Support frames to avoid misaligned bounding box crops down the line.

Config Constants: Ensure variables such as ICON_SIZE_RATIO, IMAGE_SIZE_WH, and internal folder target directions are correctly configured inside config.py to map accurately to your localized file paths and layout parameters.

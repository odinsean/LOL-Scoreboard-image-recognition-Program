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

## 2. Installation & Environment Setup
Open your terminal environment and install the required data science, computer vision, and deep learning libraries:

```bash
pip install opencv-python numpy albumentations scikit-learn tensorflow openpyxl matplotlib

## 3. Operational Workflow

To initialize, train, and deploy the vision pipeline, execute the primary modules in their strict chronological progression.

### Phase 1: Data Acquisition (`data/collect_scoreboard_icons.py`)
This script populates your asset catalog with context-accurate champion icon crops isolated from raw screenshots.
```bash
python data/collect_scoreboard_icons.py

### Phase 1: Data Acquisition (`data/collect_scoreboard_icons.py`)
This script populates your asset catalog with context-accurate champion icon crops isolated from raw screenshots.
```bash
python data/collect_scoreboard_icons.py

***

### Phase 2 Markdown Block
```markdown
### Phase 2: Compilation & Augmentation (`data/preprocess.py`)
Compile individual champion crops into a highly variable, tensor-ready training dataset.
```bash
python data/preprocess.py

***

### Phase 3 Markdown Block
```markdown
### Phase 3: Model Training (`training/train.py`)
Train your convolutional neural network to learn and distinguish champion visual signatures.
```bash
python training/train.py

***

### Phase 4 Markdown Block
```markdown
### Phase 4: Production Inference (`predict/predict.py`)
Run real-time predictions on unseen scoreboard captures and automatically catalog your match history.
```bash
python predict/predict.py

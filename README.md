# League of Legends Scoreboard Image Recognition Program
 
An end-to-end Machine Learning pipeline that automatically recognizes League of Legends champions from post-game scoreboard screenshots and logs them into an Excel file — built to speed up the scouting process for competitive LoL teams at any level.
 
---
 
## Directory Structure
 
Before running any scripts, make sure your local workspace matches the structure below:
 
```
project_root/
│
├── config.py                  # Path definitions and scaling settings
├── model.keras                # Trained model weights
│
├── champion_icons/            # Stores all champion icons (clean + scoreboard crops)
│
├── data/
│   ├── download_icons.py      # Downloads all champion icons from Riot's API
│   ├── collect_scoreboard_icons.py  # Crops and labels icons from real scoreboard screenshots
│   ├── preprocess.py          # Prepares and augments training data
│   └── scoreboard_testing/    # Put your labeled scoreboard screenshots here
│
├── predict/
│   └── predict.py             # Run this to predict champions and log to Excel
│
├── tests/
│   └── check_data.py          # Verify your training data looks correct
│
└── training/
    └── train.py               # Trains the neural network
```
 
---
 
## Installation
 
Open your terminal and install all required libraries:
 
```
pip install opencv-python numpy albumentations scikit-learn tensorflow openpyxl matplotlib requests
```
 
---
 
## Workflow
 
Follow these phases in order the first time you set up the program.
 
---
 
### Phase 0: Download Champion Icons (`data/download_icons.py`)
 
This only needs to be run once. It downloads every champion's icon directly from Riot's public API and saves them to your `champion_icons/` folder — no API key required.
 
```
python data/download_icons.py
```
 
Each icon is automatically named after the champion (e.g. `Jinx.png`, `Ahri.png`), so no manual labeling is needed.
 
---
 
### Phase 1: Collect Scoreboard Icons (`data/collect_scoreboard_icons.py`)
 
This step adds real scoreboard icon crops to your training data so the model learns what champions look like in an actual game — not just in clean promotional art.
 
```
python data/collect_scoreboard_icons.py
```
 
**How to prepare your screenshots:**
 
1. Take a post-game scoreboard screenshot from the League client
2. Save it into your `data/scoreboard_testing/` folder
3. Name the file using the 5 champions in order from Top to Support, separated by dashes
```
Example filename: Swain-Vi-Ambessa-Velkoz-Morgana.png
```
 
> **Naming rules:** Remove spaces from two-word champion names (e.g. `MissFortune`, `TwistedFate`, `LeeSin`). Wukong should be named `MonkeyKing` to match Riot's internal naming.
 
**How to run the script:**
 
1. Enter the screenshot path when prompted (e.g. `scoreboard_testing/Swain-Vi-Ambessa-Velkoz-Morgana.png`)
2. An image window will open — click the center of the **Top lane** champion icon
3. Then click the center of the **Support** champion icon
4. Press **Q** to confirm — the program will automatically calculate and crop all 5 icons
The cropped icons will be saved to your `champion_icons/` folder labeled by champion name. Collect as many screenshots as possible for better accuracy — aim for at least 10.
 
---
 
### Phase 2: Preprocess Training Data (`data/preprocess.py`)
 
Compiles all champion icons into a training-ready dataset and applies augmentation to simulate real-world scoreboard conditions like compression artifacts, blur, and lighting changes.
 
```
python data/preprocess.py
```
 
Outputs `images.npy`, `labels.npy`, and `label_encoder.pkl` to your project root.
 
---
 
### Phase 3: Train the Model (`training/train.py`)
 
Trains a Convolutional Neural Network (CNN) to recognize champions from their icons. Uses an 80/20 train/validation split and automatically stops early if the model stops improving.
 
```
python training/train.py
```
 
Saves the best performing model to `model.keras` and outputs a training accuracy/loss plot to `training_results.png`.
 
---
 
### Phase 4: Predict & Log Results (`predict/predict.py`)
 
Run this whenever you want to process scoreboard screenshots and log the results to Excel. You do not need to retrain before each run — just run this directly.
 
```
python predict/predict.py
```
 
**How it works:**
 
1. Enter the path to your scoreboard screenshot when prompted
2. Click the **Top** icon, then the **Support** icon in the image window
3. Press **Q** to confirm
4. The program predicts all 5 champions, prints them to the console, and appends a new row to `results.xlsx`
```
=== RESULTS ===
  Top:     Swain
  Jungle:  Vi
  Mid:     Ambessa
  ADC:     Velkoz
  Support: Morgana
```
 
When asked if there are more screenshots, enter **Y** to continue or **N** to finish. All results are saved to a single `results.xlsx` file that persists across runs.
 
---
 
## Retraining When a New Champion Is Released
 
Riot releases new champions a few times per year. When that happens, follow these steps to update your model:
 
1. Run `download_icons.py` to download the new champion's icon
2. Collect a few scoreboard screenshots featuring the new champion and run `collect_scoreboard_icons.py`
3. Re-run `preprocess.py` to rebuild the dataset
4. Re-run `train.py` to retrain the model
You only need to do this when new champions are released — your existing `results.xlsx` data is never affected.
 
---
 
## Notes
 
- **Click accuracy matters** — always click as close to the center of the Top and Support icons as possible. The program interpolates all 5 positions from just these 2 clicks, so misaligned anchors will cause incorrect crops.
- **Two-word champion names** — remove spaces when naming screenshot files (e.g. `MissFortune` not `Miss Fortune`)
- **More scoreboard data = better accuracy** — the more real scoreboard screenshots you collect in Phase 1, the better the model will perform on real games
---
 
## About
 
Built to automate the manual process of logging champion picks from post-game scoreboard screenshots into a scouting spreadsheet, saving time for coaching staff and analysts at any level of competitive League of Legends.

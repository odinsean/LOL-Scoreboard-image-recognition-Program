import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
from config import ICON_FOLDER, ICON_SIZE_RATIO

# Global list to track user clicks in the UI
click_points = []

def click_event(event, x, y, flags, param):
    """
    Handles mouse click events on the screenshot window.
    Captures exactly 2 points (Top lane and Support lane icons) and 
    provides visual feedback by drawing circles on the image.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(click_points) < 2:
            click_points.append((x, y))

            # Visual feedback: draw green circle at click location
            cv2.circle(param, (x, y), 8, (0, 255, 0), -1)
            cv2.imshow("Screenshot", param)

            if len(click_points) == 1:
                print("Got Top icon! Now click your SUPPORT icon")
            elif len(click_points) == 2:
                print("Got Support icon! Press Q to continue")

def validate_champions(champion_names):
    """
    Validates that the champion names parsed from the filename 
    actually exist as base icons in the target icon folder.
    
    Returns a list of invalid champion names found.
    """
    invalid = []
    for champ in champion_names:
        base_icon = os.path.join(ICON_FOLDER, f"{champ}.png")
        if not os.path.exists(base_icon):
            invalid.append(champ)
    return invalid

def collect(screenshot_path, champion_names):
    """
    Loads a scoreboard screenshot, displays it for user interaction,
    and crops out 5 individual champion icons based on the linear spacing 
    between the user's Top and Support clicks. Saves cropped icons for training.
    """
    click_points.clear()

    img = cv2.imread(screenshot_path)
    if img is None:
        print("Could not load image!")
        return

    clone = img.copy()
    cv2.imshow("Screenshot", clone)
    cv2.setMouseCallback("Screenshot", click_event, clone)

    print("\nClick TOP icon then SUPPORT icon")
    print("Press Q when done\n")

    # Wait for clicks or manual exit via 'Q'
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        if len(click_points) == 2:
            break

    cv2.destroyAllWindows()

    if len(click_points) != 2:
        print("Expected 2 clicks, try again")
        return

    top_x, top_y = click_points[0]
    sup_x, sup_y = click_points[1]

    # Calculate average pixel spacing between the 5 scoreboard positions (4 gaps)
    spacing_x = (sup_x - top_x) / 4
    spacing_y = (sup_y - top_y) / 4

    icon_size = max(15, round(img.shape[0] * ICON_SIZE_RATIO))
    ROLES = ["Top", "Jungle", "Mid", "ADC", "Support"]

    
    for i, champ in enumerate(champion_names):
        # Interpolate coordinates for the current role
        x = round(top_x + spacing_x * i)
        y = round(top_y + spacing_y * i)

        # Crop bound constraints around the calculated center point
        crop = img[y - icon_size:y + icon_size, x - icon_size:x + icon_size]

        if crop.size == 0:
            print(f"Could not crop {champ}, skipping")
            continue

        # Check existing scoreboard crops to determine next index and prevent overwrites
        existing = [f for f in os.listdir(ICON_FOLDER) if f.startswith(f"{champ}_sb")]
        count = len(existing) + 1

        filename = os.path.join(ICON_FOLDER, f"{champ}_sb{count}.png")
        cv2.imwrite(filename, crop)
        print(f"Saved {ROLES[i]}: {filename}")

# Main Interactive Loop
print("Example inputs:")
print("  scoreboard_testing/Swain-Vi-Ambessa-Velkoz-Morgana.png")
print()

while True:
    screenshot = input("\nEnter screenshot path: ").strip()

    if not os.path.exists(screenshot):
        print("File not found, try again")
        continue

    # Extract champion names from the filename (expecting: Champ1-Champ2-...)
    filename = os.path.splitext(os.path.basename(screenshot))[0]
    champion_names = [n.strip() for n in filename.split("-")]

    if len(champion_names) != 5:
        print("Filename must have exactly 5 champion names separated by -")
        print("Example: Swain-Vi-Ambessa-Velkoz-Morgana.png")
        continue

    print(f"Champions detected: {champion_names}")

    invalid = validate_champions(champion_names)
    if invalid:
        print(f"Invalid champion names: {invalid}")
        print("Check spelling and make sure names match champion_icons folder")
        continue

    collect(screenshot, champion_names)

    more = input("\nMore screenshots? (Y/N): ").strip().upper()
    if more != "Y":
        print("\nDone! Run preprocess.py and train.py to retrain.")
        break

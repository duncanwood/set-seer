import cv2
import numpy as np
import os
import glob

def check_card():
    # Find a green card
    search_path = "model-dev/data/train/1-green-solid-oval/*.png" # Try png
    files = glob.glob(search_path)
    if not files:
        search_path = "model-dev/data/train/1-green-solid-oval/*.jpg"
        files = glob.glob(search_path)
    
    if not files:
        print("No green card found to check.")
        # Try finding ANY green card dir
        root = "model-dev/data/train"
        for d in os.listdir(root):
            if "green" in d and os.path.isdir(os.path.join(root, d)):
                files = glob.glob(os.path.join(root, d, "*"))
                if files: break
    
    if not files:
        print("Could not find any green card images.")
        return

    path = files[0]
    print(f"Checking file: {path}")
    
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Failed to load.")
        return
        
    print(f"Shape: {img.shape}")
    
    # Check center crop stats
    h, w = img.shape[:2]
    center = img[h//2-10:h//2+10, w//2-10:w//2+10]
    
    mean = center.mean(axis=(0,1))
    print(f"Mean BGR(A): {mean}")
    
    # Interpretation
    b, g, r = mean[0], mean[1], mean[2]
    print(f"B: {b:.1f}, G: {g:.1f}, R: {r:.1f}")
    
    if g > r and g > b:
        print("Verified GREEN dominant.")
    elif r > g and r > b:
        print("WARNING: RED dominant (Source might be RGB interpreted as BGR or just Red).")
    elif b > g and b > r:
        print("WARNING: BLUE dominant.")
    else:
        print("Mixed color / Neutral.")

if __name__ == "__main__":
    check_card()

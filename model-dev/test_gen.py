import cv2
import numpy as np
import torch
from set_seer_dataset import SetSeerDataset
import os

def test_generation():
    # Setup dataset
    ds = SetSeerDataset(
        card_dir="model-dev/data/train",
        dtd_dir="model-dev/data/dtd/images",
        min_cards=1, max_cards=1,
        bgr=False # Standard RGB output requested
    )
    
    # Generate 10 images until we get a Green card
    for i in range(20):
        item = ds[i]
        img_tensor = item['img'] # (3, H, W) RGB
        cls_ids = item['cls']
        
        # Convert tensor back to numpy for inspection
        # Tensor is RGB (CHW) -> (HWC)
        img_np = img_tensor.permute(1, 2, 0).numpy() # RGB
        
        # Find the card class name
        # We need class_to_idx mapping reversed
        idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
        class_id = int(cls_ids[0].item())
        class_name = idx_to_class.get(class_id, "Unknown")
        
        if "green" in class_name:
            print(f"Found Green Card: {class_name} (Index {i})")
            
            # Analyze colors of the generated card area
            # We don't have the exact mask here, but we can look for non-grey pixels
            # or just save it as BGR (for OpenCV)
            
            # RGB to BGR for saving
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            filename = f"model-dev/test_gen_green_{i}.jpg"
            cv2.imwrite(filename, img_bgr)
            print(f"Saved {filename}")
            
            # Check statistics of center stats (rough) or just report
            # We want to know if G > R
            means = img_np.mean(axis=(0,1))
            print(f"Global Means RGB: {means}")
            
            # Start strict color check on pixels that are "strong" colors (saturation > threshold)
            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            mask = hsv[:, :, 1] > 50 # Saturation > 50
            if mask.any():
                strong_pixels = img_np[mask]
                mean_strong = strong_pixels.mean(axis=0)
                print(f"Strong Color Pixels Mean RGB: {mean_strong}")
            
            break

if __name__ == "__main__":
    test_generation()

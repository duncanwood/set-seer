import os
import cv2
import numpy as np
import random
from set_seer_dataset import SetSeerDataset

def main():
    card_dir = "model-dev/data/train"
    dtd_dir = "model-dev/data/dtd/images"
    output_dir = "model-dev/debug_runs/perspective_test"
    os.makedirs(output_dir, exist_ok=True)

    dataset = SetSeerDataset(card_dir, dtd_dir)
    
    # Pick a random card
    sample_path, _ = random.choice(dataset.card_samples)
    card_img = cv2.imread(sample_path, cv2.IMREAD_UNCHANGED)
    
    for i in range(10):
        # We need to manually call the transformation logic to see intermediate results
        transformed_card, mask = dataset.apply_random_transformations(card_img)
        
        if transformed_card is None:
            continue
            
        # Draw a bright border on the transformed image to see its edges
        viz = transformed_card.copy()
        if viz.shape[2] == 4:
            viz = cv2.cvtColor(viz, cv2.COLOR_BGRA_BGR)
            
        cv2.rectangle(viz, (0, 0), (viz.shape[1]-1, viz.shape[0]-1), (0, 255, 0), 2)
        
        # Save both image and mask
        cv2.imwrite(os.path.join(output_dir, f"sample_{i}_card.jpg"), viz)
        cv2.imwrite(os.path.join(output_dir, f"sample_{i}_mask.jpg"), mask)

    print(f"Saved 10 perspective test samples to {output_dir}")

if __name__ == "__main__":
    main()

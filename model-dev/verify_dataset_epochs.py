import os
import cv2
import numpy as np
import torch
from set_seer_dataset import SetSeerDataset

def compare_images(img1, img2):
    """Returns True if images are identical."""
    if img1.shape != img2.shape:
        return False
    # Use pixel-wise difference
    diff = cv2.absdiff(img1, img2)
    return np.mean(diff) == 0

def main():
    # Setup paths
    card_dir = "model-dev/data/train"
    dtd_dir = "model-dev/data/dtd/images"
    output_dir = "model-dev/debug_runs/epoch_check"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Initializing dataset with:\n  Cards: {card_dir}\n  Backgrounds: {dtd_dir}")
    
    dataset = SetSeerDataset(
        card_dir=card_dir,
        dtd_dir=dtd_dir,
        img_size=640,
        epoch_size=10, # Small size for check
        min_cards=1,
        max_cards=15
    )

    # Fetch 10 samples and average instance counts
    instance_counts = []
    images = []
    for i in range(10):
        print(f"Fetching sample {i}...")
        sample = dataset[i]
        # sample['img'] is (3, 640, 640) torch tensor
        img_np = sample['img'].permute(1, 2, 0).numpy() # (640, 640, 3)
        if len(images) < 3: # Keep first 3 for epoch comparison
            images.append(img_np)
        
        num_instances = len(sample['cls'])
        instance_counts.append(num_instances)
        
        # Save for manual inspection
        save_path = os.path.join(output_dir, f"sample_{i}.jpg")
        # SetSeerDataset returns RGB, cv2 expects BGR for imwrite
        cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        print(f"  Saved to {save_path} (Instances: {num_instances})")

    avg_instances = sum(instance_counts) / len(instance_counts)
    print(f"\nAverage instances per image: {avg_instances:.2f} (Target was up to 15)")

    # Compare
    is_0_1_same = compare_images(images[0], images[1])
    is_1_2_same = compare_images(images[1], images[2])
    is_0_2_same = compare_images(images[0], images[2])

    print("\n--- Results ---")
    print(f"Epoch 0 vs 1: {'SAME (FAIL)' if is_0_1_same else 'DIFFERENT (PASS)'}")
    print(f"Epoch 1 vs 2: {'SAME (FAIL)' if is_1_2_same else 'DIFFERENT (PASS)'}")
    print(f"Epoch 0 vs 2: {'SAME (FAIL)' if is_0_2_same else 'DIFFERENT (PASS)'}")

    if not (is_0_1_same or is_1_2_same or is_0_2_same):
        print("\nSUCCESS: All samples at index 0 across 3 'epochs' are unique!")
    else:
        print("\nFAILURE: Some samples are identical, augmentation and randomness might not be working as expected.")

if __name__ == "__main__":
    main()

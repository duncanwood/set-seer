import os
import shutil
import random
from set_seer_dataset import SET_CARDS_MASTER, SetSeerDataset

def main():
    train_root = "model-dev/data/train"
    val_root = "model-dev/data/val"
    num_to_move = 10

    if not os.path.exists(train_root):
        print(f"Error: Training root {train_root} not found.")
        return

    # Create val_root if it doesn't exist
    os.makedirs(val_root, exist_ok=True)

    # 1. Map master classes to actual folders in train
    # folder_map: normalized_name -> actual_folder_name
    actual_train_folders = [d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))]
    folder_map = {SetSeerDataset.normalize_name(d): d for d in actual_train_folders}

    print(f"Planning to rebalance 81 classes by moving {num_to_move} images each from train to val.")
    
    moved_counts = {}
    missing_classes = []

    for master_class in SET_CARDS_MASTER:
        norm_name = SetSeerDataset.normalize_name(master_class)
        
        if norm_name not in folder_map:
            missing_classes.append(master_class)
            continue
        
        actual_train_folder = folder_map[norm_name]
        src_dir = os.path.join(train_root, actual_train_folder)
        dst_dir = os.path.join(val_root, actual_train_folder) # Keep name consistent with train
        
        os.makedirs(dst_dir, exist_ok=True)
        
        # Get files to move
        files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
        files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        random.shuffle(files)
        to_move = files[:num_to_move]
        
        count = 0
        for f in to_move:
            src_path = os.path.join(src_dir, f)
            dst_path = os.path.join(dst_dir, f)
            
            # Simple move
            shutil.move(src_path, dst_path)
            count += 1
            
        moved_counts[master_class] = count
        if count < num_to_move:
            print(f"  Note: Only moved {count} images for {master_class} (fewer than {num_to_move} available).")

    print("\n--- Rebalance Summary ---")
    print(f"Classes processed: {len(moved_counts)}")
    print(f"Total images moved: {sum(moved_counts.values())}")
    
    if missing_classes:
        print(f"\nWarning: {len(missing_classes)} classes were missing in the training folder:")
        for mc in missing_classes:
            print(f"  - {mc}")

    print("\nSUCCESS: Validation set rebalanced. Files were MOVED, so there is no overlap with training.")

if __name__ == "__main__":
    main()

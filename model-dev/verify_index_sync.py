import os
from set_seer_dataset import SetSeerDataset

def main():
    train_dir = "model-dev/data/train"
    val_dir = "model-dev/data/val"
    dtd_dir = "model-dev/data/dtd/images"
    
    if not os.path.isdir(train_dir):
        print("Run from project root.")
        return

    # 1. Get master classes from train
    master_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    print(f"Master classes loaded: {len(master_classes)}")

    # 2. Setup datasets
    train_ds = SetSeerDataset(train_dir, dtd_dir, master_classes=master_classes)
    val_ds = SetSeerDataset(val_dir, dtd_dir, master_classes=master_classes)

    # 3. Find a common card (e.g., 1-red-solid-diamond)
    # Note: val might have it as 1-red-solid-diamond or plural
    common_card_search = "1-red-solid-diamond"
    
    def get_info(ds, name):
        # ds.card_samples is [(path, mapped_idx)]
        samples = []
        for path, idx in ds.card_samples:
            if SetSeerDataset.normalize_name(os.path.basename(os.path.dirname(path))) == SetSeerDataset.normalize_name(name):
                samples.append(idx)
        return list(set(samples))

    train_ids = get_info(train_ds, common_card_search)
    val_ids = get_info(val_ds, common_card_search)

    print(f"\nChecking class: {common_card_search}")
    print(f"Train IDs: {train_ids}")
    print(f"Val IDs  : {val_ids}")

    if train_ids == val_ids and len(train_ids) == 1:
        print("\nSUCCESS: Indices are synchronized!")
    else:
        print("\nFAILURE: Indices mismatch or card not found in both.")
        
    # Check a plural case
    plural_card = "3-green-striped-diamonds"
    canonical = SetSeerDataset.normalize_name(plural_card)
    print(f"\nChecking plural card: {plural_card} (canonical: {canonical})")
    
    p_train_ids = get_info(train_ds, plural_card)
    p_val_ids = get_info(val_ds, plural_card)
    
    print(f"Train IDs: {p_train_ids}")
    print(f"Val IDs  : {p_val_ids}")
    
    if p_train_ids == p_val_ids and len(p_train_ids) == 1:
        print("Plurality normalization works!")
    else:
        print("Plurality normalization FAILED or card missing.")

if __name__ == "__main__":
    main()

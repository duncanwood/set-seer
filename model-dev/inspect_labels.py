import torch
import os
from torch.utils.data import DataLoader
from set_seer_dataset import SetSeerDataset

def main():
    card_dir = "model-dev/data/train"
    dtd_dir = "model-dev/data/dtd/images"
    
    if not os.path.exists(card_dir) or not os.path.exists(dtd_dir):
        print("Data directories not found. Please run from project root.")
        return

    dataset = SetSeerDataset(
        card_dir=card_dir,
        dtd_dir=dtd_dir,
        img_size=640,
        epoch_size=32,
        min_cards=1,
        max_cards=15
    )

    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=SetSeerDataset.collate_fn)

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes (dataset): {len(dataset.class_to_idx)}")
    
    # Get one batch
    batch = next(iter(loader))
    
    print("\n--- Batch Inspection ---")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k:12}: {v.shape} {v.dtype}")
        elif isinstance(v, list):
            print(f"{k:12}: list (length {len(v)})")
        else:
            print(f"{k:12}: {type(v)}")

    # Check labels content
    if 'cls' in batch and len(batch['cls']) > 0:
        print("\n--- Labels Sample ---")
        print(f"Batch Index: {batch['batch_idx'][:10]}")
        print(f"Classes    : {batch['cls'][:10].flatten()}")
        print(f"BBoxes     : {batch['bboxes'][:10]}")
        
        # Check ranges
        cls_min = batch['cls'].min().item()
        cls_max = batch['cls'].max().item()
        print(f"\nClass ID Range: [{cls_min}, {cls_max}]")
        
        box_min = batch['bboxes'].min().item()
        box_max = batch['bboxes'].max().item()
        print(f"BBox Coord Range: [{box_min}, {box_max}]")
        
        if box_max > 1.0 or box_min < 0.0:
            print("WARNING: BBox coordinates should be normalized [0, 1].")
        else:
            print("SUCCESS: BBox coordinates are normalized.")
            
    else:
        print("\nNo labels found in this batch (unlikely with min_cards=1).")

if __name__ == "__main__":
    main()

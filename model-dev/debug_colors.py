from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.data.dataset import YOLODataset
from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import os
import shutil

class RedDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self.img_size = 640
        self.len = 4

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # Create solid RED image in BGR (Standard OpenCV)
        # Red = (0, 0, 255)
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        img[:] = (0, 0, 255) 
        
        # Convert to RGB for YOLO (as implemented in SetSeerDataset)
        img_rgb = img[:, :, ::-1].transpose(2, 0, 1)
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_rgb)) # uint8 0-255
        
        # Dummy box [class, x, y, w, h] center
        label = np.array([[0, 0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
        
        return {
            'img': img_tensor,
            'cls': torch.tensor(label[:, 0:1]),
            'bboxes': torch.tensor(label[:, 1:]),
            'ori_shape': (640, 640),
            'resized_shape': (640, 640),
            'im_file': f"red_{index}.jpg",
            'ratio_pad': ((1.0, 1.0), (0.0, 0.0)),
            'batch_idx': torch.tensor([index]*len(label)) # Handled by collate usually, but simple here
        }

    @staticmethod
    def collate_fn(batch):
        # Minimal collate
        new_batch = {}
        new_batch['img'] = torch.stack([b['img'] for b in batch])
        new_batch['batch_idx'] = torch.cat([torch.full((len(b['cls']),), i) for i, b in enumerate(batch)])
        new_batch['cls'] = torch.cat([b['cls'] for b in batch])
        new_batch['bboxes'] = torch.cat([b['bboxes'] for b in batch])
        # Add required keys for plotting
        new_batch['im_file'] = [b['im_file'] for b in batch]
        new_batch['ori_shape'] = [b['ori_shape'] for b in batch]
        new_batch['resized_shape'] = [b['resized_shape'] for b in batch]
        new_batch['ratio_pad'] = [b['ratio_pad'] for b in batch]
        return new_batch

class DebugValidator(DetectionValidator):
    def build_dataset(self, img_path, mode="val", batch=None):
        return RedDataset()

class DebugTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        return RedDataset()
    def get_validator(self):
        return DebugValidator(args=self.args)

def analyze_image(path):
    if not os.path.exists(path):
        print(f"MISSING: {path}")
        return
    img = cv2.imread(path)
    # Check average color
    mean_bgr = img.mean(axis=(0,1))
    print(f"File: {path}")
    print(f"  Mean BGR: {mean_bgr}")
    if mean_bgr[2] > mean_bgr[0]:
        print("  -> Predominantly RED (Correct for BGR file viewed as image)")
    else:
        print("  -> Predominantly BLUE (Inverted!)")

def run_test():
    # Clean previous
    if os.path.exists("debug_colors"):
        shutil.rmtree("debug_colors")
    
    # Create dummy yaml
    with open("debug_colors.yaml", "w") as f:
        f.write("train: .\nval: .\nnc: 1\nnames: ['red']")
    
    # 1. Run Trainer plotting
    print("\n--- Testing Trainer Plotting ---")
    trainer = DebugTrainer(overrides={'model':'yolov8n.pt', 'data':'debug_colors.yaml', 'epochs':1, 'imgsz':640, 'project':'debug_colors', 'name':'train_run', 'plots':True})
    # We can hack to just plot
    # But usually just training 1 epoch is fastest robustness check
    try:
        trainer.train()
    except Exception as e:
        print(f"Training crashed (expected if dummy model fails): {e}")

    # 2. Run Validator plotting
    print("\n--- Testing Validator Plotting ---")
    validator = DebugValidator(args={'model':'yolov8n.pt', 'data':'debug_colors.yaml', 'imgsz':640, 'project':'debug_colors', 'name':'val_run', 'plots':True})
    validator(model='yolov8n.pt')

    # 3. Analyze
    print("\n--- Analysis ---")
    # Trainer should produce debug_colors/train_run/train_batch0.jpg
    analyze_image("debug_colors/train_run/train_batch0.jpg")
    
    # Validator should produce debug_colors/val_run/val_batch0_labels.jpg
    analyze_image("debug_colors/val_run/val_batch0_labels.jpg")

if __name__ == "__main__":
    run_test()

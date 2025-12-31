from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import DEFAULT_CFG
from set_seer_dataset import SetSeerDataset
import torch

class SetSeerValidator(DetectionValidator):
    def build_dataset(self, img_path, mode="val", batch=None):
        return SetSeerDataset(
            card_dir="model-dev/data/train", # Use training cards for now
            dtd_dir="model-dev/data/dtd/images",
            img_size=640,
            epoch_size=10, # Short epoch
            min_cards=1,
            max_cards=5
        )

def run_debug_val():
    args = dict(
        model="yolov8n.pt",
        data="model-dev/set_seer_data.yaml",
        imgsz=640,
        batch=4,
        plots=True,
        save_json=False,
        name="debug_val",
        project="model-dev/debug_runs"
    )
    
    validator = SetSeerValidator(args=args)
    validator(model="yolov8n.pt")

if __name__ == "__main__":
    run_debug_val()

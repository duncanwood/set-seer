from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from set_seer_dataset import SetSeerDataset
import os
import argparse

class SetSeerValidator(DetectionValidator):
    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Builds the custom SetSeerDataset for validation.
        """
        possible_dtd = [
            "data/dtd/images",
            "model-dev/data/dtd/images",
            "../data/dtd/images"
        ]
        dtd_dir = None
        for p in possible_dtd:
            if os.path.exists(p):
                dtd_dir = p
                break
        
        if dtd_dir is None:
             raise FileNotFoundError("Could not find dtd/images directory.")
             
        # Validation epoch size
        epoch_size = 100 
        
        # Check for custom bgr flag in args
        bgr = getattr(self.args, 'bgr', False)

        return SetSeerDataset(
            card_dir=img_path, # Passed from data.yaml usually, but here we override or use CLI
            dtd_dir=dtd_dir,
            img_size=self.args.imgsz,
            epoch_size=epoch_size,
            min_cards=1,
            bgr=bgr
        )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to model weight")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--bgr", action="store_true", help="Use BGR output (fixes inversion if plotting assumes BGR)")
    args = parser.parse_args()

    # Define args for validator
    val_args = dict(
        model=args.model,
        data="model-dev/set_seer_data.yaml", # Config created by train script
        imgsz=args.imgsz,
        batch=args.batch,
        plots=True,
        save_json=False,
        project="set-seer-runs",
        name="validation_run",
        bgr=args.bgr # Pass custom arg
    )
    
    # ...
    
    validator = SetSeerValidator(args=val_args)
    validator(model=args.model)

if __name__ == "__main__":
    main()

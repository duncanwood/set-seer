import os
import copy
import yaml
import glob
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.data.utils import check_det_dataset
from set_seer_dataset import SetSeerDataset

class SetSeerValidator(DetectionValidator):
    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Builds the custom SetSeerDataset for validation.
        """
         # Hardcoded path to background images (DTD) - same logic as Trainer
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
             
        # Extract master classes from trainer config
        master_classes = list(self.data['names'].values())

        # Validation epoch size
        epoch_size = 100 

        # Check for custom bgr flag in args
        bgr = getattr(self.args, 'bgr', False)
        
        return SetSeerDataset(
            card_dir=img_path,
            dtd_dir=dtd_dir,
            img_size=self.args.imgsz,
            epoch_size=epoch_size,
            min_cards=1,
            max_cards=15,
            bgr=bgr,
            master_classes=master_classes
        )

class SetSeerTrainer(DetectionTrainer):
    def get_validator(self):
        """Returns a SetSeerValidator for validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return SetSeerValidator(self.test_loader, save_dir=self.save_dir, args=copy.copy(self.args))

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Builds the custom SetSeerDataset.
        
        Args:
            img_path (str): Path to the image folder (from data.yaml).
                            For this custom setup, it points to the card assets.
            mode (str): "train" or "val".
            batch (int): Batch size (unused by dataset init but passed by framework).
        """
        # img_path will be whatever is in data.yaml for 'train' or 'val'.
        # We assume it points to the directory containing class folders (e.g., data/train)
        
        # Hardcoded path to background images (DTD)
        # Assuming script is run from project root or model-dev.
        # We try to locate it relative to common locations.
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
            raise FileNotFoundError("Could not find dtd/images directory. Please ensure it exists.")

        # Extract master classes from trainer config
        master_classes = list(self.data['names'].values())

        # Determine epoch size
        epoch_size = 1000 if mode == 'train' else 100
        
        # Consistent card limits
        min_cards = 1
        max_cards = 15 

        # Respect bgr flag from args
        bgr = getattr(self.args, 'bgr', False)

        dataset = SetSeerDataset(
            card_dir=img_path,
            dtd_dir=dtd_dir,
            img_size=self.args.imgsz,
            epoch_size=epoch_size,
            min_cards=min_cards,
            max_cards=max_cards,
            bgr=bgr
        )
        
        return dataset
    
    def plot_training_labels(self):
        """
        Override to skip plotting training labels as our dataset is dynamic 
        and does not have a static 'labels' attribute.
        """
        pass
    
    def plot_training_samples(self, batch, ni):
        """
        Override to ensure we can plot samples if needed, but default might work
        if batch structure is correct. We'll leave default for now.
        """
        super().plot_training_samples(batch, ni)

    # We might need to override get_dataloader if standard one fails with our dict,
    # but let's try default first. 
    # v8 default dataloader uses a custom collate that handles the batch dicts.

def create_data_yaml(output_path="model-dev/custom_data.yaml"):
    """
    scans model-dev/data/train to find classes and writes a data.yaml
    """
    train_dir = "model-dev/data/train"
    val_dir = "model-dev/data/val"
    
    if not os.path.exists(train_dir):
        # Fallback if running from proper root
        train_dir = "data/train"
        val_dir = "data/val"
    
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    # We use the raw folder names for the yaml 'names' dict to keep it simple,
    # but SetSeerDataset will normalize them internally when matching.
    # However, to be extra safe, let's make sure they are unique when normalized.
     
    # Create the config dict
    data_config = {
        'path': os.path.abspath("."), # Base path
        'train': os.path.abspath(train_dir),
        'val': os.path.abspath(val_dir),
        'names': {i: name for i, name in enumerate(classes)},
        'nc': len(classes)
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(data_config, f)
    
    return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--bgr", action="store_true", help="Use BGR output (fixes inversion if plotting assumes BGR)")
    cli_args = parser.parse_args()

    # 1. Generate Config
    yaml_path = create_data_yaml("model-dev/set_seer_data.yaml")
    print(f"Created config at {yaml_path}")
    
    # 2. Initialize Trainer
    model = YOLO("yolov8n.pt") 
    
    # Args for training
    args = dict(
        model="yolov8n.pt",
        data=yaml_path,
        epochs=cli_args.epochs, 
        imgsz=cli_args.imgsz,
        batch=cli_args.batch, 
        device="mps", # Apple Silicon
        project="set-seer-runs",
        name="custom_aug_run",
        plots=True,
        bgr=cli_args.bgr
    )
    
    # Define callback to save validation plots per epoch
    def save_epoch_plots(trainer):
        # trainer.epoch is current epoch (0-indexed)
        # trainer.save_dir is the run directory
        # Plots are usually named 'val_batch0_pred.jpg' etc.
        import glob
        import shutil
        import os
        
        # files = glob.glob(os.path.join(trainer.save_dir, "val_batch*_pred.jpg"))
        # Using glob in save_dir
        
        # We want to wait until validation is done. 
        # on_fit_epoch_end is called after validation.
        
        # Check if any val plot exists and copy it
        epoch = trainer.epoch + 1
        src_pattern = os.path.join(trainer.save_dir, "val_batch*_pred.jpg")
        for src in glob.glob(src_pattern):
            basename = os.path.basename(src)
            # Example: val_batch0_pred.jpg -> epoch_1_val_batch0_pred.jpg
            dst = os.path.join(trainer.save_dir, f"epoch_{epoch}_{basename}")
            # Only copy if we haven't already (though unlikely in same epoch)
            if not os.path.exists(dst):
                 shutil.copy(src, dst)
                 
    trainer = SetSeerTrainer(overrides=args)
    trainer.add_callback("on_fit_epoch_end", save_epoch_plots)
    trainer.train()

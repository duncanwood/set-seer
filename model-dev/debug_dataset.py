import cv2
import os
import logging
import numpy as np
from set_seer_dataset import SetSeerDataset

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def test_generation():
    logging.info("Starting dataset test...")
    
    # Use paths relative to project root or adjusted for where script runs
    # Assuming running from set-seer root
    card_dir = "model-dev/data/train"
    dtd_dir = "model-dev/data/dtd/images"
    
    if not os.path.exists(card_dir):
        # try without model-dev prefix
        card_dir = "data/train"
        dtd_dir = "data/dtd/images"

    try:
        dataset = SetSeerDataset(card_dir, dtd_dir, img_size=640, epoch_size=10, min_cards=1, max_cards=5)
    except FileNotFoundError as e:
        logging.error(f"Setup failed: {e}")
        return

    # Get one item
    logging.info("Fetching item 0...")
    data = dataset[0]
    
    # Inspect the output dict
    img_tensor = data['img']
    logging.info(f"Output Img Tensor Shape: {img_tensor.shape}")
    logging.info(f"Output Img Tensor ToType: {img_tensor.dtype}")
    logging.info(f"Output Img Tensor Min: {img_tensor.min()}, Max: {img_tensor.max()}")
    
    # Convert back to numpy to view
    # Tensor is (C, H, W) RGB uint8
    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0) # H, W, C
    # img_np = (img_np * 255).astype(np.uint8) # Already uint8
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    output_path = "model-dev/debug_output_test.jpg"
    cv2.imwrite(output_path, img_np)
    logging.info(f"Saved debug image to {output_path}")
    
    # Also save the raw background from dataset if possible?
    # We can't access internal local vars of __getitem__ easily without modifying it.
    
if __name__ == "__main__":
    test_generation()

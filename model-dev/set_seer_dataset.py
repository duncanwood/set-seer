import os
import random
import cv2
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from ultralytics.data.dataset import YOLODataset


SET_CARDS_MASTER = [
    '1-green-empty-diamond', '1-green-empty-oval', '1-green-empty-squiggle', 
    '1-green-solid-diamond', '1-green-solid-oval', '1-green-solid-squiggle', 
    '1-green-striped-diamond', '1-green-striped-oval', '1-green-striped-squiggle', 
    '1-purple-empty-diamond', '1-purple-empty-oval', '1-purple-empty-squiggle', 
    '1-purple-solid-diamond', '1-purple-solid-oval', '1-purple-solid-squiggle', 
    '1-purple-striped-diamond', '1-purple-striped-oval', '1-purple-striped-squiggle', 
    '1-red-empty-diamond', '1-red-empty-oval', '1-red-empty-squiggle', 
    '1-red-solid-diamond', '1-red-solid-oval', '1-red-solid-squiggle', 
    '1-red-striped-diamond', '1-red-striped-oval', '1-red-striped-squiggle', 
    '2-green-empty-diamond', '2-green-empty-oval', '2-green-empty-squiggle', 
    '2-green-solid-diamond', '2-green-solid-oval', '2-green-solid-squiggle', 
    '2-green-striped-diamond', '2-green-striped-oval', '2-green-striped-squiggle', 
    '2-purple-empty-diamond', '2-purple-empty-oval', '2-purple-empty-squiggle', 
    '2-purple-solid-diamond', '2-purple-solid-oval', '2-purple-solid-squiggle', 
    '2-purple-striped-diamond', '2-purple-striped-oval', '2-purple-striped-squiggle', 
    '2-red-empty-diamond', '2-red-empty-oval', '2-red-empty-squiggle', 
    '2-red-solid-diamond', '2-red-solid-oval', '2-red-solid-squiggle', 
    '2-red-striped-diamond', '2-red-striped-oval', '2-red-striped-squiggle', 
    '3-green-empty-diamond', '3-green-empty-oval', '3-green-empty-squiggle', 
    '3-green-solid-diamond', '3-green-solid-oval', '3-green-solid-squiggle', 
    '3-green-striped-diamond', '3-green-striped-oval', '3-green-striped-squiggle', 
    '3-purple-empty-diamond', '3-purple-empty-oval', '3-purple-empty-squiggle', 
    '3-purple-solid-diamond', '3-purple-solid-oval', '3-purple-solid-squiggle', 
    '3-purple-striped-diamond', '3-purple-striped-oval', '3-purple-striped-squiggle', 
    '3-red-empty-diamond', '3-red-empty-oval', '3-red-empty-squiggle', 
    '3-red-solid-diamond', '3-red-solid-oval', '3-red-solid-squiggle', 
    '3-red-striped-diamond', '3-red-striped-oval', '3-red-striped-squiggle'
]


class SetSeerDataset(Dataset):
    """
    Custom Dataset for Set Seer that generates images on the fly using
    backgrounds and card assets.
    """
    def __init__(self, card_dir, dtd_dir, img_size=640, epoch_size=1000, 
                 min_cards=1, max_cards=20, bgr=False, master_classes=None):
        self.img_size = img_size
        self.epoch_size = epoch_size
        self.min_cards = min_cards
        self.max_cards = max_cards
        self.bgr_output = bgr
        
        # Load card data
        if not os.path.isdir(card_dir):
            raise FileNotFoundError(f"Card directory not found at {card_dir}")
        self.card_dataset = datasets.ImageFolder(card_dir)
        local_samples = self.card_dataset.samples
        local_class_to_idx = self.card_dataset.class_to_idx # {'folder': local_idx}
        
        # Default to master classes if none provided
        if master_classes is None:
            master_classes = SET_CARDS_MASTER
            
        # master_classes is a list of strings [class0, class1, ...]
        # normalized
        master_class_to_idx = {self.normalize_name(name): i for i, name in enumerate(master_classes)}
        self.class_to_idx = {name: i for i, name in enumerate(master_classes)}
        
        # Build local to master mapping
        local_to_master = {}
        for local_name, local_idx in local_class_to_idx.items():
            norm_name = self.normalize_name(local_name)
            if norm_name in master_class_to_idx:
                local_to_master[local_idx] = master_class_to_idx[norm_name]
            else:
                print(f"Warning: Local folder '{local_name}' (normalized: '{norm_name}') not found in master classes.")
        
        # Remap samples
        self.card_samples = []
        for path, local_idx in local_samples:
            if local_idx in local_to_master:
                self.card_samples.append((path, local_to_master[local_idx]))
        
        if not self.card_samples:
                raise ValueError(f"No valid cards found in {card_dir} that match master classes.")
        
        # Load background paths
        if not os.path.isdir(dtd_dir):
            raise FileNotFoundError(f"Backgrounds directory not found at {dtd_dir}")
        self.bg_paths = glob.glob(os.path.join(dtd_dir, '**', '*.jpg'), recursive=True)
        
        # Augmentation parameters
        self.min_crop = 2
        self.max_crop = 10
        self.max_off_frame = 0.15
        self.max_occlusion = 0.15
        self.max_occlusion = 0.15
        self.max_attempts = 100

    @staticmethod
    def normalize_name(name):
        """Normalize folder names: lowercase, rstrip 's', replace hyphens/underscores."""
        n = name.lower().strip()
        n = n.replace('_', '-')
        # Remove trailing 's' only if it seems to be plural (diamonds -> diamond)
        # But be careful with words that naturally end in s if any.
        # For Set cards, the properties are: empty, solid, striped, diamond, oval, squiggle.
        # None of these naturally end in 's' except plural.
        if n.endswith('s'):
            n = n[:-1]
        return n

    @staticmethod
    def collate_fn(batch):
        new_batch = {}
        # Stack images
        imgs = [b['img'] for b in batch]
        new_batch['img'] = torch.stack(imgs, 0)
        
        # Handle labels
        cls_list = [b['cls'] for b in batch]
        bboxes_list = [b['bboxes'] for b in batch]
        
        # Add batch index for each box
        batch_idx = []
        for i, c in enumerate(cls_list):
            n = len(c)
            # Use float32 to allow concatenation with targets later
            batch_idx.append(torch.full((n,), i, dtype=torch.float32))
            
        if batch_idx:
            new_batch['batch_idx'] = torch.cat(batch_idx, 0)
            new_batch['cls'] = torch.cat(cls_list, 0)
            new_batch['bboxes'] = torch.cat(bboxes_list, 0)
        else:
             new_batch['batch_idx'] = torch.zeros((0,), dtype=torch.float32)
             new_batch['cls'] = torch.zeros((0, 1), dtype=torch.float32)
             new_batch['bboxes'] = torch.zeros((0, 4), dtype=torch.float32)

        # Pass through other keys
        new_batch['ori_shape'] = [b['ori_shape'] for b in batch]
        new_batch['resized_shape'] = [b['resized_shape'] for b in batch]
        new_batch['im_file'] = [b['im_file'] for b in batch]
        
        # Add default ratio_pad if missing (needed for validation)
        if 'ratio_pad' in batch[0]:
            new_batch['ratio_pad'] = [b['ratio_pad'] for b in batch]
        else:
            # Default to no padding/resizing ((1.0, 1.0), (0.0, 0.0))
            new_batch['ratio_pad'] = [((1.0, 1.0), (0.0, 0.0)) for _ in batch]
        
        return new_batch

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, index):
        
        # 1. Select and Resize Background (Load as RGB)
        bg_path = random.choice(self.bg_paths)
        background = cv2.imread(bg_path)
        if background is None:
            background = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        
        background = cv2.resize(background, (self.img_size, self.img_size))
        
        placed_cards_data = [] # Stores (mask, pixel_count)
        yolo_labels = [] # Stores [class_id, x, y, w, h] (normalized)
        
        num_cards = random.randint(self.min_cards, self.max_cards)
        bg_h, bg_w = background.shape[:2]

        for _ in range(num_cards):
            # Select random card (Load as RGBA/RGB)
            card_path, card_class_id = random.choice(self.card_samples)
            card_img = cv2.imread(card_path, cv2.IMREAD_UNCHANGED)
            if card_img is None: continue
            
            # Convert to RGBA
            if card_img.shape[2] == 3:
                # BGR -> RGBA
                card_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2RGBA)
            elif card_img.shape[2] == 4:
                # BGRA -> RGBA
                card_img = cv2.cvtColor(card_img, cv2.COLOR_BGRA2RGBA)
            
            # Random Crop (on RGBA)
            h, w, _ = card_img.shape
            crop_top = random.randint(self.min_crop, self.max_crop)
            crop_bottom = random.randint(self.min_crop, self.max_crop)
            crop_left = random.randint(self.min_crop, self.max_crop)
            crop_right = random.randint(self.min_crop, self.max_crop)
            
            if h > crop_top + crop_bottom and w > crop_left + crop_right:
                card_img = card_img[crop_top:h-crop_bottom, crop_left:w-crop_right]
                
            # Apply transformations (works on any channels, returns RGBA content + Mask)
            transformed_card, mask = self.apply_random_transformations(card_img)
            if transformed_card is None: continue
            
            # Find Valid Placement
            placement = self.find_valid_placement(background, transformed_card, mask, placed_cards_data)
            
            if placement:
                # placement contains: (dest_x1, dest_y1, dest_x2, dest_y2, src_x1, src_y1, src_x2, src_y2, new_card_mask_canvas, total_card_pixels)
                dx1, dy1, dx2, dy2, sx1, sy1, sx2, sy2, mask_canvas, total_pixels = placement
                
                # Composite Image
                card_crop = transformed_card[sy1:sy2, sx1:sx2]
                mask_crop = mask[sy1:sy2, sx1:sx2]
                roi = background[dy1:dy2, dx1:dx2] # RGB
                
                alpha = mask_crop.astype(float) / 255.0
                alpha = np.stack([alpha]*3, axis=-1)
                
                foreground = card_crop[:, :, :3].astype(float) # RGB
                foreground = cv2.multiply(alpha, foreground)
                
                # Check shapes for safety
                if foreground.shape[:2] != roi.shape[:2]: continue

                background_part = cv2.multiply(1.0 - alpha, roi.astype(float))
                blended_roi = cv2.add(foreground, background_part)
                
                background[dy1:dy2, dx1:dx2] = blended_roi.astype(np.uint8)
                
                placed_cards_data.append((mask_canvas, total_pixels))
                
                # Compute Bounding Box
                contours, _ = cv2.findContours(mask_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    all_points = np.concatenate(contours, axis=0)
                    x, y, w, h = cv2.boundingRect(all_points)
                    if w > 0 and h > 0:
                        x_center = (x + w / 2) / bg_w
                        y_center = (y + h / 2) / bg_h
                        norm_w = w / bg_w
                        norm_h = h / bg_h
                        yolo_labels.append([card_class_id, x_center, y_center, norm_w, norm_h])

        return self.format_for_yolo(background, yolo_labels, index)

    def format_for_yolo(self, img, labels, index):
        # Img is RGB (H, W, 3) 0-255 uint8 due to explicit conversion above
        
        if self.bgr_output:
            img = img[:, :, ::-1] # RGB to BGR
            
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img) 

        count = len(labels)
        if count > 0:
            labels_array = np.array(labels, dtype=np.float32)
            # YOLO expects: batch_idx, class, x, y, w, h if running via collate.
            # But normally Dataset returns dict with 'cls' and 'bboxes' separately or 'labels' tensor.
            # Ultralytics v8 Dataset returns:
            # {
            #   'img': (H,W,C), 
            #   'cls': (N,1), 
            #   'bboxes': (N,4), # xywh or xyxy
            #   'batch_idx': batch_idx (added by collate usually)
            # }
            # Check BaseDataset. The trainer expects 'cls' (N,1) and 'bboxes' (N,4).
            
            cls = torch.tensor(labels_array[:, 0:1], dtype=torch.float32)
            bboxes = torch.tensor(labels_array[:, 1:], dtype=torch.float32)
        else:
            cls = torch.zeros((0, 1), dtype=torch.float32)
            bboxes = torch.zeros((0, 4), dtype=torch.float32)

        # Labels usually include batch_index in the 0th column in older YOLO logic, 
        # but in formatting for build_dataset/v8, we typically provide Components.
        
        # We also usually return 'ori_shape', 'resized_shape', 'img_file'.
        
        return {
            'img': img, 
            'cls': cls,
            'bboxes': bboxes,
            'ori_shape': (img.shape[1], img.shape[2]),
            'resized_shape': (img.shape[1], img.shape[2]),
            'im_file': f"synth_{index}.jpg",
            # 'ratio_pad': (1.0, 1.0), # Optional
        }

    def apply_random_transformations(self, card_img):
        rows, cols = card_img.shape[:2]
        bg_w, bg_h = self.img_size, self.img_size
        
        scale_factor = random.uniform(0.04, 0.35)
        new_w = int(bg_w * scale_factor)
        new_h = int(new_w / (cols / rows)) if cols > 0 and rows > 0 else 0
        if new_w == 0 or new_h == 0: return None, None
        
        card_img = cv2.resize(card_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rows, cols = card_img.shape[:2]
        
        angle = random.uniform(-180, 180)
        M_rot = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        
        pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
        px_shift = 0.1 * cols
        pts2 = np.float32([
            [random.uniform(-px_shift, px_shift), random.uniform(-px_shift, px_shift)],
            [cols - random.uniform(-px_shift, px_shift), random.uniform(-px_shift, px_shift)],
            [random.uniform(-px_shift, px_shift), rows - random.uniform(-px_shift, px_shift)],
            [cols - random.uniform(-px_shift, px_shift), rows - random.uniform(-px_shift, px_shift)]
        ])
        M_persp = cv2.getPerspectiveTransform(pts1, pts2)
        
        cos, sin = np.abs(M_rot[0, 0]), np.abs(M_rot[0, 1])
        new_cols = int((rows * sin) + (cols * cos))
        new_rows = int((rows * cos) + (cols * sin))
        
        M_rot[0, 2] += (new_cols / 2) - (cols / 2)
        M_rot[1, 2] += (new_rows / 2) - (rows / 2)
        
        rotated_card = cv2.warpAffine(card_img, M_rot, (new_cols, new_rows))
        transformed_card = cv2.warpPerspective(rotated_card, M_persp, (new_cols, new_rows))
        
        if card_img.shape[2] == 4:
            mask = card_img[:, :, 3]
        else:
            gray_card = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_card, 1, 255, cv2.THRESH_BINARY)
            
        rotated_mask = cv2.warpAffine(mask, M_rot, (new_cols, new_rows))
        transformed_mask = cv2.warpPerspective(rotated_mask, M_persp, (new_cols, new_rows))
        
        kernel = np.ones((3, 3), np.uint8)
        eroded_mask = cv2.erode(transformed_mask, kernel, iterations=1)
        
        return transformed_card, eroded_mask

    def find_valid_placement(self, background, transformed_card, mask, placed_cards_data):
        h_t, w_t = transformed_card.shape[:2]
        bg_h, bg_w = background.shape[:2]
        
        for _ in range(self.max_attempts):
            pos_x, pos_y = random.randint(-int(w_t*0.5), bg_w - int(w_t*0.5)), random.randint(-int(h_t*0.5), bg_h - int(h_t*0.5))
            
            new_card_mask_canvas = np.zeros((bg_h, bg_w), dtype=np.uint8)
            dest_x1, dest_y1 = max(pos_x, 0), max(pos_y, 0)
            dest_x2, dest_y2 = min(pos_x + w_t, bg_w), min(pos_y + h_t, bg_h)
            
            src_x1, src_y1 = max(0, -pos_x), max(0, -pos_y)
            src_x2 = src_x1 + (dest_x2 - dest_x1)
            src_y2 = src_y1 + (dest_y2 - dest_y1)
            
            if dest_x2 <= dest_x1 or dest_y2 <= dest_y1: continue
            
            # Place mask on canvas to check overlaps
            new_card_mask_canvas[dest_y1:dest_y2, dest_x1:dest_x2] = mask[src_y1:src_y2, src_x1:src_x2]
            
            total_card_pixels = np.count_nonzero(mask)
            if total_card_pixels == 0: continue
            
            visible_card_pixels = np.count_nonzero(new_card_mask_canvas)
            off_frame_ratio = (total_card_pixels - visible_card_pixels) / total_card_pixels
            if off_frame_ratio > self.max_off_frame: continue
            
            occludes_too_much = False
            for placed_mask, placed_pixel_count in placed_cards_data:
                intersection = cv2.bitwise_and(new_card_mask_canvas, placed_mask)
                if placed_pixel_count > 0:
                    occlusion_of_old = np.count_nonzero(intersection) / placed_pixel_count
                    if occlusion_of_old > self.max_occlusion: 
                        occludes_too_much = True
                        break
            
            if not occludes_too_much:
                return (dest_x1, dest_y1, dest_x2, dest_y2, src_x1, src_y1, src_x2, src_y2, new_card_mask_canvas, total_card_pixels)
        
        return None

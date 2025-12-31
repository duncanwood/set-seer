import sys
import os
sys.path.append(os.getcwd())
# from model_dev.train_custom import create_data_yaml

# Fix import issue by renaming the import or adding path
# Actually since model-dev has hyphen, we can't import it as module easily.
# But train_custom.py is inside it.
# Let's just copy the logic or load from file.
# Easiest: The create_data_yaml function is standalone.

def create_data_config():
    train_path = os.path.abspath("model-dev/data/train")
    val_path = os.path.abspath("model-dev/data/val") # Or train if val doesn't exist
    yaml_path = "model-dev/set_seer_data.yaml"
    
    classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    classes.sort()
    
    content = f"""
train: {train_path}
val: {train_path} # Use train for val structure if needed, or val_path
nc: {len(classes)}
names: {classes}
"""
    with open(yaml_path, "w") as f:
        f.write(content.strip())
    print(f"Regenerated {yaml_path} with {len(classes)} classes.")
    print("First 5 classes:", classes[:5])

if __name__ == "__main__":
    create_data_config()

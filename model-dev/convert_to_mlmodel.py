from ultralytics import YOLO
import os

def convert():
    # Path to the source weights
    weights_path = "set-seer-runs/custom_aug_run32/weights/best-2.pt"
    
    if not os.path.exists(weights_path):
        print(f"Error: Could not find weights at {weights_path}")
        return

    print(f"Loading model from {weights_path}...")
    model = YOLO(weights_path)

    print("Exporting to CoreML...")
    # NMS is often required for CoreML deployment in vision apps
    # we export with nms=True and format='coreml'
    # Ultralytics will embed class names automatically if they are in the model metadata.
    export_path = model.export(format="coreml", nms=False)

    print(f"Successfully exported model to: {export_path}")

if __name__ == "__main__":
    convert()

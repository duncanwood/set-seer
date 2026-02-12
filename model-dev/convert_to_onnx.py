"""
Export Set Seer YOLO model to ONNX format for web deployment.

Usage:
    python convert_to_onnx.py [--weights PATH] [--output PATH] [--imgsz SIZE]
"""
from ultralytics import YOLO
import argparse
import os
import shutil


def convert(weights_path: str, output_path: str, imgsz: int = 640):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Could not find weights at {weights_path}")

    print(f"Loading model from {weights_path}...")
    model = YOLO(weights_path)

    print(f"Exporting to ONNX (imgsz={imgsz})...")
    # Export without NMS — NMS is handled in JavaScript post-processing
    # simplify=True runs onnx-simplifier to optimize for inference
    export_path = model.export(format="onnx", imgsz=imgsz, simplify=True, nms=False)

    print(f"Exported to: {export_path}")

    # Move to desired output path if specified
    if output_path and export_path != output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy2(export_path, output_path)
        print(f"Copied to: {output_path}")

    return output_path or export_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX")
    parser.add_argument(
        "--weights",
        default="set-seer-runs/custom_aug_run32/weights/best-2.pt",
        help="Path to .pt weights file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for .onnx file (default: same directory as weights)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    args = parser.parse_args()
    convert(args.weights, args.output, args.imgsz)

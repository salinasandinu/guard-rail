"""
YOLOv8 Guardrail Segmentation Model Inference Script
Run predictions on images using the trained guardrail detection model.
"""

from ultralytics import YOLO
import argparse
from pathlib import Path
import cv2


def predict(
    model_path: str,
    source: str,
    imgsz: int = 512,
    conf: float = 0.25,
    iou: float = 0.7,
    device: str = "0",
    save: bool = True,
    save_txt: bool = False,
    save_crop: bool = False,
    show: bool = False,
    project: str = "runs/predict",
    name: str = "guardrail_pred",
):
    """
    Run inference with trained YOLOv8 segmentation model.

    Args:
        model_path: Path to trained model weights (.pt file)
        source: Image/video source (file, directory, URL, webcam)
        imgsz: Input image size
        conf: Confidence threshold
        iou: IoU threshold for NMS
        device: CUDA device
        save: Save prediction images
        save_txt: Save results to text files
        save_crop: Save cropped predictions
        show: Display results
        project: Project directory for saving results
        name: Prediction run name
    """
    # Load trained model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Run inference
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        save=save,
        save_txt=save_txt,
        save_crop=save_crop,
        show=show,
        project=project,
        name=name,
        # Segmentation specific
        retina_masks=True,  # High-quality masks
    )

    print(f"\nPrediction complete!")
    if save:
        print(f"Results saved to: {project}/{name}")

    return results


def predict_single_image(model_path: str, image_path: str, conf: float = 0.25):
    """
    Predict on a single image and return results.
    
    Args:
        model_path: Path to trained model weights
        image_path: Path to input image
        conf: Confidence threshold
    
    Returns:
        Annotated image and detection results
    """
    model = YOLO(model_path)
    results = model(image_path, conf=conf)
    
    # Get annotated image
    annotated_image = results[0].plot()
    
    # Print detection info
    for result in results:
        if result.masks is not None:
            print(f"Found {len(result.masks)} guardrail(s)")
            for i, (box, mask) in enumerate(zip(result.boxes, result.masks)):
                conf = box.conf.item()
                print(f"  Guardrail {i+1}: confidence = {conf:.2f}")
        else:
            print("No guardrails detected")
    
    return annotated_image, results


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Guardrail Detection Inference")
    
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained model weights (e.g., runs/train/guardrail_seg/weights/best.pt)"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Image/video source (file, directory, URL, or webcam index)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=512,
        help="Input image size"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--iou", type=float, default=0.7,
        help="IoU threshold for NMS"
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="CUDA device (0, 0,1, cpu)"
    )
    parser.add_argument(
        "--save", action="store_true", default=True,
        help="Save prediction images"
    )
    parser.add_argument(
        "--no-save", dest="save", action="store_false",
        help="Don't save prediction images"
    )
    parser.add_argument(
        "--save-txt", action="store_true",
        help="Save results to text files"
    )
    parser.add_argument(
        "--save-crop", action="store_true",
        help="Save cropped predictions"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display results"
    )
    parser.add_argument(
        "--project", type=str, default="runs/predict",
        help="Project directory"
    )
    parser.add_argument(
        "--name", type=str, default="guardrail_pred",
        help="Prediction run name"
    )

    args = parser.parse_args()

    predict(
        model_path=args.model,
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=args.save,
        save_txt=args.save_txt,
        save_crop=args.save_crop,
        show=args.show,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()

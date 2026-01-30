"""
YOLOv8 Guardrail Segmentation Model Training Script
Train a YOLOv8 instance segmentation model to detect guardrails.
"""

from ultralytics import YOLO
import argparse
from pathlib import Path


def train(
    data_yaml: str = "datasets/data.yaml",
    model_size: str = "s",
    epochs: int = 100,
    imgsz: int = 512,
    batch: int = 16,
    device: str = "cpu",
    project: str = "runs/train",
    name: str = "guardrail_seg",
    resume: bool = False,
):
    """
    Train YOLOv8 segmentation model for guardrail detection.

    Args:
        data_yaml: Path to data.yaml configuration file
        model_size: Model size - 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size (-1 for auto)
        device: CUDA device (e.g., '0' or '0,1' for multi-GPU, 'cpu' for CPU)
        project: Project directory for saving results
        name: Experiment name
        resume: Resume training from last checkpoint
    """
    # Select pretrained model based on size (segmentation models)
    model_name = f"yolov8{model_size}-seg.pt"
    
    print(f"Loading pretrained model: {model_name}")
    model = YOLO(model_name)

    # Training configuration
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        resume=resume,
        # Data augmentation (enabled by default, but explicitly set for clarity)
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,  # Good for segmentation
        # Training hyperparameters
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        # Save settings
        save=True,
        save_period=-1,  # Save only best and last
        plots=True,
        # Validation
        val=True,
    )

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {results.save_dir}")
    print(f"Best model: {results.save_dir}/weights/best.pt")
    
    # Print training metrics
    print(f"\n{'='*60}")
    print("TRAINING METRICS")
    print(f"{'='*60}")
    if hasattr(results, 'results_dict'):
        for key, value in results.results_dict.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    # Run validation on best model and print results
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS (Best Model)")
    print(f"{'='*60}")
    
    best_model_path = f"{results.save_dir}/weights/best.pt"
    best_model = YOLO(best_model_path)
    val_results = best_model.val(data=data_yaml, imgsz=imgsz, device=device)
    
    # Print validation metrics
    print(f"\n  Box Metrics:")
    print(f"    mAP50:      {val_results.box.map50:.4f}")
    print(f"    mAP50-95:   {val_results.box.map:.4f}")
    print(f"    Precision:  {val_results.box.mp:.4f}")
    print(f"    Recall:     {val_results.box.mr:.4f}")
    
    if hasattr(val_results, 'seg') and val_results.seg is not None:
        print(f"\n  Segmentation Metrics:")
        print(f"    mAP50:      {val_results.seg.map50:.4f}")
        print(f"    mAP50-95:   {val_results.seg.map:.4f}")
        print(f"    Precision:  {val_results.seg.mp:.4f}")
        print(f"    Recall:     {val_results.seg.mr:.4f}")
    
    print(f"\n{'='*60}")
    
    return results, val_results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 Guardrail Segmentation Model")
    
    parser.add_argument(
        "--data", type=str, default="datasets/data.yaml",
        help="Path to data.yaml file"
    )
    parser.add_argument(
        "--model", type=str, default="s", choices=["n", "s", "m", "l", "x"],
        help="Model size: n(ano), s(mall), m(edium), l(arge), x(large)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--imgsz", type=int, default=512,
        help="Input image size"
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size (-1 for auto)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="CUDA device (0, 0,1, cpu)"
    )
    parser.add_argument(
        "--project", type=str, default="runs/train",
        help="Project directory"
    )
    parser.add_argument(
        "--name", type=str, default="guardrail_seg",
        help="Experiment name"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint"
    )

    args = parser.parse_args()

    train(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

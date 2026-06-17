"""
Training Script for HKANet
---------------------------------------------------------
This script initializes the HKANet model topology and starts 
the training process. It utilizes argparse for cross-domain
dataset selection.

Usage:
    python train.py --dataset rdrd --epochs 100
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
from pathlib import Path
from ultralytics import YOLO
from ultralytics import settings

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HKANet Training Script")
    parser.add_argument("--dataset", type=str, required=True, choices=['rdrd', 'usrdd', 'rdd2020'], help="Target dataset for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image resolution")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size")
    parser.add_argument("--device", type=str, default='0', help="CUDA device ID, e.g., '0' or 'cpu'")
    args = parser.parse_args()

    # Dynamically resolve project root
    PROJECT_ROOT = Path(__file__).resolve().parent
    settings.update({'datasets_dir': str(PROJECT_ROOT)})
    
    # Base model architecture config
    model_config = PROJECT_ROOT / 'ultralytics' / 'cfg' / 'models' / 'YOLOv11-RDD' / 'yolov11-RDD.yaml'
    
    # Route paths based on the selected dataset
    if args.dataset == 'rdrd':
        data_yaml = PROJECT_ROOT / 'configs' / 'RDRD' / 'RDRD.yaml'
        run_name = 'HKANet_RDRD_Train'
    elif args.dataset == 'usrdd':
        data_yaml = PROJECT_ROOT / 'configs' / 'USRDD' / 'USRDD_1.yaml'
        run_name = 'HKANet_USRDD_Train'
    elif args.dataset == 'rdd2020':
        data_yaml = PROJECT_ROOT / 'configs' / 'RDD2020' / 'RDD2020_1.yaml'
        run_name = 'HKANet_RDD2020_Train'
    
    print(f"[*] Starting training pipeline for HKANet ({args.dataset.upper()})...")
    print(f"[*] Model Config: {model_config}")
    print(f"[*] Data Config:  {data_yaml}")
    
    if not model_config.exists() or not data_yaml.exists():
        print("[!] Critical Error: Configuration files not found. Please check paths.")
        raise SystemExit(1)

    # Initialize the model architecture
    model = YOLO(str(model_config))
    
    # Execute the training loop
    model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs, 
        batch=args.batch,
        optimizer="SGD",
        cos_lr=True,
        seed=0,
        mosaic=1,
        project=str(PROJECT_ROOT / "runs" / "train"),
        name=run_name,
        patience=30,
        device=args.device 
    )
    
    print("\n[✔] Training process finished successfully.")
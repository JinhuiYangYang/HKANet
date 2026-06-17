"""
Validation Script for HKANet
---------------------------------------------------------
This script loads a pre-trained HKANet model and validates
its performance on a single dataset split. It utilizes dynamic 
path resolution and argparse for Code Ocean deployment readiness.

Usage:
    python val.py --dataset rdd2020
    python val.py --dataset rdrd
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
from pathlib import Path
from ultralytics import YOLO
from ultralytics import settings

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HKANet Single Validation Script")
    parser.add_argument("--dataset", type=str, required=True, choices=['rdrd', 'usrdd', 'rdd2020'], help="Target dataset to evaluate")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image resolution (default: 1024)")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size (default: 16)")
    parser.add_argument("--device", type=str, default='0', help="CUDA device ID, e.g., '0' or 'cpu'")
    args = parser.parse_args()

    # Dynamically resolve project root
    PROJECT_ROOT = Path(__file__).resolve().parent
    settings.update({'datasets_dir': str(PROJECT_ROOT)})
    
    # Route paths based on the selected dataset
    if args.dataset == 'rdrd':
        weights_path = PROJECT_ROOT / 'weights' / 'RDRD' / 'best.pt'
        data_yaml = PROJECT_ROOT / 'configs' / 'RDRD' / 'RDRD.yaml'
        run_name = 'HKANet_RDRD_Val'
    elif args.dataset == 'usrdd':
        # Defaulting to Fold 1 for single validation demonstration
        weights_path = PROJECT_ROOT / 'weights' / 'USRDD' / 'dataset_1' / 'weights' / 'best.pt'
        data_yaml = PROJECT_ROOT / 'configs' / 'USRDD' / 'USRDD_1.yaml'
        run_name = 'HKANet_USRDD_Val'
    elif args.dataset == 'rdd2020':
        # Defaulting to Fold 1 for single validation demonstration
        weights_path = PROJECT_ROOT / 'weights' / 'RDD2020' / 'dataset_1' / 'weights' / 'best.pt'
        data_yaml = PROJECT_ROOT / 'configs' / 'RDD2020' / 'RDD2022_1.yaml'
        run_name = 'HKANet_RDD2020_Val'
    
    print(f"[*] Starting validation pipeline for HKANet ({args.dataset.upper()})...")
    print(f"[*] Weights: {weights_path}")
    print(f"[*] Config:  {data_yaml}")
    
    # Check if files exist to prevent engine crash
    if not weights_path.exists():
        print(f"[!] Critical Error: Weight file not found at {weights_path}")
        raise SystemExit(1)
    if not data_yaml.exists():
        print(f"[!] Critical Error: YAML config not found at {data_yaml}")
        raise SystemExit(1)

    # Initialize the model with the pre-trained weights
    model = YOLO(str(weights_path)) 
    
    # Execute the validation process
    model.val(
        data=str(data_yaml),
        split='val',      
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(PROJECT_ROOT / 'runs' / 'val'),
        name=run_name,
        device=args.device        
    )
    
    print("\n[✔] Validation process finished successfully.")
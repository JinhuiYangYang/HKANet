"""
Unified Evaluation Script for HKANet (Code Ocean Deployment Version)
-------------------------------------------------------------------
This script evaluates the HKANet model across different datasets (RDRD, USRDD, RDD2020).
It dynamically resolves relative paths to ensure reproducibility across different environments.
It supports both single-run evaluation (e.g., RDRD) and k-fold cross-validation (e.g., USRDD, RDD2020).

Usage:
    python evaluate.py --dataset rdrd --imgsz 1024 --batch 16 --device 0
"""
import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import re
from pathlib import Path
import numpy as np
from prettytable import PrettyTable
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info
from ultralytics import settings

def get_weight_size(path):
    """Calculate the file size of the model weights in MB."""
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

def evaluate_single_model(weights_path, yaml_path, project_dir, run_name, args):
    """
    Core evaluation function for a single model weight and dataset config.
    """
    print(f"\n[{run_name}] Starting Evaluation...")
    print(f"[*] Weights: {weights_path}")
    print(f"[*] Dataset: {yaml_path}")

    # Validate file existence before starting YOLO engine
    if not os.path.exists(weights_path):
        print(f"[!] Warning: Weight file not found at {weights_path}. Skipping.")
        return
    if not os.path.exists(yaml_path):
        print(f"[!] Warning: YAML configuration not found at {yaml_path}. Skipping.")
        return

    # Initialize model
    model = YOLO(str(weights_path)) 
    
    # Run validation
    result = model.val(
        data=str(yaml_path),
        split='val', 
        imgsz=args.imgsz,
        batch=args.batch,
        
        device=args.device,
        augment=True,       # Enabled by default for best metric performance (TTA)
        project=project_dir,
        name=run_name,
    )
        
    if model.task == 'detect':
        # Extract performance metrics
        model_names = list(result.names.values())
        preprocess_time = result.speed['preprocess']
        inference_time = result.speed['inference']
        postprocess_time = result.speed['postprocess']
        total_time = preprocess_time + inference_time + postprocess_time
        
        _, n_p, _, flops = model_info(model.model)
        
        # 1. Generate Model Architecture Info Table
        info_table = PrettyTable()
        info_table.title = "HKANet - Model Architecture Information"
        info_table.field_names = [
            "GFLOPs", "Parameters", "Pre-process (ms)", 
            "Inference (ms)", "Post-process (ms)", 
            "FPS (Total)", "FPS (Inference)", "Model Size"
        ]
        info_table.add_row([
            f'{flops:.1f}', f'{n_p:,}', 
            f'{preprocess_time:.3f}', f'{inference_time:.3f}', 
            f'{postprocess_time:.3f}', f'{1000 / total_time:.2f}', 
            f'{1000 / inference_time:.2f}', f'{get_weight_size(weights_path)} MB'
        ])
        print(info_table)

        # 2. Generate Detection Metrics Table
        metrics_table = PrettyTable()
        metrics_table.title = f"HKANet - Detection Metrics ({run_name})"
        metrics_table.field_names = [
            "Class Name", "Precision", "Recall", "F1-Score", 
            "mAP@0.5", "mAP@0.75", "mAP@0.5:0.95"
        ]
        
        # Populate class-specific metrics
        for idx, cls_name in enumerate(model_names):
            try:
                p_val = f"{result.box.p[idx]:.4f}"
                r_val = f"{result.box.r[idx]:.4f}"
                f1_val = f"{result.box.f1[idx]:.4f}"
                ap50_val = f"{result.box.ap50[idx]:.4f}"
                ap75_val = f"{result.box.all_ap[idx, 5]:.4f}"
                ap_val = f"{result.box.ap[idx]:.4f}"
            except IndexError:
                p_val = "0.0000"
                r_val = "0.0000"
                f1_val = "0.0000"
                ap50_val = "0.0000"
                ap75_val = "0.0000"
                ap_val = "0.0000"

            metrics_table.add_row([
                cls_name, p_val, r_val, f1_val, ap50_val, ap75_val, ap_val
            ])
            
        # Populate overall mean metrics
        metrics_table.add_row([
            "All (Mean)", 
            f"{result.results_dict['metrics/precision(B)']:.4f}", 
            f"{result.results_dict['metrics/recall(B)']:.4f}", 
            f"{np.mean(result.box.f1):.4f}", 
            f"{result.results_dict['metrics/mAP50(B)']:.4f}", 
            f"{np.mean(result.box.all_ap[:, 5]):.4f}", 
            f"{result.results_dict['metrics/mAP50-95(B)']:.4f}"
        ])
        print(metrics_table)

        # Export metrics to a text file for academic reporting
        save_file = result.save_dir / 'paper_metrics_report.txt'
        with open(save_file, 'w+', encoding='utf-8') as f:
            f.write(str(info_table) + '\n\n')
            f.write(str(metrics_table))
        
        print(f"\n[*] Metrics successfully exported to: {save_file}\n" + "-"*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HKANet Unified Evaluation Framework")
    parser.add_argument("--dataset", type=str, required=True, choices=['rdrd', 'usrdd', 'rdd2020'], help="Target dataset to evaluate")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image resolution (default: 1024)")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size (default: 16)")
    parser.add_argument("--device", type=str, default='0', help="CUDA device ID, e.g., '0' or 'cpu'")
    args = parser.parse_args()

    # Determine Project Root dynamically
    PROJECT_ROOT = Path(__file__).resolve().parent 
    
    # CRITICAL: Tell Ultralytics to use this project root as the base for all relative dataset paths in YAMLs
    settings.update({'datasets_dir': str(PROJECT_ROOT)})
    
    print("="*60)
    print(f"HKANet Evaluation Protocol Initialized")
    print(f"Project Root: {PROJECT_ROOT}")
    print("="*60)

    # ---------------------------------------------------------
    # Route 1: RDRD Dataset (Single Validation Run)
    # ---------------------------------------------------------
    if args.dataset == 'rdrd':
        # NOTE: Please adjust the exact filename of the weights/yaml if they differ in your folder
        w_path = PROJECT_ROOT / "weights" / "RDRD" / "best.pt"
        y_path = PROJECT_ROOT / "configs" / "RDRD" / "rdrd.yaml"
        project_out = str(PROJECT_ROOT / "runs" / "evaluate" / "RDRD")
        
        evaluate_single_model(w_path, y_path, project_out, "RDRD_Single_Run", args)

    # ---------------------------------------------------------
    # Route 2: USRDD & RDD2020 Datasets (5-Fold Cross Validation)
    # ---------------------------------------------------------
    elif args.dataset in ['usrdd', 'rdd2020']:
        dataset_upper = args.dataset.upper()
        weights_root = PROJECT_ROOT / "weights" / dataset_upper
        project_out = str(PROJECT_ROOT / "runs" / "evaluate" / dataset_upper)

        # Scan for all available weights in the dataset's subdirectories
        weight_files = sorted(weights_root.glob("*/weights/best.pt"))
        
        if not weight_files:
            print(f"[!] Critical Error: No 'best.pt' found inside {weights_root}")
            raise SystemExit(1)

        for w_path in weight_files:
            # Extract dataset index (e.g., from "dataset_1" -> 1)
            exp_folder_name = w_path.parent.parent.name
            match = re.search(r'dataset_?(\d+)$', exp_folder_name)
            
            if not match:
                print(f"[!] Warning: Cannot parse fold index from {exp_folder_name}. Skipping.")
                continue
            
            fold_idx = int(match.group(1))
            
            # NOTE: Assuming YAML configs are named like RDD2022_1.yaml or USRDD_1.yaml
            # Adjust the filename template below if your YAML files are named differently.
            if args.dataset == 'rdd2020':
                yaml_name = f"RDD2020_{fold_idx}.yaml"
            else:
                # Based on previous scripts, USRDD might use "RDD2022_United_States_{fold_idx}.yaml"
                yaml_name = f"USRDD_{fold_idx}.yaml" 
                
            y_path = PROJECT_ROOT / "configs" / dataset_upper / yaml_name
            run_name = f"Fold_{fold_idx}"

            evaluate_single_model(w_path, y_path, project_out, run_name, args)
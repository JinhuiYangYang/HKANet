# HKANet: An Efficient Software Framework for Cross-Domain Road Damage Detection

## 📝 Abstract

**HKANet** is a highly engineered, open-source deep learning framework explicitly designed for robust and real-time road damage detection (RDD) across diverse domains. By integrating customized dynamic attention mechanisms (including Mixed Kernel Attention and C2PSA modules) into a streamlined YOLO-based architecture, HKANet efficiently balances high-precision localization with computational constraints (16.5M parameters, 48.5 GFLOPs).

This repository provides a unified, highly cohesive API to guarantee strict reproducibility for cross-domain experiments spanning the **RDRD**, **USRDD**, and **RDD2020** datasets.

------

## 📂 Project Structure

The repository is structured following strict software engineering and Code Ocean deployment standards. All paths are resolved dynamically to ensure seamless cross-platform execution.

```
HKANet/
├── configs/            # Dataset configuration YAMLs (RDRD, USRDD, RDD2020)
├── datasets/           # Pre-split datasets (Images & Labels for 5-fold cross-validation)
├── runs/               # Automated output directory for logs, metrics, and figures
├── ultralytics/        # Customized core engine containing HKA, CKAConv, and C2PSA modules
├── weights/            # Pre-trained optimal weights (best.pt) for all datasets
├── evaluate.py         # Automated benchmark script (Rigorous 5-fold cross-validation)
├── tool.py             # Dataset aggregation and train/val splitting utility
├── train.py            # Training pipeline initialization
└── val.py              # Single-run quick validation script (Demo)
```



------

## ⚙️ Environment Setup

To ensure strict reproducibility, it is recommended to run this framework within a dedicated Conda environment.

```
bash
# 1. Create and activate a conda environment
conda create -n hkanet_env python=3.9
conda activate hkanet_env

# 2. Install PyTorch (Ensure CUDA compatibility with your hardware)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install required dependencies
pip install -r requirements.txt
```

------

## 🚀 Execution & Usage

The framework is operated via unified, parametrized command-line scripts. All paths are inherently relative to the project root.

### 1. Data Preparation (Optional)

If you are starting with raw data, `tool.py` is provided to seamlessly aggregate multiple folders and split them into rigorous `train` (80%) and `val` (20%) sets using a fixed random seed.

*Note: The provided datasets/ folder already contains the pre-split data required for exact academic reproduction.*

```
python tool.py
```

### 2. Quick Validation (Demo Run)

To rapidly verify the environment and model interface, execute the single-run validation script. This loads the pre-trained weights and evaluates a single fold of the specified dataset.

```
# Available datasets: rdrd, usrdd, rdd2020
python val.py --dataset rdd2020
```

### 3. Rigorous Benchmark (Cross-Domain Evaluation)

To reproduce the core experimental results reported in the paper (e.g., 5-fold cross-validation metrics), use the unified evaluation script. This script automatically iterates through all dataset folds, compiles the metrics, and exports a standardized academic report.

```
# Evaluate across all 5 folds of the USRDD dataset
python evaluate.py --dataset usrdd

# Evaluate on the RDRD dataset
python evaluate.py --dataset rdrd
```

### 4. Model Training

To train the HKANet architecture from scratch or fine-tune it on a specific dataset:

```
python train.py --dataset rdrd --epochs 100 --batch 16 --device 0
```

------

## 📊 Evaluation Outputs

All execution outputs, including quantitative metric tables (Precision, Recall, mAP@0.5, mAP@0.5:0.95), inference speeds, and parameter complexity, are automatically saved to the `runs/` directory.

During evaluation, a comprehensive `paper_metrics_report.txt` is generated in the respective output folder (e.g., `runs/evaluate/USRDD/Fold_1/`), ready for direct academic reporting.

------

## 📜 Acknowledgements & Citation

This framework is built upon the foundational engine of [Ultralytics](https://www.google.com/search?q=https://github.com/ultralytics/ultralytics&authuser=1), heavily customized to incorporate novel spatial and structural attention mechanisms for Civil Engineering applications.

------


"""
Dataset Preparation Tool
---------------------------------------------------------
This script is used for merging and splitting datasets.
It utilizes `pathlib` for robust, cross-platform path resolution.
"""

import os
import random
import shutil
from pathlib import Path

# ==============================================================================
# PART 1: Merge Datasets (Currently Commented Out)
# ==============================================================================
# PROJECT_ROOT = Path(__file__).resolve().parent
# base_path = PROJECT_ROOT / 'Competition_Dataset' / 'train_dataset'
# source_folders = ['difficulty_0', 'difficulty_1', 'difficulty_2']
# all_data_path = base_path / 'all_data'
#
# print("[*] Starting dataset merge...")
# (all_data_path / 'images').mkdir(parents=True, exist_ok=True)
# (all_data_path / 'labels').mkdir(parents=True, exist_ok=True)
#
# for folder in source_folders:
#     print(f"\nProcessing folder: {folder}")
#     src_images_path = base_path / folder / 'images'
#     src_labels_path = base_path / folder / 'labels'
#
#     if src_images_path.exists():
#         for img_file in src_images_path.iterdir():
#             shutil.copy2(img_file, all_data_path / 'images')
#         print(f"Copied files from {folder}/images to all_data/images")
#
#     if src_labels_path.exists():
#         for lbl_file in src_labels_path.iterdir():
#             shutil.copy2(lbl_file, all_data_path / 'labels')
#         print(f"Copied files from {folder}/labels to all_data/labels")

# ==============================================================================
# PART 2: Split Dataset into Train and Validation Sets
# ==============================================================================

def copy_files(file_list, source_img_dir, source_lbl_dir, dest_img_dir, dest_lbl_dir):
    """
    Copies image and corresponding label files from source to destination directories.
    """
    for file_stem in file_list:
        original_image_name = ""
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = source_img_dir / (file_stem + ext)
            if img_path.exists():
                original_image_name = file_stem + ext
                break
        
        if not original_image_name:
            print(f"[!] Warning: Original image file for {file_stem} not found.")
            continue

        label_name = file_stem + '.txt'
        
        src_image = source_img_dir / original_image_name
        src_label = source_lbl_dir / label_name
        dest_image = dest_img_dir / original_image_name
        dest_label = dest_lbl_dir / label_name
        
        shutil.copy2(src_image, dest_image)
        
        if src_label.exists():
            shutil.copy2(src_label, dest_label)
        else:
            print(f"[!] Warning: Label file {label_name} not found for image {original_image_name}.")

if __name__ == '__main__':
    # --- Cross-Platform Path Configuration ---
    
    # 1. Dynamically resolve the project root directory
    PROJECT_ROOT = Path(__file__).resolve().parent
    
    # 2. Define standard paths relative to the project root
    all_data_path = PROJECT_ROOT / 'Competition_Dataset' / 'train_dataset' / 'all_data'
    output_path = PROJECT_ROOT / 'Competition_Dataset' / 'train_dataset' / 'split_dataset'
    
    val_split_ratio = 0.2
    random_seed = 42
    random.seed(random_seed)

    # --- Execution ---
    if not all_data_path.exists():
        print(f"[Error] Source directory '{all_data_path}' does not exist.")
        raise SystemExit(1)

    source_images_path = all_data_path / 'images'
    source_labels_path = all_data_path / 'labels'

    train_images_path = output_path / 'train' / 'images'
    train_labels_path = output_path / 'train' / 'labels'
    val_images_path = output_path / 'val' / 'images'
    val_labels_path = output_path / 'val' / 'labels'

    print(f"[*] Creating output directories at: {output_path}")
    train_images_path.mkdir(parents=True, exist_ok=True)
    train_labels_path.mkdir(parents=True, exist_ok=True)
    val_images_path.mkdir(parents=True, exist_ok=True)
    val_labels_path.mkdir(parents=True, exist_ok=True)

    print("[*] Reading image list...")
    # Use robust pathlib methods to read files
    image_files = [f.stem for f in source_images_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

    random.shuffle(image_files)
    split_index = int(len(image_files) * (1 - val_split_ratio))

    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    print(f"\nTotal images found: {len(image_files)}")
    print(f"Split ratio: Train 80% ({len(train_files)} files), Val 20% ({len(val_files)} files)")

    print("\n[*] Copying training set files...")
    copy_files(train_files, source_images_path, source_labels_path, train_images_path, train_labels_path)

    print("\n[*] Copying validation set files...")
    copy_files(val_files, source_images_path, source_labels_path, val_images_path, val_labels_path)

    print("\n[✔] Dataset splitting completed successfully!")
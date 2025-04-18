import torch
import os
import glob
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from tqdm import tqdm
import random
import sys

# --- Configuration (Match Training Script Params) ---
manual_dataset_root = './data'
dataset_name = 'DIV2K'
train_hr_folder_name = 'DIV2K_train_HR'
valid_hr_folder_name = 'DIV2K_valid_HR'
manual_dataset_base = os.path.join(manual_dataset_root, dataset_name)
manual_train_hr_dir = os.path.join(manual_dataset_base, train_hr_folder_name)
manual_valid_hr_dir = os.path.join(manual_dataset_base, valid_hr_folder_name)

hr_patch_size = 96
lr_scale = 2
lr_patch_size = hr_patch_size // lr_scale
channels = 3

# --- Output Directories for Preprocessed Patches ---
# It's good practice to include patch size/scale in the name
output_base_dir = os.path.join(manual_dataset_base, f"preprocessed_patches_HR{hr_patch_size}_LR{lr_patch_size}")
output_train_dir = os.path.join(output_base_dir, 'train')
output_valid_dir = os.path.join(output_base_dir, 'valid')

os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_valid_dir, exist_ok=True)

print(f"Preprocessed patches will be saved to: {output_base_dir}")

# --- Number of patches to extract per image ---
# Adjust based on desired dataset size and disk space
# Example: 800 train images * 200 patches = 160,000 train patches
# Example: 100 valid images * 50 patches = 5,000 valid patches
num_patches_per_train_image = 200
num_patches_per_valid_image = 50 # Usually fewer for validation

# --- Transforms (Identical to Training) ---
norm_mean = (0.5, 0.5, 0.5)
norm_std = (0.5, 0.5, 0.5)
# We only need the tensor + norm transforms here, cropping/resizing is done manually
hr_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])
# LR transform includes the resize!
lr_transform = transforms.Compose([
    transforms.Resize((lr_patch_size, lr_patch_size), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# --- Helper Function for Processing ---
def process_and_save_patches(image_dir, output_dir, num_patches_per_image, set_name="train"):
    print(f"\nProcessing images in: {image_dir} for {set_name} set...")
    image_filenames = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    if not image_filenames:
        image_filenames = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    if not image_filenames:
        print(f"WARNING: No PNG or JPG images found in {image_dir}. Skipping.")
        return

    total_patches_saved = 0
    image_pbar = tqdm(image_filenames, desc=f"Processing {set_name} images")

    for img_idx, img_path in enumerate(image_pbar):
        try:
            hr_image_pil = Image.open(img_path).convert('RGB')
            w, h = hr_image_pil.size

            # Handle images smaller than patch size (resize up)
            if w < hr_patch_size or h < hr_patch_size:
                new_h = max(h, hr_patch_size)
                new_w = max(w, hr_patch_size)
                # Use BICUBIC for consistency if upscaling, though quality impact is minor here
                hr_image_pil = transforms.functional.resize(hr_image_pil, (new_h, new_w), interpolation=InterpolationMode.BICUBIC)
                w, h = hr_image_pil.size
                # If still too small after resize (shouldn't happen with max), skip
                if w < hr_patch_size or h < hr_patch_size:
                    print(f"WARN: Skipped {img_path} (too small even after resize attempt).")
                    continue

            patches_saved_for_this_image = 0
            # Try to extract the desired number of patches, but handle cases where image is small
            max_attempts = num_patches_per_image * 5 # Give some leeway for random crop failures
            attempts = 0
            while patches_saved_for_this_image < num_patches_per_image and attempts < max_attempts:
                attempts += 1
                rand_h = np.random.randint(0, h - hr_patch_size + 1)
                rand_w = np.random.randint(0, w - hr_patch_size + 1)

                hr_patch_pil = hr_image_pil.crop((rand_w, rand_h, rand_w + hr_patch_size, rand_h + hr_patch_size))

                # Apply transforms
                hr_patch_tensor = hr_transform(hr_patch_pil)
                # IMPORTANT: Apply LR transform to the *PIL* patch before HR transform
                lr_patch_tensor = lr_transform(hr_patch_pil)

                # Save the patch pair
                # Naming convention: {set}_{original_img_idx}_{patch_num}.pt
                patch_filename = f"{set_name}_{img_idx:04d}_{patches_saved_for_this_image:04d}.pt"
                save_path = os.path.join(output_dir, patch_filename)

                # Save as a dictionary
                torch.save({
                    'lr': lr_patch_tensor,
                    'hr': hr_patch_tensor
                }, save_path)

                patches_saved_for_this_image += 1
                total_patches_saved += 1

            # Update progress bar description with total patches saved so far
            image_pbar.set_postfix({"Total Patches": f"{total_patches_saved:,}"})

        except FileNotFoundError:
            print(f"\nWARN FNF: {img_path}. Skipping.")
            continue
        except Exception as e:
            print(f"\nWARN Processing {img_path}: {e}. Skipping.")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            continue

    print(f"\nFinished processing {set_name} set. Total patches saved: {total_patches_saved}")
    return total_patches_saved

# --- Main Execution ---
if __name__ == '__main__':
    print("--- Starting Offline Patch Pre-processing ---")
    print(f"HR Patch Size: {hr_patch_size}x{hr_patch_size}, LR Patch Size: {lr_patch_size}x{lr_patch_size}")
    print(f"Number of patches per TRAIN image: {num_patches_per_train_image}")
    print(f"Number of patches per VALID image: {num_patches_per_valid_image}")
    print("--- !!! This may take a significant amount of time and disk space !!! ---")

    # Process Training Data
    train_count = process_and_save_patches(manual_train_hr_dir, output_train_dir, num_patches_per_train_image, "train")

    # Process Validation Data
    valid_count = process_and_save_patches(manual_valid_hr_dir, output_valid_dir, num_patches_per_valid_image, "valid")

    print("\n--- Pre-processing Complete ---")
    print(f"Total Training Patches Saved: {train_count}")
    print(f"Total Validation Patches Saved: {valid_count}")
    print(f"Patches saved in: {output_base_dir}")
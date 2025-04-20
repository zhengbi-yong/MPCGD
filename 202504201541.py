# 2025年4月18日 - Training with Preprocessed Patches (CNN, Intermediate Loss Contributes to Grad, Final Layer L1 Grad)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode, to_pil_image
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from PIL import Image
import glob
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import sys
import traceback
from tqdm import tqdm
import random
import math # For calculating padding in CNN

# --- Constants and Configuration ---
manual_dataset_root = './data' # Base data folder
dataset_name = 'DIV2K'

# Directories for preprocessed patches
hr_patch_size = 96
lr_scale = 2
lr_patch_size = hr_patch_size // lr_scale
preprocessed_patch_dir_name = f"preprocessed_patches_HR{hr_patch_size}_LR{lr_patch_size}"
preprocessed_base_dir = os.path.join(manual_dataset_root, dataset_name, preprocessed_patch_dir_name)
train_patch_dir = os.path.join(preprocessed_base_dir, 'train')
valid_patch_dir = os.path.join(preprocessed_base_dir, 'valid')


# Model and Training Parameters
channels = 3
cnn_features = 64 # Number of feature channels in CNN layers
num_cnn_blocks = 4 # Number of intermediate convolutional blocks (defines intermediate losses)

num_epochs = 100 # Adjusted epochs
batch_size = 256  # Adjusted batch size
learning_rate = 1e-4
# <<< ADDED: Weight for intermediate losses contribution to total loss >>>
intermediate_loss_weight = 0.25 # Hyperparameter: How much intermediate MSE losses contribute
INITIAL_NUM_WORKERS = 8
NUM_VIS_EXAMPLES = 5

# Results directory naming convention
# <<< MODIFIED: Updated results directory name for combined loss >>>
results_base_dir_name = f"results_div2k_manual_sr_CNN_f{cnn_features}_COMBINED_loss_w{intermediate_loss_weight}_patch_FINAL_L1_PREPROCESSED_HR{hr_patch_size}"
now = datetime.datetime.now()
timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
results_base_dir = "results"
results_dir = os.path.join(results_base_dir, results_base_dir_name, timestamp_str)


# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transforms ---
norm_mean = torch.tensor((0.5, 0.5, 0.5)).view(channels, 1, 1)
norm_std = torch.tensor((0.5, 0.5, 0.5)).view(channels, 1, 1)

# --- Function Definitions ---

# check_preprocessed_dataset (Unchanged)
def check_preprocessed_dataset(train_dir, valid_dir):
    train_ok = os.path.isdir(train_dir) and bool(glob.glob(os.path.join(train_dir, '*.pt')))
    valid_ok = os.path.isdir(valid_dir) and bool(glob.glob(os.path.join(valid_dir, '*.pt')))
    if train_ok and valid_ok:
        print(f"Found preprocessed dataset:\n  Train: {train_dir}\n  Valid: {valid_dir}")
        return True
    else:
        print("ERROR: Preprocessed dataset directories not found or empty!")
        print(f"  Expected Train Dir: {train_dir}")
        print(f"  Expected Valid Dir: {valid_dir}")
        print("Please run the preprocessing script first.")
        return False

# setup_results_dir (Unchanged)
def setup_results_dir(base_dir_name):
    now = datetime.datetime.now(); timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    base_results_path = "results"
    full_results_dir = os.path.join(base_results_path, base_dir_name, timestamp_str)
    os.makedirs(full_results_dir, exist_ok=True)
    print(f"结果将保存在: {full_results_dir}"); return full_results_dir

# <<< MODIFIED: plot_loss_history title/labels for Combined Loss CNN >>>
def plot_loss_history(losses_history, num_epochs, results_dir, weight):
    plt.figure(figsize=(12, 7)); epochs_range = range(1, num_epochs + 1)
    num_actual_layers_logged = len(losses_history) # Includes intermediate + final
    final_layer_idx = num_actual_layers_logged - 1

    intermediate_mse_history = [losses_history[i] for i in range(final_layer_idx)]
    final_l1_history = losses_history[final_layer_idx] # This is the L1 loss of the final layer

    # Plot intermediate MSE losses (contributed to gradient)
    for i in range(final_layer_idx):
        if not intermediate_mse_history[i]: continue
        label = f'Intermediate Block {i+1} Avg MSE (Weight: {weight})'
        valid_indices = [idx for idx, l in enumerate(intermediate_mse_history[i]) if l is not None and not np.isnan(l)]
        if not valid_indices: continue
        valid_epochs = [epochs_range[idx] for idx in valid_indices]
        valid_losses = [intermediate_mse_history[i][idx] for idx in valid_indices]
        if valid_losses:
            plt.plot(valid_epochs, valid_losses, label=label, marker='.', linestyle=':', linewidth=1.0, markersize=3)

    # Plot final layer L1 loss (contributed to gradient)
    if final_l1_history:
        label = f'Final Output Avg L1 Loss (Weight: 1.0)'
        line_style = '-'
        line_width = 2.0
        valid_indices = [idx for idx, l in enumerate(final_l1_history) if l is not None and not np.isnan(l)]
        if valid_indices:
             valid_epochs = [epochs_range[idx] for idx in valid_indices]
             valid_losses = [final_l1_history[idx] for idx in valid_indices]
             if valid_losses:
                  plt.plot(valid_epochs, valid_losses, label=label, marker='o', linestyle=line_style, linewidth=line_width, markersize=4)

    plt.title(f'Div2k Patch SR Training (CNN f{cnn_features}): Combined Loss History (Int. MSE w={weight}, Final L1)')
    plt.xlabel('Epoch')
    plt.ylabel('Avg Loss Value per Component') # Y-axis shows individual component averages
    if num_epochs > 0:
        tick_step = max(1, num_epochs // 10); plt.xticks(range(1, num_epochs + 1, tick_step))
    plt.legend(fontsize='small', loc='best'); plt.grid(True);
    loss_plot_path = os.path.join(results_dir, 'div2k_cnn_patch_sr_combined_loss_history.png')
    plt.savefig(loss_plot_path); print(f"Loss plot saved: {loss_plot_path}"); plt.close()

# imshow_patch (Unchanged)
def imshow_patch(ax, img_numpy, title):
     if img_numpy.shape[0] == channels: # If C, H, W -> H, W, C
         img_numpy = img_numpy.transpose(1, 2, 0)
     ax.imshow(np.clip(img_numpy, 0, 1)); ax.set_title(title, fontsize=8); ax.axis('off')

# denormalize_tensor (Unchanged)
def denormalize_tensor(tensor):
    denorm_tensor = tensor.clone(); _mean = norm_mean.to(denorm_tensor.device); _std = norm_std.to(denorm_tensor.device)
    if denorm_tensor.ndim == 4:
         _mean = _mean.unsqueeze(0); _std = _std.unsqueeze(0); denorm_tensor = denorm_tensor * _std + _mean
    elif denorm_tensor.ndim == 3:
         denorm_tensor = denorm_tensor * _std + _mean
    else: print("WARN: Unexpected tensor dimension in denormalize_tensor:", denorm_tensor.ndim)
    return torch.clamp(denorm_tensor, 0, 1)

# --- Class Definitions ---

# Preprocessed_SR_Patch_Dataset (Unchanged)
class Preprocessed_SR_Patch_Dataset(Dataset):
    def __init__(self, patch_dir):
        self.patch_dir = patch_dir
        print(f"Searching for preprocessed patches (.pt files) in: {patch_dir} ...")
        self.patch_filenames = sorted(tqdm(glob.glob(os.path.join(patch_dir, '*.pt')), desc="Finding patches", leave=False, unit=" file"))
        if not self.patch_filenames: raise FileNotFoundError(f"FATAL: No '.pt' patch files found in: {patch_dir}")
        self.num_patches = len(self.patch_filenames); print(f"Initialized dataset: '{patch_dir}' ({self.num_patches} patches).")
    def __len__(self): return self.num_patches
    def __getitem__(self, idx):
        patch_path = self.patch_filenames[idx]
        try:
            data = torch.load(patch_path, map_location='cpu')
            return data['lr'].float(), data['hr'].float()
        except Exception as e: print(f"\nERROR loading {patch_path}: {e}. Returning None."); return None

# collate_fn_skip_none (Unchanged)
def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch));
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)

# SR_CNNWithIntermediateOutputs Model (Unchanged - structure is already suitable)
class SR_CNNWithIntermediateOutputs(nn.Module):
    def __init__(self, num_channels, feature_dim, num_blocks, upscale_factor, target_hr_size):
        super(SR_CNNWithIntermediateOutputs, self).__init__()
        self.upscale_factor = upscale_factor
        self.target_hr_size = target_hr_size
        self.num_blocks = num_blocks

        # Main Path
        self.conv1 = nn.Conv2d(num_channels, feature_dim, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ))
        self.conv_before_upsample = nn.Conv2d(feature_dim, feature_dim * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.conv_final = nn.Conv2d(feature_dim, num_channels, kernel_size=3, padding=1)

        # Intermediate Output Projections
        self.intermediate_projections = nn.ModuleList()
        self.intermediate_upsamplers = nn.ModuleList()
        for i in range(num_blocks):
            proj_conv = nn.Conv2d(feature_dim, num_channels, kernel_size=1)
            self.intermediate_projections.append(proj_conv)
            upsample_layer = nn.Upsample(size=target_hr_size, mode='bilinear', align_corners=False)
            self.intermediate_upsamplers.append(upsample_layer)

        self.output_activation = nn.Tanh()

    def forward(self, x_lr):
        all_outputs = []
        h = self.relu1(self.conv1(x_lr))
        current_h = h
        for i in range(self.num_blocks):
            current_h = self.blocks[i](current_h)
            intermediate_proj = self.intermediate_projections[i](current_h)
            intermediate_upsampled = self.intermediate_upsamplers[i](intermediate_proj)
            all_outputs.append(self.output_activation(intermediate_upsampled)) # Append intermediate activated output

        upsample_ready = self.conv_before_upsample(current_h)
        upsampled = self.pixel_shuffle(upsample_ready)
        final_output = self.conv_final(upsampled)
        all_outputs.append(self.output_activation(final_output)) # Append final activated output
        return all_outputs


# =====================================================
# Main Execution Block
# =====================================================
if __name__ == '__main__':

    # --- Initial Setup ---
    print(f"Using device: {device}")
    torch.manual_seed(42); np.random.seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

    # --- Check for Preprocessed Dataset ---
    if not check_preprocessed_dataset(train_patch_dir, valid_patch_dir): sys.exit(1)

    # --- Create Results Directory ---
    results_dir = setup_results_dir(results_base_dir_name)
    print(f"Intermediate Loss Weight: {intermediate_loss_weight}")

    # --- Create Datasets and Loaders ---
    try:
        train_dataset = Preprocessed_SR_Patch_Dataset(train_patch_dir)
        test_dataset = Preprocessed_SR_Patch_Dataset(valid_patch_dir)
    except FileNotFoundError as e: print(f"Error initializing dataset: {e}"); sys.exit(1)

    print(f"Using {INITIAL_NUM_WORKERS} dataloader workers.")
    current_num_workers = INITIAL_NUM_WORKERS
    while current_num_workers >= 0:
        try:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=current_num_workers, pin_memory=(device.type == 'cuda'), persistent_workers=(current_num_workers > 0), drop_last=True, collate_fn=collate_fn_skip_none)
            test_batch_size = min(batch_size, 128)
            test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=current_num_workers, pin_memory=(device.type == 'cuda'), persistent_workers=(current_num_workers > 0), drop_last=False, collate_fn=collate_fn_skip_none)
            print(f"Successfully created DataLoaders with {current_num_workers} workers.")
            break
        except RuntimeError as e:
            print(f"DataLoader error with {current_num_workers} workers: {e}")
            current_num_workers -= 4
            if current_num_workers < 0: print("ERROR: Failed to create DataLoaders even with 0 workers."); sys.exit(1)
            print(f"Retrying with {current_num_workers} workers...")

    print(f"Training config: LR={lr_patch_size}x{lr_patch_size}, HR={hr_patch_size}x{hr_patch_size}, Scale={lr_scale}x")
    print(f"CNN Params: Features={cnn_features}, Blocks={num_cnn_blocks}")

    # --- Instantiate Model, Optimizer, Loss ---
    model = SR_CNNWithIntermediateOutputs(
        num_channels=channels, feature_dim=cnn_features, num_blocks=num_cnn_blocks,
        upscale_factor=lr_scale, target_hr_size=(hr_patch_size, hr_patch_size)
    ).to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # <<< Loss for FINAL output (typically L1 for SR) >>>
    final_criterion = nn.L1Loss()
    # <<< Loss for INTERMEDIATE outputs (MSE is common) >>>
    intermediate_criterion = nn.MSELoss()

    # --- Metrics Initialization (For Evaluation/Visualization on [0, 1] data) ---
    eval_psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    eval_ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    eval_mae_metric = nn.L1Loss() # Use L1 instance for MAE on denormalized data

    # --- Training Loop ---
    # losses_history stores *average* intermediate MSEs and *average* final L1 loss per epoch
    losses_history = [[] for _ in range(num_cnn_blocks + 1)]
    print(f"--- 开始训练 (CNN f{cnn_features}, Preprocessed Div2k Patches - Combined Loss, Int Weight={intermediate_loss_weight}) ---")

    epochs_pbar = tqdm(range(num_epochs), desc="Epochs")
    for epoch in epochs_pbar:
        model.train()
        # Track epoch totals for calculating averages
        epoch_batch_intermediate_mse_losses_items = [[] for _ in range(num_cnn_blocks)]
        total_epoch_final_l1_sum = 0.0
        # <<< Track total combined loss for reporting convenience >>>
        total_epoch_combined_loss_sum = 0.0
        processed_batches_in_epoch = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False, unit="batch")
        for batch_idx, batch_data in enumerate(train_pbar):
            if batch_data is None: continue
            lr_data, hr_target_norm = batch_data # LR: [B,C,Hlr,Wlr], HR: [B,C,Hhr,Whr] [-1,1]
            current_batch_size = lr_data.size(0)
            if current_batch_size == 0: continue

            lr_data, hr_target_norm = lr_data.to(device), hr_target_norm.to(device)

            # --- Forward Pass ---
            try:
                # all_predictions_norm: List [interm1_hr, interm2_hr, ..., final_hr]
                # Shape: [B, C, Hhr, Whr], Range: [-1, 1]
                all_predictions_norm = model(lr_data)
                if len(all_predictions_norm) != num_cnn_blocks + 1:
                     print(f"\n!!! ERROR E{epoch+1} B{batch_idx}: Model returned {len(all_predictions_norm)} outputs, expected {num_cnn_blocks + 1}")
                     continue
            except Exception as model_e:
                print(f"\n!!! ERROR Forward E{epoch+1} B{batch_idx}: {model_e}"); traceback.print_exc(); continue

            # --- Combined Loss Calculation ---
            optimizer.zero_grad()
            intermediate_preds_norm = all_predictions_norm[:-1] # First num_cnn_blocks elements
            final_pred_norm = all_predictions_norm[-1]       # Last element

            # Calculate Final Loss (L1)
            try:
                loss_final = final_criterion(final_pred_norm, hr_target_norm)
                if torch.isnan(loss_final):
                    print(f"!!! NaN final L1 loss E{epoch+1} B{batch_idx}. Skip batch !!!"); optimizer.zero_grad(); continue
            except Exception as final_loss_e:
                 print(f"\n!!! ERROR Final L1 Calc E{epoch+1} B{batch_idx}: {final_loss_e}"); traceback.print_exc(); optimizer.zero_grad(); continue

            # Calculate Intermediate Losses (MSE)
            current_batch_intermediate_losses_tensors = [] # Store loss tensors
            current_batch_intermediate_losses_items = []   # Store .item() values
            intermediate_nan = False
            for i in range(num_cnn_blocks):
                pred_interm_norm = intermediate_preds_norm[i]
                if pred_interm_norm.shape != hr_target_norm.shape:
                    print(f"\n!!! Shape Mismatch Interm. E{epoch+1} B{batch_idx} L{i+1}: Pred={pred_interm_norm.shape}, Target={hr_target_norm.shape}"); intermediate_nan = True; break
                try:
                    loss_interm_i = intermediate_criterion(pred_interm_norm, hr_target_norm)
                    if torch.isnan(loss_interm_i):
                        print(f"!!! NaN interm. loss E{epoch+1} B{batch_idx} L{i+1}. Skip batch !!!"); intermediate_nan = True; break
                    current_batch_intermediate_losses_tensors.append(loss_interm_i)
                    current_batch_intermediate_losses_items.append(loss_interm_i.item()) # For logging history
                except Exception as interm_loss_e:
                     print(f"\n!!! ERROR Interm. Loss Calc E{epoch+1} B{batch_idx} L{i+1}: {interm_loss_e}"); traceback.print_exc(); intermediate_nan = True; break
            if intermediate_nan:
                optimizer.zero_grad(); continue # Skip batch if any intermediate loss failed

            # <<< Calculate Total Combined Loss >>>
            total_intermediate_loss_tensor = torch.sum(torch.stack(current_batch_intermediate_losses_tensors))
            total_loss = loss_final + intermediate_loss_weight * total_intermediate_loss_tensor

            # --- Backward Pass on Combined Loss ---
            try:
                total_loss.backward()
                optimizer.step()
            except Exception as backward_e:
                print(f"\n!!! ERROR Backward/Step E{epoch+1} B{batch_idx}: {backward_e}"); traceback.print_exc(); optimizer.zero_grad(); continue

            # --- Accumulate Batch Stats for Epoch Average ---
            processed_batches_in_epoch += 1
            total_epoch_final_l1_sum += loss_final.item() # Accumulate final L1 .item()
            total_epoch_combined_loss_sum += total_loss.item() # Accumulate total combined loss .item()
            # Accumulate intermediate MSE .item() values
            if len(current_batch_intermediate_losses_items) == num_cnn_blocks:
                for i in range(num_cnn_blocks):
                    epoch_batch_intermediate_mse_losses_items[i].append(current_batch_intermediate_losses_items[i])
            else:
                 print(f"Warn: Inconsistent number of intermediate losses logged B{batch_idx}. Skip history append."); continue # Should not happen if checks above pass

            # Update tqdm postfix showing the combined loss and final L1 loss
            train_pbar.set_postfix(Combined_Loss=f"{total_loss.item():.6f}", Final_L1=f"{loss_final.item():.6f}", refresh=False)

        # --- Epoch Summary ---
        avg_epoch_final_l1 = total_epoch_final_l1_sum / processed_batches_in_epoch if processed_batches_in_epoch > 0 else float('nan')
        avg_epoch_combined_loss = total_epoch_combined_loss_sum / processed_batches_in_epoch if processed_batches_in_epoch > 0 else float('nan')
        # Calculate average for EACH intermediate layer's logged MSE loss
        avg_epoch_intermediate_mse_losses = [np.mean(batch_losses) if batch_losses else float('nan') for batch_losses in epoch_batch_intermediate_mse_losses_items]

        epochs_pbar.set_postfix(avg_Combined=f"{avg_epoch_combined_loss:.6f}", avg_L1=f"{avg_epoch_final_l1:.6f}")

        tqdm.write(f"\n--- Epoch {epoch+1} Summary ---")
        loss_str = " | ".join([f"Block {i+1}:{l:.6f}" for i, l in enumerate(avg_epoch_intermediate_mse_losses) if not np.isnan(l)])
        tqdm.write(f"Avg Intermediate Block MSE Losses (Component): [{loss_str}]")
        tqdm.write(f"Avg FINAL Output L1 Loss (Component): {avg_epoch_final_l1:.6f}")
        tqdm.write(f"Avg TOTAL Combined Loss (Used for Grad): {avg_epoch_combined_loss:.6f} (L1 + {intermediate_loss_weight} * Sum(MSEs))")
        tqdm.write("---")

        # Store history of avg intermediate MSEs (component average)
        for i in range(num_cnn_blocks):
             while len(losses_history[i]) < epoch: losses_history[i].append(float('nan'))
             losses_history[i].append(avg_epoch_intermediate_mse_losses[i])
        # Store history of avg final L1 loss (component average)
        while len(losses_history[num_cnn_blocks]) < epoch: losses_history[num_cnn_blocks].append(float('nan'))
        losses_history[num_cnn_blocks].append(avg_epoch_final_l1)

    print("--- 训练完成 ---")

    # --- Plot Loss History ---
    # Pass the weight used during training for accurate labeling
    plot_loss_history(losses_history, num_epochs, results_dir, intermediate_loss_weight)

    # --- Evaluate Model Performance & Save Visualizations ---
    # Evaluation still primarily focuses on the quality of the FINAL output
    model.eval()
    test_total_final_l1 = 0.0 # L1 on normalized [-1, 1] data (final output)
    test_total_psnr = 0.0 # PSNR on denormalized [0, 1] data (final output)
    test_total_ssim = 0.0 # SSIM on denormalized [0, 1] data (final output)
    test_samples_count = 0
    saved_vis_count = 0

    print(f"\n--- 开始评估 (CNN f{cnn_features}, Combined Loss Trained, Preprocessed) & Saving Visualizations ---")
    test_pbar = tqdm(test_loader, desc="Evaluating", leave=False, unit="batch")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_pbar):
            if batch_data is None: continue
            lr_data, hr_target_norm = batch_data # Normalized [-1, 1], [B, C, H, W]
            current_batch_size = lr_data.size(0)
            if current_batch_size == 0: continue

            lr_data, hr_target_norm = lr_data.to(device), hr_target_norm.to(device)

            try:
                # Get all outputs [interm1, ..., final] (normalized [-1, 1])
                all_outputs_norm = model(lr_data)
                final_output_norm = all_outputs_norm[-1] # Shape [B, C, Hhr, Whr]

                # --- Calculate Final L1 Loss for the batch (on normalized data) ---
                if final_output_norm.shape != hr_target_norm.shape:
                    print(f"!!! Eval Shape Mismatch B{batch_idx}: Pred={final_output_norm.shape}, Target={hr_target_norm.shape}")
                    batch_final_l1_loss = torch.tensor(float('nan'))
                else:
                    # Use the final_criterion (L1) to calculate loss on normalized data
                    batch_final_l1_loss = final_criterion(final_output_norm, hr_target_norm)

                if not torch.isnan(batch_final_l1_loss):
                    test_total_final_l1 += batch_final_l1_loss.item() * current_batch_size
                    test_samples_count += current_batch_size # Increment only if final L1 loss is valid
                    if test_samples_count > 0:
                       test_pbar.set_postfix(avg_Final_L1=f"{test_total_final_l1/test_samples_count:.6f}")
                else: print(f"!!! Eval NaN Final L1 Loss B{batch_idx}")

                # --- Generate and Save Visualizations (uses all layers' outputs) ---
                # Metrics (PSNR, SSIM, MAE) are calculated on DENORMALIZED [0, 1] data for the FINAL output primarily
                num_layers_output = len(all_outputs_norm)

                if torch.isnan(batch_final_l1_loss): continue # Skip vis if final output had issues

                for sample_idx in range(current_batch_size):
                    if saved_vis_count >= NUM_VIS_EXAMPLES: break

                    lr_sample_norm = lr_data[sample_idx]; hr_gt_norm = hr_target_norm[sample_idx]
                    hr_gt_denorm = denormalize_tensor(hr_gt_norm)
                    hr_gt_denorm_vis = hr_gt_denorm.cpu().numpy()
                    lr_sample_denorm = denormalize_tensor(lr_sample_norm)
                    lr_display_upscaled = F.interpolate(lr_sample_denorm.unsqueeze(0), size=(hr_patch_size, hr_patch_size), mode='bicubic', align_corners=False).squeeze(0)
                    lr_display_upscaled_vis = lr_display_upscaled.cpu().numpy()

                    predicted_patches_denorm_vis = []
                    metrics_per_layer = [] # Store metrics for each layer's output
                    gt_tensor_for_metrics = hr_gt_denorm.unsqueeze(0).to(device) # [1, C, Hhr, Whr], range [0, 1]

                    for i in range(num_layers_output): # Loop through intermediate and final outputs
                        pred_patch_norm_sample = all_outputs_norm[i][sample_idx] # [C, Hhr, Whr], range [-1, 1]
                        pred_patch_denorm = denormalize_tensor(pred_patch_norm_sample) # [C, Hhr, Whr], range [0, 1]
                        predicted_patches_denorm_vis.append(pred_patch_denorm.cpu().numpy())

                        # Calculate metrics on denormalized data ([0, 1] range)
                        pred_tensor_for_metrics = pred_patch_denorm.unsqueeze(0).to(device) # [1, C, Hhr, Whr], range [0, 1]
                        try:
                             mse_val = F.mse_loss(pred_tensor_for_metrics, gt_tensor_for_metrics).item()
                             mae_val = eval_mae_metric(pred_tensor_for_metrics, gt_tensor_for_metrics).item()
                             psnr_val = eval_psnr_metric(pred_tensor_for_metrics, gt_tensor_for_metrics).item()
                             ssim_val = eval_ssim_metric(pred_tensor_for_metrics.float(), gt_tensor_for_metrics.float()).item()
                             # Accumulate FINAL layer PSNR/SSIM for reporting average across test set
                             if i == num_layers_output - 1: # If it's the final output
                                if not np.isnan(psnr_val): test_total_psnr += psnr_val # Accumulate per sample
                                if not np.isnan(ssim_val): test_total_ssim += ssim_val # Accumulate per sample
                        except Exception as e:
                             print(f"WARN: Metric calc failed Layer {i} Sample {saved_vis_count+1}: {e}"); mse_val, mae_val, psnr_val, ssim_val = [float('nan')]*4
                        metrics = {'MSE': mse_val, 'MAE': mae_val, 'PSNR': psnr_val, 'SSIM': ssim_val}; metrics_per_layer.append(metrics)

                    # Plotting logic (remains the same)
                    num_total_plots = 2 + num_layers_output
                    fig, axes = plt.subplots(1, num_total_plots, figsize=(3 * num_total_plots, 3.5))
                    imshow_patch(axes[0], lr_display_upscaled_vis, f"Input LR ({lr_patch_size}x{lr_patch_size} upscaled)")
                    imshow_patch(axes[1], hr_gt_denorm_vis, f"Ground Truth HR ({hr_patch_size}x{hr_patch_size})")
                    for i in range(num_layers_output):
                        ax_pred = axes[i + 2]; pred_patch_denorm_numpy = predicted_patches_denorm_vis[i]
                        if i < num_cnn_blocks: layer_name = f'Interm. Block {i+1}'
                        else: layer_name = 'Final Output'
                        metrics = metrics_per_layer[i]
                        title = (f"Pred. from {layer_name}\n"
                                 f"MSE:{metrics.get('MSE', float('nan')):.3f} MAE:{metrics.get('MAE', float('nan')):.3f}\n"
                                 f"PSNR:{metrics.get('PSNR', float('nan')):.2f} SSIM:{metrics.get('SSIM', float('nan')):.3f}")
                        imshow_patch(ax_pred, pred_patch_denorm_numpy, title)
                    plt.tight_layout(pad=0.3, h_pad=0.5); vis_plot_path = os.path.join(results_dir, f'eval_sample_{saved_vis_count + 1}.png')
                    plt.savefig(vis_plot_path); plt.close(fig)
                    saved_vis_count += 1

                if saved_vis_count >= NUM_VIS_EXAMPLES:
                    print(f"\nSaved required {NUM_VIS_EXAMPLES} visualization examples.")

            except Exception as eval_e:
                print(f"\nError during eval batch {batch_idx}: {eval_e}"); traceback.print_exc(); continue

    # --- Final Test Summary ---
    if test_samples_count > 0:
        # Report metrics for the FINAL output
        final_avg_l1_eval = test_total_final_l1 / test_samples_count
        final_avg_psnr = test_total_psnr / test_samples_count
        final_avg_ssim = test_total_ssim / test_samples_count

        print(f'\n--- Test Evaluation Finished ---')
        print(f'Test set ({test_samples_count} samples): FINAL OUTPUT Performance')
        print(f'  Avg FINAL Output L1 Loss (calculated on normalized [-1,1] data): {final_avg_l1_eval:.6f}')
        print(f'  Avg FINAL Output PSNR (calculated on denormalized [0,1] data): {final_avg_psnr:.4f}')
        print(f'  Avg FINAL Output SSIM (calculated on denormalized [0,1] data): {final_avg_ssim:.4f}\n')
    else: print("\nTest set: No valid batches processed for metric calculation.\n")
    if saved_vis_count < NUM_VIS_EXAMPLES: print(f"Warning: Only saved {saved_vis_count}/{NUM_VIS_EXAMPLES} visualization examples.")

    print("\n--- Script Finished ---")
    print(f"Using PREPROCESSED Div2k dataset. Results in: {results_dir}")
    print(f"NOTE: Trained using CNN (f{cnn_features}, {num_cnn_blocks} blocks) with COMBINED loss.")
    print(f"      Loss = Final_L1 + {intermediate_loss_weight} * Sum(Intermediate_MSEs)")
    if current_num_workers > 4: print(f"NOTE: Using {current_num_workers} dataloader workers.")

# End of the if __name__ == '__main__': block
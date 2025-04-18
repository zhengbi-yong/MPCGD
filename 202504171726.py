# 2025年4月18日 - Training with Preprocessed Patches
# ... (Keep imports and other setup as before) ...
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode, to_pil_image # Added to_pil_image for potential debugging
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
import random # Keep random for visualization selection if needed later, but now using first N

# --- Constants and Configuration ---
manual_dataset_root = './data' # Base data folder
dataset_name = 'DIV2K'

# <<< MODIFIED: Point to preprocessed patch directories >>>
hr_patch_size = 96 # Keep for reference and model output size calculation
lr_scale = 2
lr_patch_size = hr_patch_size // lr_scale
preprocessed_patch_dir_name = f"preprocessed_patches_HR{hr_patch_size}_LR{lr_patch_size}"
preprocessed_base_dir = os.path.join(manual_dataset_root, dataset_name, preprocessed_patch_dir_name)
train_patch_dir = os.path.join(preprocessed_base_dir, 'train')
valid_patch_dir = os.path.join(preprocessed_base_dir, 'valid')
# <<< END MODIFICATION >>>

results_base_dir_name = "results_div2k_manual_sr_mlp_patch_combined_loss_metrics_PREPROCESSED" # Change results name
now = datetime.datetime.now()
timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
results_base_dir = "results"
results_dir = os.path.join(results_base_dir, results_base_dir_name, timestamp_str)

channels = 3
input_dim = lr_patch_size * lr_patch_size * channels
output_dim = hr_patch_size * hr_patch_size * channels
hidden_dim = 512
num_hidden_layers = 4
num_epochs = 200 # Keep reduced epochs for faster testing
batch_size = 1024 # Can try reducing if memory/compute is still an issue
learning_rate = 1e-4
INITIAL_NUM_WORKERS = 16 # Keep high for now, might be reducible if disk I/O is fast

# <<< NEW: Number of visualization examples to save during evaluation >>>
NUM_VIS_EXAMPLES = 5
# <<< END NEW >>>

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transforms ---
norm_mean = torch.tensor((0.5, 0.5, 0.5)).view(channels, 1, 1) # Keep as tensors for denormalization
norm_std = torch.tensor((0.5, 0.5, 0.5)).view(channels, 1, 1)

# --- Function Definitions ---

# check_preprocessed_dataset (Unchanged)
def check_preprocessed_dataset(train_dir, valid_dir):
    # Check if the *patch* directories exist and contain .pt files
    train_ok = os.path.isdir(train_dir) and bool(glob.glob(os.path.join(train_dir, '*.pt')))
    valid_ok = os.path.isdir(valid_dir) and bool(glob.glob(os.path.join(valid_dir, '*.pt')))
    if train_ok and valid_ok:
        print(f"Found preprocessed dataset:")
        print(f"  Train patches: {train_dir}")
        print(f"  Valid patches: {valid_dir}")
        return True
    else:
        print("ERROR: Preprocessed dataset directories not found or empty!")
        if not os.path.isdir(train_dir): print(f"  Missing: {train_dir}")
        if not os.path.isdir(valid_dir): print(f"  Missing: {valid_dir}")
        if os.path.isdir(train_dir) and not bool(glob.glob(os.path.join(train_dir, '*.pt'))): print(f"  No .pt files in: {train_dir}")
        if os.path.isdir(valid_dir) and not bool(glob.glob(os.path.join(valid_dir, '*.pt'))): print(f"  No .pt files in: {valid_dir}")
        print("\nPlease run the 'preprocess_div2k_patches.py' script first.")
        return False

# setup_results_dir (Unchanged)
def setup_results_dir(base_dir_name):
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    base_results_path = "results"
    full_results_dir = os.path.join(base_results_path, base_dir_name, timestamp_str)
    os.makedirs(full_results_dir, exist_ok=True)
    print(f"结果将保存在: {full_results_dir}")
    return full_results_dir

# plot_loss_history (Unchanged)
def plot_loss_history(losses_history, num_epochs, results_dir):
    plt.figure(figsize=(12, 7)); epochs_range = range(1, num_epochs + 1)
    num_actual_layers = len(losses_history) # Use actual length in case model changes
    for i in range(num_actual_layers):
        # Ensure losses_history[i] has data before accessing
        if not losses_history[i]: continue

        label = f'MSE from Input Act.' if i == 0 else f'MSE from Hidden {i} Act.'
        # Filter out None or NaN before plotting
        valid_indices = [idx for idx, l in enumerate(losses_history[i]) if l is not None and not np.isnan(l)]
        if not valid_indices: continue # Skip if no valid data for this layer
        valid_epochs = [epochs_range[idx] for idx in valid_indices]
        valid_losses = [losses_history[i][idx] for idx in valid_indices]

        if valid_losses: plt.plot(valid_epochs, valid_losses, label=label, marker='o', linestyle='-')

    sum_avg_losses, valid_sum_epochs = [], []
    for e_idx in range(num_epochs):
        epoch_losses = []
        valid_layer_losses_in_epoch = True
        for layer_idx in range(num_actual_layers):
             # Check bounds and validity
             if e_idx < len(losses_history[layer_idx]) and losses_history[layer_idx][e_idx] is not None and not np.isnan(losses_history[layer_idx][e_idx]):
                 epoch_losses.append(losses_history[layer_idx][e_idx])
             else:
                 valid_layer_losses_in_epoch = False
                 break # Don't calculate sum if any layer is missing/NaN for this epoch

        if valid_layer_losses_in_epoch:
            sum_avg_losses.append(sum(epoch_losses))
            valid_sum_epochs.append(epochs_range[e_idx])

    if valid_sum_epochs: plt.plot(valid_sum_epochs, sum_avg_losses, label='Sum of Average MSEs', color='black', linestyle='--', linewidth=2)
    plt.title('Div2k (Manual) Patch SR Training: Loss History (Preprocessed)'); plt.xlabel('Epoch'); plt.ylabel('Avg MSE Loss (Norm. Data)');
    # Adjust x-ticks dynamically
    if num_epochs > 0:
      tick_step = max(1, num_epochs // 10)
      plt.xticks(range(1, num_epochs + 1, tick_step))
    plt.legend(fontsize='small', loc='upper right'); plt.grid(True); plt.yscale('log')
    loss_plot_path = os.path.join(results_dir, 'div2k_manual_patch_sr_loss_history_preprocessed.png'); plt.savefig(loss_plot_path); print(f"Loss plot saved: {loss_plot_path}"); plt.close()


# imshow_patch (Unchanged)
def imshow_patch(ax, img_numpy, title):
     # Input img_numpy should be HWC, [0, 1] range
     ax.imshow(np.clip(img_numpy, 0, 1)); ax.set_title(title, fontsize=8); ax.axis('off') # Reduced font size

# --- NEW: Helper function for denormalization ---
def denormalize_tensor(tensor):
    """Denormalizes a tensor from [-1, 1] to [0, 1]"""
    # Clone to avoid modifying original tensor if it's needed elsewhere
    denorm_tensor = tensor.clone()
    # Ensure mean/std are on the same device as the tensor
    _mean = norm_mean.to(denorm_tensor.device)
    _std = norm_std.to(denorm_tensor.device)
    # Handle both batch (N, C, H, W) and single image (C, H, W)
    if denorm_tensor.ndim == 4: # Batch
         # Reshape mean/std for broadcasting over batch dim
        _mean = _mean.unsqueeze(0)
        _std = _std.unsqueeze(0)
        denorm_tensor = denorm_tensor * _std + _mean
    elif denorm_tensor.ndim == 3: # Single image
        denorm_tensor = denorm_tensor * _std + _mean
    else:
        print("WARN: Unexpected tensor dimension in denormalize_tensor:", denorm_tensor.ndim)

    return torch.clamp(denorm_tensor, 0, 1)

# --- Class Definitions ---

# Preprocessed_SR_Patch_Dataset (Unchanged)
class Preprocessed_SR_Patch_Dataset(Dataset):
    def __init__(self, patch_dir):
        self.patch_dir = patch_dir
        print(f"Searching for preprocessed patches (.pt files) in: {patch_dir} ...")
        self.patch_filenames = sorted(tqdm(glob.glob(os.path.join(patch_dir, '*.pt')), desc="Finding patches", leave=False, unit=" file"))
        if not self.patch_filenames:
             raise FileNotFoundError(f"FATAL: No '.pt' patch files found in directory: {patch_dir}")
        self.num_patches = len(self.patch_filenames)
        print(f"Initialized dataset from preprocessed patches: '{patch_dir}' ({self.num_patches} patches found).")
    def __len__(self): return self.num_patches
    def __getitem__(self, idx):
        patch_path = self.patch_filenames[idx]
        try:
            data = torch.load(patch_path, map_location='cpu') # Load to CPU first
            return data['lr'].float(), data['hr'].float()
        except Exception as e:
            print(f"\nERROR loading patch file {patch_path}: {e}. Corrupted file? Returning None.")
            return None # Signal error

# collate_fn_skip_none (Unchanged)
def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)

# SR_MLPWithSharedOutputHead (Unchanged)
class SR_MLPWithSharedOutputHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        super(SR_MLPWithSharedOutputHead, self).__init__(); self.activation = nn.ReLU()
        self.input_layer = nn.Linear(input_dim, hidden_dim); self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)])
        self.final_output_head = nn.Linear(hidden_dim, output_dim); self.output_activation = nn.Tanh()
    def forward(self, x):
        x = x.view(x.size(0), -1); representations_to_evaluate = []
        h = self.activation(self.input_layer(x)); representations_to_evaluate.append(h); current_h = h
        for hidden_layer in self.hidden_layers: current_h = self.activation(hidden_layer(current_h)); representations_to_evaluate.append(current_h)
        all_outputs = []
        for h_repr in representations_to_evaluate: prediction_flat = self.final_output_head(h_repr); prediction_flat = self.output_activation(prediction_flat); all_outputs.append(prediction_flat)
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
    if not check_preprocessed_dataset(train_patch_dir, valid_patch_dir):
        sys.exit(1)

    # --- Create Results Directory ---
    results_dir = setup_results_dir(results_base_dir_name)

    # --- Create Datasets and Loaders ---
    try:
        train_dataset = Preprocessed_SR_Patch_Dataset(train_patch_dir)
        test_dataset = Preprocessed_SR_Patch_Dataset(valid_patch_dir)
    except FileNotFoundError as e:
        print(f"Error initializing dataset: {e}"); sys.exit(1)

    print(f"Using {INITIAL_NUM_WORKERS} dataloader workers.")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=INITIAL_NUM_WORKERS, pin_memory=(device.type == 'cuda'), persistent_workers=(INITIAL_NUM_WORKERS > 0), drop_last=True, collate_fn=collate_fn_skip_none)
    # Reduce batch size for testing loader if memory becomes an issue during eval visualization
    test_batch_size = min(batch_size, 128) # Smaller batch for eval potentially
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=INITIAL_NUM_WORKERS, pin_memory=(device.type == 'cuda'), persistent_workers=(INITIAL_NUM_WORKERS > 0), drop_last=False, collate_fn=collate_fn_skip_none)

    print(f"Training config: LR={lr_patch_size}x{lr_patch_size}, HR={hr_patch_size}x{hr_patch_size}")
    print(f"MLP dims: In={input_dim}, Out={output_dim}, Hidden={hidden_dim}, Layers={num_hidden_layers}")

    # --- Instantiate Model, Loss, Optimizer ---
    model = SR_MLPWithSharedOutputHead(input_dim, hidden_dim, output_dim, num_hidden_layers).to(device)
    criterion = nn.MSELoss(); optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Metrics Initialization (Moved here for clarity) ---
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    mae_metric = nn.L1Loss() # Use standard L1Loss for MAE

    # --- Training Loop ---
    losses_history = [[] for _ in range(num_hidden_layers + 1)]; print("--- 开始训练 (Preprocessed Div2k Patches) ---"); print("!!! WARNING: MLP still generally unsuitable for Div2k SR !!!")

    epochs_pbar = tqdm(range(num_epochs), desc="Epochs")
    for epoch in epochs_pbar:
        model.train(); epoch_batch_losses = [[] for _ in range(num_hidden_layers + 1)]; total_epoch_loss_sum = 0.0; processed_batches_in_epoch = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False, unit="batch")
        for batch_idx, batch_data in enumerate(train_pbar):
            if batch_data is None: continue # Skip bad batches
            lr_data, hr_target_norm = batch_data
            current_batch_size = lr_data.size(0)
            if current_batch_size == 0: continue

            lr_data, hr_target_norm = lr_data.to(device), hr_target_norm.to(device)
            hr_target_flat_norm = hr_target_norm.view(current_batch_size, -1)

            # --- Forward, Loss, Backward ---
            try: all_predictions_flat_norm = model(lr_data)
            except Exception as model_e: print(f"\n!!! ERROR Forward Pass E{epoch+1} B{batch_idx}: {model_e}\nLR shape: {lr_data.shape}"); traceback.print_exc(); continue

            optimizer.zero_grad(); total_loss_for_grad = torch.tensor(0.0, device=device); current_batch_losses = []; nan_detected = False
            for i in range(num_hidden_layers + 1): # Iterate through Input Act + Hidden Layers outputs
                if all_predictions_flat_norm[i].shape != hr_target_flat_norm.shape: print(f"\n!!! Shape Mismatch E{epoch+1} B{batch_idx} L{i}: Pred {all_predictions_flat_norm[i].shape}, Target {hr_target_flat_norm.shape}"); nan_detected = True; break
                try:
                    loss_for_grad = criterion(all_predictions_flat_norm[i], hr_target_flat_norm)
                except Exception as loss_e: print(f"\n!!! ERROR Loss Calc E{epoch+1} B{batch_idx} L{i}: {loss_e}"); traceback.print_exc(); nan_detected = True; break
                if torch.isnan(loss_for_grad): print(f"!!! NaN loss E{epoch+1} B{batch_idx} L{i}. Skip batch !!!"); nan_detected = True; break
                total_loss_for_grad = total_loss_for_grad + loss_for_grad
                current_batch_losses.append(loss_for_grad.item())

            if nan_detected: optimizer.zero_grad(); continue

            if len(current_batch_losses) == num_hidden_layers + 1:
                 for i in range(num_hidden_layers + 1): epoch_batch_losses[i].append(current_batch_losses[i])
            else: print(f"Warn: Incomplete losses B{batch_idx}. Skip history."); continue

            try:
                total_loss_for_grad.backward(); optimizer.step();
            except Exception as backward_e: print(f"\n!!! ERROR Backward/Step E{epoch+1} B{batch_idx}: {backward_e}"); traceback.print_exc(); optimizer.zero_grad(); continue

            processed_batches_in_epoch += 1
            total_epoch_loss_sum += total_loss_for_grad.item()
            train_pbar.set_postfix(loss=f"{total_loss_for_grad.item():.4f}", refresh=False)

        # --- Epoch Summary ---
        avg_epoch_total_loss = total_epoch_loss_sum / processed_batches_in_epoch if processed_batches_in_epoch > 0 else float('nan')
        epochs_pbar.set_postfix(avg_loss=f"{avg_epoch_total_loss:.4f}")
        avg_epoch_losses = [np.mean(batch_losses) if batch_losses else float('nan') for batch_losses in epoch_batch_losses]
        tqdm.write(f"\n--- Epoch {epoch+1} Summary ---")
        loss_str = " | ".join([f"L{i}:{l:.6f}" for i, l in enumerate(avg_epoch_losses) if not np.isnan(l)])
        tqdm.write(f"Avg Layer Losses (InputAct, Hidden1...): [{loss_str}]")
        tqdm.write(f"Avg Summed Epoch Loss (for grad): {avg_epoch_total_loss:.6f}\n")
        for i in range(num_hidden_layers + 1):
             while len(losses_history[i]) < epoch: losses_history[i].append(float('nan'))
             losses_history[i].append(avg_epoch_losses[i])

    print("--- 训练完成 ---")

    # --- Plot Loss History ---
    plot_loss_history(losses_history, num_epochs, results_dir)

    # --- Evaluate Model Performance & Save Visualizations ---
    model.eval()
    test_mse_loss_final_layer_norm = 0.0
    test_batches = 0
    saved_vis_count = 0 # Counter for saved visualization examples

    print("\n--- 开始评估 (Preprocessed) & Saving Visualizations ---")
    test_pbar = tqdm(test_loader, desc="Evaluating", leave=False, unit="batch")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_pbar):
            if batch_data is None: continue
            lr_data, hr_target_norm = batch_data # Normalized data from loader
            current_batch_size = lr_data.size(0)
            if current_batch_size == 0: continue

            lr_data, hr_target_norm = lr_data.to(device), hr_target_norm.to(device)
            hr_target_flat_norm = hr_target_norm.view(current_batch_size, -1)

            try:
                # Get model predictions for the whole batch
                all_outputs_flat_norm = model(lr_data)
                final_output_flat_norm = all_outputs_flat_norm[-1]

                # --- Calculate Loss for the batch (on final layer output) ---
                if final_output_flat_norm.shape != hr_target_flat_norm.shape:
                    print(f"!!! Eval Shape Mismatch Batch {batch_idx}: Skip Loss Calc.")
                else:
                    loss = criterion(final_output_flat_norm, hr_target_flat_norm)
                    if not torch.isnan(loss):
                        test_mse_loss_final_layer_norm += loss.item() * current_batch_size # Accumulate total loss
                        test_batches += current_batch_size # Accumulate total samples
                        # Update eval progress bar postfix with running avg loss
                        if test_batches > 0:
                           test_pbar.set_postfix(avg_mse_loss=f"{test_mse_loss_final_layer_norm/test_batches:.4f}")
                    else:
                        print(f"!!! Eval NaN Loss Batch {batch_idx}: Skip Loss.")

                # --- Generate and Save Visualizations for samples in the batch ---
                num_layers_output = len(all_outputs_flat_norm) # Should be num_hidden_layers + 1
                for sample_idx in range(current_batch_size):
                    if saved_vis_count >= NUM_VIS_EXAMPLES:
                        break # Stop saving once we have enough examples

                    # --- Prepare data for this specific sample ---
                    lr_sample_norm = lr_data[sample_idx] # (C, H_lr, W_lr), normalized
                    hr_gt_norm = hr_target_norm[sample_idx] # (C, H_hr, W_hr), normalized

                    # Denormalize GT for visualization and metric comparison
                    hr_gt_denorm = denormalize_tensor(hr_gt_norm) # (C, H_hr, W_hr), [0, 1] range
                    hr_gt_denorm_vis = hr_gt_denorm.cpu().permute(1, 2, 0).numpy() # HWC for imshow

                    # Denormalize LR for visualization (upscaled)
                    lr_sample_denorm = denormalize_tensor(lr_sample_norm) # (C, H_lr, W_lr), [0, 1]
                    # Upscale LR using bicubic for display comparison
                    lr_display_upscaled = F.interpolate(
                        lr_sample_denorm.unsqueeze(0), # Add batch dim
                        size=(hr_patch_size, hr_patch_size),
                        mode='bicubic',
                        align_corners=False
                    ).squeeze(0) # Remove batch dim
                    lr_display_upscaled_vis = lr_display_upscaled.cpu().permute(1, 2, 0).numpy() # HWC

                    # --- Prepare predicted outputs for this sample ---
                    predicted_patches_denorm = []
                    metrics_per_layer = []
                    gt_tensor_for_metrics = hr_gt_denorm.unsqueeze(0).to(device) # (1, C, H, W), [0,1], on device

                    for i in range(num_layers_output):
                        # Extract the prediction for this sample from the batch output
                        pred_flat_norm_sample = all_outputs_flat_norm[i][sample_idx]
                        # Reshape and denormalize
                        pred_patch_norm = pred_flat_norm_sample.view(1, channels, hr_patch_size, hr_patch_size)
                        pred_patch_denorm = denormalize_tensor(pred_patch_norm) # (1, C, H, W), [0, 1]

                        # Store for plotting (move to CPU, permute)
                        predicted_patches_denorm.append(pred_patch_denorm.squeeze(0).cpu().permute(1, 2, 0).numpy())

                        # Calculate metrics
                        pred_tensor_for_metrics = pred_patch_denorm.to(device) # Ensure it's on device
                        try:
                             mse_val = F.mse_loss(pred_tensor_for_metrics, gt_tensor_for_metrics).item()
                             mae_val = mae_metric(pred_tensor_for_metrics, gt_tensor_for_metrics).item()
                             psnr_val = psnr_metric(pred_tensor_for_metrics, gt_tensor_for_metrics).item()
                             # SSIM needs batch dim, ensure inputs are float
                             ssim_val = ssim_metric(pred_tensor_for_metrics.float(), gt_tensor_for_metrics.float()).item()
                        except ValueError as e:
                            # Common error: "Expected input batch_size (1) to match target batch_size (xx)" - check tensor shapes
                            # Common error: Window size > image size
                             print(f"WARN: Metric calc failed L{i} Sample {saved_vis_count+1}: {e}")
                             print(f"  Pred shape: {pred_tensor_for_metrics.shape}, GT shape: {gt_tensor_for_metrics.shape}")
                             mse_val, mae_val, psnr_val, ssim_val = float('nan'), float('nan'), float('nan'), float('nan')
                        except Exception as e:
                             print(f"WARN: Metric calc failed L{i} Sample {saved_vis_count+1}: {e}")
                             mse_val, mae_val, psnr_val, ssim_val = float('nan'), float('nan'), float('nan'), float('nan')

                        metrics = {'MSE': mse_val, 'MAE': mae_val, 'PSNR': psnr_val, 'SSIM': ssim_val}
                        metrics_per_layer.append(metrics)

                    # --- Create and Save the Plot for this sample ---
                    num_layers_to_plot = num_layers_output
                    num_total_plots = 2 + num_layers_to_plot # LR, HR_GT, Preds
                    fig, axes = plt.subplots(1, num_total_plots, figsize=(3 * num_total_plots, 3.5)) # Adjusted size

                    # Plot LR Input (Upscaled)
                    imshow_patch(axes[0], lr_display_upscaled_vis, f"Input LR ({lr_patch_size}x{lr_patch_size} upscaled)")

                    # Plot Ground Truth HR
                    imshow_patch(axes[1], hr_gt_denorm_vis, f"Ground Truth HR ({hr_patch_size}x{hr_patch_size})")

                    # Plot Predictions from each layer
                    for i in range(num_layers_to_plot):
                        ax_pred = axes[i + 2]
                        pred_patch_denorm_numpy = predicted_patches_denorm[i]
                        layer_name = f'Input Act.' if i == 0 else f'Hidden {i} Act.'
                        metrics = metrics_per_layer[i]
                        # Format title string carefully to handle potential NaNs
                        title = (f"Pred. from {layer_name}\n"
                                 f"MSE:{metrics.get('MSE', float('nan')):.3f} MAE:{metrics.get('MAE', float('nan')):.3f}\n"
                                 f"PSNR:{metrics.get('PSNR', float('nan')):.2f} SSIM:{metrics.get('SSIM', float('nan')):.3f}")
                        imshow_patch(ax_pred, pred_patch_denorm_numpy, title)

                    plt.tight_layout(pad=0.3, h_pad=0.5) # Adjust padding
                    vis_plot_path = os.path.join(results_dir, f'eval_sample_{saved_vis_count + 1}.png')
                    plt.savefig(vis_plot_path)
                    plt.close(fig) # Close the figure to free memory
                    # print(f"Saved visualization: {vis_plot_path}") # Optional: print confirmation

                    saved_vis_count += 1 # Increment counter

                # Break outer loop (batch loop) if enough samples are saved
                if saved_vis_count >= NUM_VIS_EXAMPLES:
                    print(f"\nSaved required {NUM_VIS_EXAMPLES} visualization examples.")
                    # We could break here, but let's finish evaluating the whole test set for the average loss
                    # break # Uncomment if you want to stop evaluation early after saving images

            except Exception as eval_e:
                print(f"\nError during eval batch {batch_idx}: {eval_e}");
                traceback.print_exc();
                continue # Continue to next batch

        # End of evaluation loop

    # --- Final Test Summary ---
    if test_batches > 0:
        final_avg_mse = test_mse_loss_final_layer_norm / test_batches
        print(f'\n--- Test Evaluation Finished ---')
        print(f'Test set Avg MSE loss (Final Layer, {test_batches} samples): {final_avg_mse:.6f}\n')
    else:
        print("\nTest set: No valid batches processed for loss calculation.\n")

    if saved_vis_count < NUM_VIS_EXAMPLES:
         print(f"\nWarning: Only saved {saved_vis_count}/{NUM_VIS_EXAMPLES} visualization examples (dataset might be smaller than expected or errors occurred).")


    # --- Visualize Single Sample Patch (Optional - Keep or Remove) ---
    # This section is now redundant as visualization happens during eval loop.
    # You can comment it out or remove it.
    # print("Skipping final single patch visualization (done during evaluation loop).")


    print("\n--- Script Finished ---")
    print(f"Using PREPROCESSED Div2k dataset. Results in: {results_dir}")
    if INITIAL_NUM_WORKERS == 0: print("\nNOTE: DataLoader workers = 0.")
    elif INITIAL_NUM_WORKERS > 8: print(f"\nNOTE: Using {INITIAL_NUM_WORKERS} workers.")


# End of the if __name__ == '__main__': block
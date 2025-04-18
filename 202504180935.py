# 2025年4月18日 - Training with Preprocessed Patches (Final Layer Loss Only)
# ... (Imports remain the same) ...
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

# Results directory naming convention
results_base_dir_name = "results_div2k_manual_sr_mlp_patch_FINAL_loss_metrics_PREPROCESSED" # Updated name
now = datetime.datetime.now()
timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
results_base_dir = "results"
results_dir = os.path.join(results_base_dir, results_base_dir_name, timestamp_str)

# Model and Training Parameters
channels = 3
input_dim = lr_patch_size * lr_patch_size * channels
output_dim = hr_patch_size * hr_patch_size * channels
hidden_dim = 512
num_hidden_layers = 4
num_epochs = 10 # Adjusted epochs
batch_size = 1024
learning_rate = 1e-4
INITIAL_NUM_WORKERS = 16
NUM_VIS_EXAMPLES = 5

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
        # ... (rest of error messages) ...
        return False

# setup_results_dir (Unchanged)
def setup_results_dir(base_dir_name):
    now = datetime.datetime.now(); timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    base_results_path = "results"
    full_results_dir = os.path.join(base_results_path, base_dir_name, timestamp_str)
    os.makedirs(full_results_dir, exist_ok=True)
    print(f"结果将保存在: {full_results_dir}"); return full_results_dir

# plot_loss_history (Modified to potentially highlight final loss)
def plot_loss_history(losses_history, num_epochs, results_dir):
    plt.figure(figsize=(12, 7)); epochs_range = range(1, num_epochs + 1)
    num_actual_layers = len(losses_history)
    final_layer_idx = num_actual_layers - 1

    for i in range(num_actual_layers):
        if not losses_history[i]: continue
        is_final_layer = (i == final_layer_idx)
        label_prefix = f'Final Layer (Used for Grad)' if is_final_layer else f'Intermediate Layer {i}'
        label = f'{label_prefix} MSE'
        line_style = '-' if is_final_layer else ':' # Different style for final layer
        line_width = 2.0 if is_final_layer else 1.0

        valid_indices = [idx for idx, l in enumerate(losses_history[i]) if l is not None and not np.isnan(l)]
        if not valid_indices: continue
        valid_epochs = [epochs_range[idx] for idx in valid_indices]
        valid_losses = [losses_history[i][idx] for idx in valid_indices]

        if valid_losses:
            plt.plot(valid_epochs, valid_losses, label=label, marker='o', linestyle=line_style, linewidth=line_width, markersize=4)

    # No need to plot the "Sum of Average MSEs" anymore as it wasn't used for training
    # Optional: Plot only the final layer loss if intermediate ones are too noisy

    plt.title('Div2k Patch SR Training: Loss History (Preprocessed, Final Layer Grad)'); plt.xlabel('Epoch'); plt.ylabel('Avg MSE Loss (Norm. Data)');
    if num_epochs > 0:
        tick_step = max(1, num_epochs // 10); plt.xticks(range(1, num_epochs + 1, tick_step))
    plt.legend(fontsize='small', loc='best'); plt.grid(True); plt.yscale('log') # Changed legend location
    loss_plot_path = os.path.join(results_dir, 'div2k_manual_patch_sr_loss_history_final_grad.png')
    plt.savefig(loss_plot_path); print(f"Loss plot saved: {loss_plot_path}"); plt.close()


# imshow_patch (Unchanged)
def imshow_patch(ax, img_numpy, title):
     ax.imshow(np.clip(img_numpy, 0, 1)); ax.set_title(title, fontsize=8); ax.axis('off')

# denormalize_tensor (Unchanged)
def denormalize_tensor(tensor):
    denorm_tensor = tensor.clone(); _mean = norm_mean.to(denorm_tensor.device); _std = norm_std.to(denorm_tensor.device)
    if denorm_tensor.ndim == 4: _mean = _mean.unsqueeze(0); _std = _std.unsqueeze(0); denorm_tensor = denorm_tensor * _std + _mean
    elif denorm_tensor.ndim == 3: denorm_tensor = denorm_tensor * _std + _mean
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
        try: data = torch.load(patch_path, map_location='cpu'); return data['lr'].float(), data['hr'].float()
        except Exception as e: print(f"\nERROR loading {patch_path}: {e}. Returning None."); return None

# collate_fn_skip_none (Unchanged)
def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch));
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)

# SR_MLPWithSharedOutputHead (Unchanged - Forward pass generates all outputs)
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
    if not check_preprocessed_dataset(train_patch_dir, valid_patch_dir): sys.exit(1)

    # --- Create Results Directory ---
    results_dir = setup_results_dir(results_base_dir_name)

    # --- Create Datasets and Loaders ---
    try:
        train_dataset = Preprocessed_SR_Patch_Dataset(train_patch_dir)
        test_dataset = Preprocessed_SR_Patch_Dataset(valid_patch_dir)
    except FileNotFoundError as e: print(f"Error initializing dataset: {e}"); sys.exit(1)

    print(f"Using {INITIAL_NUM_WORKERS} dataloader workers.")
    # Set persistent_workers=False to prevent hangs
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=INITIAL_NUM_WORKERS, pin_memory=(device.type == 'cuda'), persistent_workers=False, drop_last=True, collate_fn=collate_fn_skip_none)
    test_batch_size = min(batch_size, 128)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=INITIAL_NUM_WORKERS, pin_memory=(device.type == 'cuda'), persistent_workers=False, drop_last=False, collate_fn=collate_fn_skip_none)

    print(f"Training config: LR={lr_patch_size}x{lr_patch_size}, HR={hr_patch_size}x{hr_patch_size}")
    print(f"MLP dims: In={input_dim}, Out={output_dim}, Hidden={hidden_dim}, Layers={num_hidden_layers}")

    # --- Instantiate Model, Loss, Optimizer ---
    model = SR_MLPWithSharedOutputHead(input_dim, hidden_dim, output_dim, num_hidden_layers).to(device)
    criterion = nn.MSELoss(); optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Metrics Initialization ---
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    mae_metric = nn.L1Loss()

    # --- Training Loop ---
    # losses_history stores *all* layer losses for logging/plotting
    losses_history = [[] for _ in range(num_hidden_layers + 1)]
    print("--- 开始训练 (Preprocessed Div2k Patches - Final Layer Loss Only) ---")
    print("!!! WARNING: MLP still generally unsuitable for Div2k SR !!!")

    epochs_pbar = tqdm(range(num_epochs), desc="Epochs")
    for epoch in epochs_pbar:
        model.train()
        # epoch_batch_losses stores losses for each layer *within* the current epoch
        epoch_batch_losses = [[] for _ in range(num_hidden_layers + 1)]
        # <<< MODIFIED: Track sum of final layer loss for epoch avg >>>
        total_epoch_final_loss_sum = 0.0
        processed_batches_in_epoch = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False, unit="batch")
        for batch_idx, batch_data in enumerate(train_pbar):
            if batch_data is None: continue
            lr_data, hr_target_norm = batch_data
            current_batch_size = lr_data.size(0)
            if current_batch_size == 0: continue

            lr_data, hr_target_norm = lr_data.to(device), hr_target_norm.to(device)
            hr_target_flat_norm = hr_target_norm.view(current_batch_size, -1)

            # --- Forward Pass (Generates all intermediate outputs) ---
            try: all_predictions_flat_norm = model(lr_data)
            except Exception as model_e: print(f"\n!!! ERROR Forward E{epoch+1} B{batch_idx}: {model_e}"); traceback.print_exc(); continue

            # --- Loss Calculation & Backward (Using Final Layer Only) ---
            optimizer.zero_grad()
            current_batch_losses_items = [] # For storing .item() of all layer losses
            final_loss_for_grad = None    # Store the tensor loss of the final layer
            nan_detected = False

            for i in range(num_hidden_layers + 1): # Index 0=InputAct, 1=Hidden1Act, ..., N=FinalHiddenAct
                pred_norm = all_predictions_flat_norm[i]
                if pred_norm.shape != hr_target_flat_norm.shape:
                    print(f"\n!!! Shape Mismatch E{epoch+1} B{batch_idx} L{i}"); nan_detected = True; break
                try:
                    # Calculate loss for layer i (for logging)
                    loss_i = criterion(pred_norm, hr_target_flat_norm)
                except Exception as loss_e:
                    print(f"\n!!! ERROR Loss Calc E{epoch+1} B{batch_idx} L{i}: {loss_e}"); traceback.print_exc(); nan_detected = True; break
                if torch.isnan(loss_i):
                    print(f"!!! NaN loss E{epoch+1} B{batch_idx} L{i}. Skip batch !!!"); nan_detected = True; break

                # Store the calculated loss value for logging history
                current_batch_losses_items.append(loss_i.item())

                # <<< MODIFIED: Identify and store the loss tensor for the FINAL layer >>>
                # The final prediction uses the representation from the last hidden layer (index num_hidden_layers)
                if i == num_hidden_layers:
                    final_loss_for_grad = loss_i

            # If NaN/error occurred OR final loss wasn't reached/calculated, skip batch gradient update
            if nan_detected or final_loss_for_grad is None:
                optimizer.zero_grad() # Ensure grads are zeroed even if backward isn't called
                continue

            # Log all calculated layer losses for this batch
            if len(current_batch_losses_items) == num_hidden_layers + 1:
                for i in range(num_hidden_layers + 1):
                    epoch_batch_losses[i].append(current_batch_losses_items[i])
            else:
                # Should be less likely now, but keep as safeguard
                print(f"Warn: Incomplete losses logged B{batch_idx}. Skip history append."); continue

            # <<< MODIFIED: Perform backward pass ONLY on the final layer's loss >>>
            try:
                final_loss_for_grad.backward()
                optimizer.step()
            except Exception as backward_e:
                print(f"\n!!! ERROR Backward/Step E{epoch+1} B{batch_idx}: {backward_e}"); traceback.print_exc(); optimizer.zero_grad(); continue

            processed_batches_in_epoch += 1
            # <<< MODIFIED: Update total sum using only the final loss item >>>
            total_epoch_final_loss_sum += final_loss_for_grad.item()
            # Update tqdm postfix showing the final loss used for grad update
            train_pbar.set_postfix(final_loss=f"{final_loss_for_grad.item():.4f}", refresh=False)

        # --- Epoch Summary ---
        # Calculate average of the FINAL layer loss across the epoch
        avg_epoch_final_loss = total_epoch_final_loss_sum / processed_batches_in_epoch if processed_batches_in_epoch > 0 else float('nan')
        # Calculate average for EACH layer's logged loss
        avg_epoch_layer_losses = [np.mean(batch_losses) if batch_losses else float('nan') for batch_losses in epoch_batch_losses]

        # Update epoch progress bar
        epochs_pbar.set_postfix(avg_final_loss=f"{avg_epoch_final_loss:.4f}")

        # Print epoch summary using tqdm.write
        tqdm.write(f"\n--- Epoch {epoch+1} Summary ---")
        loss_str = " | ".join([f"L{i}:{l:.6f}" for i, l in enumerate(avg_epoch_layer_losses) if not np.isnan(l)])
        tqdm.write(f"Avg Logged Layer Losses (InputAct, Hidden1...): [{loss_str}]")
        tqdm.write(f"Avg FINAL Layer Loss (Used for Grad): {avg_epoch_final_loss:.6f}\n") # Clearly state which loss drove training

        # Store history of avg losses for EACH layer
        for i in range(num_hidden_layers + 1):
             while len(losses_history[i]) < epoch: losses_history[i].append(float('nan'))
             losses_history[i].append(avg_epoch_layer_losses[i])

    print("--- 训练完成 ---")

    # --- Plot Loss History ---
    # The plot function now expects history of all layers, but only the last one drove training.
    plot_loss_history(losses_history, num_epochs, results_dir)

    # --- Evaluate Model Performance & Save Visualizations ---
    # Evaluation logic remains the same (calculates final loss, visualizes all layers)
    model.eval()
    test_mse_loss_final_layer_norm = 0.0
    test_samples_count = 0 # Renamed for clarity
    saved_vis_count = 0

    print("\n--- 开始评估 (Preprocessed) & Saving Visualizations ---")
    test_pbar = tqdm(test_loader, desc="Evaluating", leave=False, unit="batch")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_pbar):
            if batch_data is None: continue
            lr_data, hr_target_norm = batch_data
            current_batch_size = lr_data.size(0)
            if current_batch_size == 0: continue

            lr_data, hr_target_norm = lr_data.to(device), hr_target_norm.to(device)
            hr_target_flat_norm = hr_target_norm.view(current_batch_size, -1)

            try:
                all_outputs_flat_norm = model(lr_data)
                final_output_flat_norm = all_outputs_flat_norm[-1] # Index -1 is equivalent to index num_hidden_layers

                # --- Calculate FINAL Loss for the batch (for reporting avg test loss) ---
                if final_output_flat_norm.shape != hr_target_flat_norm.shape:
                    print(f"!!! Eval Shape Mismatch B{batch_idx}")
                else:
                    loss = criterion(final_output_flat_norm, hr_target_flat_norm)
                    if not torch.isnan(loss):
                        test_mse_loss_final_layer_norm += loss.item() * current_batch_size
                        test_samples_count += current_batch_size
                        if test_samples_count > 0:
                           test_pbar.set_postfix(avg_final_mse=f"{test_mse_loss_final_layer_norm/test_samples_count:.4f}")
                    else: print(f"!!! Eval NaN Loss B{batch_idx}")

                # --- Generate and Save Visualizations (uses all layers' outputs) ---
                num_layers_output = len(all_outputs_flat_norm)
                for sample_idx in range(current_batch_size):
                    if saved_vis_count >= NUM_VIS_EXAMPLES: break

                    lr_sample_norm = lr_data[sample_idx]; hr_gt_norm = hr_target_norm[sample_idx]
                    hr_gt_denorm = denormalize_tensor(hr_gt_norm)
                    hr_gt_denorm_vis = hr_gt_denorm.cpu().permute(1, 2, 0).numpy()
                    lr_sample_denorm = denormalize_tensor(lr_sample_norm)
                    lr_display_upscaled = F.interpolate(lr_sample_denorm.unsqueeze(0), size=(hr_patch_size, hr_patch_size), mode='bicubic', align_corners=False).squeeze(0)
                    lr_display_upscaled_vis = lr_display_upscaled.cpu().permute(1, 2, 0).numpy()

                    predicted_patches_denorm = []; metrics_per_layer = []
                    gt_tensor_for_metrics = hr_gt_denorm.unsqueeze(0).to(device)

                    for i in range(num_layers_output):
                        pred_flat_norm_sample = all_outputs_flat_norm[i][sample_idx]
                        pred_patch_norm = pred_flat_norm_sample.view(1, channels, hr_patch_size, hr_patch_size)
                        pred_patch_denorm = denormalize_tensor(pred_patch_norm)
                        predicted_patches_denorm.append(pred_patch_denorm.squeeze(0).cpu().permute(1, 2, 0).numpy())
                        pred_tensor_for_metrics = pred_patch_denorm.to(device)
                        try:
                             mse_val = F.mse_loss(pred_tensor_for_metrics, gt_tensor_for_metrics).item()
                             mae_val = mae_metric(pred_tensor_for_metrics, gt_tensor_for_metrics).item()
                             psnr_val = psnr_metric(pred_tensor_for_metrics, gt_tensor_for_metrics).item()
                             ssim_val = ssim_metric(pred_tensor_for_metrics.float(), gt_tensor_for_metrics.float()).item()
                        except Exception as e:
                             print(f"WARN: Metric calc failed L{i} Sample {saved_vis_count+1}: {e}"); mse_val, mae_val, psnr_val, ssim_val = [float('nan')]*4
                        metrics = {'MSE': mse_val, 'MAE': mae_val, 'PSNR': psnr_val, 'SSIM': ssim_val}; metrics_per_layer.append(metrics)

                    # Plotting logic remains the same
                    num_layers_to_plot = num_layers_output
                    num_total_plots = 2 + num_layers_to_plot
                    fig, axes = plt.subplots(1, num_total_plots, figsize=(3 * num_total_plots, 3.5))
                    imshow_patch(axes[0], lr_display_upscaled_vis, f"Input LR ({lr_patch_size}x{lr_patch_size} upscaled)")
                    imshow_patch(axes[1], hr_gt_denorm_vis, f"Ground Truth HR ({hr_patch_size}x{hr_patch_size})")
                    for i in range(num_layers_to_plot):
                        ax_pred = axes[i + 2]; pred_patch_denorm_numpy = predicted_patches_denorm[i]
                        layer_name = f'Input Act.' if i == 0 else f'Hidden {i} Act.'
                        metrics = metrics_per_layer[i]
                        title = (f"Pred. from {layer_name}\n"
                                 f"MSE:{metrics.get('MSE', float('nan')):.3f} MAE:{metrics.get('MAE', float('nan')):.3f}\n"
                                 f"PSNR:{metrics.get('PSNR', float('nan')):.2f} SSIM:{metrics.get('SSIM', float('nan')):.3f}")
                        imshow_patch(ax_pred, pred_patch_denorm_numpy, title)
                    plt.tight_layout(pad=0.3, h_pad=0.5); vis_plot_path = os.path.join(results_dir, f'eval_sample_{saved_vis_count + 1}.png')
                    plt.savefig(vis_plot_path); plt.close(fig)
                    saved_vis_count += 1

                if saved_vis_count >= NUM_VIS_EXAMPLES: print(f"\nSaved required {NUM_VIS_EXAMPLES} visualization examples.") # No need to break eval loop

            except Exception as eval_e: print(f"\nError during eval batch {batch_idx}: {eval_e}"); traceback.print_exc(); continue

    # --- Final Test Summary ---
    if test_samples_count > 0:
        final_avg_mse = test_mse_loss_final_layer_norm / test_samples_count
        print(f'\n--- Test Evaluation Finished ---')
        print(f'Test set Avg FINAL layer MSE loss ({test_samples_count} samples): {final_avg_mse:.6f}\n')
    else: print("\nTest set: No valid batches processed for loss calculation.\n")
    if saved_vis_count < NUM_VIS_EXAMPLES: print(f"Warning: Only saved {saved_vis_count}/{NUM_VIS_EXAMPLES} visualization examples.")

    print("\n--- Script Finished ---")
    print(f"Using PREPROCESSED Div2k dataset. Results in: {results_dir}")
    if INITIAL_NUM_WORKERS > 8: print(f"NOTE: Using {INITIAL_NUM_WORKERS} workers.")

# End of the if __name__ == '__main__': block
#Kóði sem greinir módelin (þjálfar ekki!) og teiknar viðeigani grafík fyrir þau.

# --- IMPORTS ---
import torch
import torch.nn as nn
import torch.optim as optim # Not strictly needed for eval, but maybe for helpers
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models # Keep models for potential future use, though not ResNet here
# from torch.optim import lr_scheduler # Not needed for eval

import numpy as np
import copy
import os
import re # Needed for parsing checkpoint filenames
from PIL import Image # Keep PIL for image loading helpers
# from PIL import Image, ImageOps # ImageOps not needed if not cropping
import matplotlib.pyplot as plt
import math # For feature map grid calculation if used
import seaborn as sns
from sklearn.metrics import (roc_auc_score, roc_curve, auc, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, classification_report)
# from sklearn.utils.class_weight import compute_class_weight # Not needed for eval
import time
import warnings

# Filter warnings (optional)
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Google Drive Integration ---
import shutil
from google.colab import drive
try:
    print("Force remounting drive...")
    drive.mount('/content/drive', force_remount=True)
    print("Drive remounted.")
except Exception as e:
    print(f"Error during remount: {e}")

# --- Configuration ---
# ** IMPORTANT: Point this to the *final* model saved by GrowingCNN script **
# Example: Might look like 'final_model_gen_3_layers_4.pth' if NUM_GENERATIONS=4
# You MUST determine the correct filename for the model you want to evaluate.
FINAL_MODEL_FILENAME = 'final_model_trained_gen_9_layers_21.pth' # <--- CHANGE THIS FILENAME

# --- Checkpoint/Model Parameters (MUST match the GrowingCNN training run) ---
NUM_CLASSES = 2
IMG_HEIGHT, IMG_WIDTH = 299, 299 # Match GrowingCNN training
BASE_CONV_OUT_CHANNELS = 16 # Match GrowingCNN training
EXTRA_CONV_CHANNELS = 32 # Match GrowingCNN training

# --- Paths (MUST match GrowingCNN training) ---
DRIVE_MOUNT_POINT = '/content/drive'
# Directory where the GrowingCNN checkpoints AND final model are saved
MODEL_SAVE_DIR = os.path.join(DRIVE_MOUNT_POINT, 'MyDrive', 'Gervigreind - verkefni', 'LayerGrowth')
FINAL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, FINAL_MODEL_FILENAME)

# We might need the *latest checkpoint file* to determine the number of layers
CHECKPOINT_BASENAME = 'layer_growth_checkpoint_gen_'
CHECKPOINT_SUFFIX = '.pth'

# Data directory for the TEST set used in the original evaluation script
TEST_DATA_DIR = '/content/Test_images' # Or use VAL_DATA_DIR if that's your test set
# TEST_DATA_DIR = '/content/Validation_images' # Alternative if using validation set for final test

BATCH_SIZE = 32 # Can be larger for evaluation if memory allows

# --- Setup Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Mount Google Drive ---
print("Mounting Google Drive...")
try:
    drive.mount(DRIVE_MOUNT_POINT, force_remount=True)
    print("Google Drive mounted successfully.")
    # Check if model save directory exists
    if not os.path.isdir(MODEL_SAVE_DIR):
         print(f"Error: Model save directory not found: {MODEL_SAVE_DIR}")
         exit()
except Exception as e:
    print(f"Error mounting Google Drive: {e}. Exiting.")
    exit()

# --- 1. Define GrowingCNN Model Architecture (Copy from training script) ---
class GrowingCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, num_extra_conv_layers=0):
        super(GrowingCNN, self).__init__()
        self.num_extra_conv_layers = num_extra_conv_layers
        # Use constants defined above
        # print(f"  Instantiating GrowingCNN with {num_extra_conv_layers} extra conv layers for evaluation.")

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=BASE_CONV_OUT_CHANNELS, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU(); self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.extra_conv_layers = nn.ModuleList()
        current_channels = BASE_CONV_OUT_CHANNELS
        for i in range(num_extra_conv_layers):
            conv_layer = nn.Conv2d(in_channels=current_channels, out_channels=EXTRA_CONV_CHANNELS, kernel_size=3, stride=1, padding=1)
            self.extra_conv_layers.append(conv_layer); self.extra_conv_layers.append(nn.ReLU()); current_channels = EXTRA_CONV_CHANNELS
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._calculate_flattened_size(); self.fc1 = nn.Linear(self.flattened_size, 128); self.relu_fc1 = nn.ReLU(); self.fc2 = nn.Linear(128, num_classes)
    def _calculate_flattened_size(self):
        try:
            with torch.no_grad(): dummy_input = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH); x = self.pool1(self.relu1(self.conv1(dummy_input))); [x := layer(x) for layer in self.extra_conv_layers]; x = self.pool2(x); self.flattened_size = int(np.prod(x.shape[1:]))
        except Exception as e: print(f"Warning: Error calculating flattened size ({e}). Using fallback."); self.flattened_size = 32 * 74 * 74 # ADJUST FALLBACK IF NEEDED
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x))); [x := layer(x) for layer in self.extra_conv_layers]; x = self.pool2(x)
        x = x.view(x.size(0), -1); x = self.relu_fc1(self.fc1(x)); x = self.fc2(x); return x


# --- 2. Helper Functions (Copied/Adapted from ResNet Script) ---

# Simple plotting function (if needed for history, but we don't have history here)
# def plot_history(...): ...

def show_augmented_image(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], title="Sample Image"):
    """Displays a single image tensor after reversing normalization."""
    img = img_tensor.cpu().clone()
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    img = img * std + mean
    img = img.permute(1, 2, 0)
    img = torch.clamp(img, 0, 1)
    plt.figure(figsize=(6, 6))
    plt.imshow(img.numpy())
    plt.title(title)
    plt.axis('off')
    plt.show()

# --- MODIFIED Feature Map Plotting (Adapting Layer Names) ---
def plot_feature_maps(model, layer_name, image_tensor, device, max_maps=16):
    """Plots feature maps from a specific layer in GrowingCNN."""
    model.eval(); feature_maps = None; hook_handle = None
    def hook_fn(module, input, output): nonlocal feature_maps; feature_maps = output.detach().clone()
    target_layer = None; found = False

    # Attempt to access layers by standard names or within ModuleList
    if hasattr(model, layer_name):
        target_layer = getattr(model, layer_name)
        found = True
    # --- Adaptation for extra_conv_layers ---
    # Example: Access the *first* extra Conv layer (index 0 in ModuleList)
    elif layer_name == 'extra_conv_0' and hasattr(model, 'extra_conv_layers') and len(model.extra_conv_layers) > 0:
         target_layer = model.extra_conv_layers[0] # Get the Conv layer itself
         found = True
         layer_name = 'extra_conv_layers[0]' # Update name for title
    # Example: Access the *last* extra Conv layer
    elif layer_name == 'extra_conv_last' and hasattr(model, 'extra_conv_layers') and len(model.extra_conv_layers) > 0:
         # Find the last Conv layer in the list (every other element is ReLU)
         last_conv_idx = -1
         for idx, lyr in enumerate(reversed(model.extra_conv_layers)):
             if isinstance(lyr, nn.Conv2d):
                 last_conv_idx = len(model.extra_conv_layers) - 1 - idx
                 break
         if last_conv_idx != -1:
             target_layer = model.extra_conv_layers[last_conv_idx]
             found = True
             layer_name = f'extra_conv_layers[{last_conv_idx}] (Last Conv)'

    # Add more elif clauses here to access specific extra layers if needed

    if not found: print(f"Error: Layer '{layer_name}' or adapted name not found/accessible in GrowingCNN."); return

    hook_handle = target_layer.register_forward_hook(hook_fn)
    image_tensor = image_tensor.to(device)
    with torch.no_grad(): _ = model(image_tensor)
    if hook_handle: hook_handle.remove()
    if feature_maps is None: print(f"Error: Failed to capture maps for '{layer_name}'."); return

    feature_maps_np = feature_maps.squeeze(0).cpu().numpy(); num_channels = feature_maps_np.shape[0]
    maps_to_show = min(num_channels, max_maps)
    print(f"Plotting {maps_to_show}/{num_channels} maps from '{layer_name}' (Shape: {feature_maps_np.shape[1:]})")
    grid_size = math.ceil(math.sqrt(maps_to_show))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(max(grid_size*1.5,6), max(grid_size*1.5,6)))
    if maps_to_show == 1: axes = np.array([axes]) # Make single subplot iterable
    axes = axes.flatten()
    for i in range(maps_to_show):
        ax = axes[i]; feature_map = feature_maps_np[i, :, :]; im = ax.imshow(feature_map, cmap='viridis')
        ax.set_title(f'Map {i+1}', fontsize=8); ax.axis('off')
    for j in range(maps_to_show, len(axes)): axes[j].axis('off')
    fig.suptitle(f'Feature Maps: {layer_name}', fontsize=14); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

def display_image_from_path(filepath, title="Image"):
    """Displays an image loaded from a file path."""
    try:
        img = Image.open(filepath).convert('RGB')
        plt.figure(figsize=(5,5))
        plt.imshow(img)
        plt.title(title, fontsize=10)
        plt.axis('off')
        plt.show()
    except Exception as e: print(f"Error displaying image {filepath}: {e}")

# --- Detailed Evaluation Function (Copied from ResNet script) ---
def evaluate_model_detailed(model, data_loader, device, num_classes):
    """Evaluates model and returns detailed predictions, targets, probabilities, and file paths."""
    model.eval()
    all_targets = []
    all_predictions = []
    all_probas = []
    all_filepaths = []
    filepaths_in_order = None
    current_idx = 0
    try:
        if isinstance(data_loader.dataset, datasets.ImageFolder): filepaths_in_order = [item[0] for item in data_loader.dataset.samples]
        else: print("Warning: Cannot get file paths, dataset is not an ImageFolder.")
    except Exception as e: print(f"Warning: Error getting file paths: {e}")

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device); target_np = target.cpu().numpy()
            outputs = model(data); pred_prob = torch.softmax(outputs, dim=1)
            proba_pos_class = pred_prob[:, 1].cpu().numpy() if num_classes == 2 else None
            _, predicted = torch.max(pred_prob.data, 1); predicted_np = predicted.cpu().numpy()
            all_targets.extend(target_np); all_predictions.extend(predicted_np)
            if proba_pos_class is not None: all_probas.extend(proba_pos_class)
            if filepaths_in_order is not None:
                batch_size = data.size(0); batch_indices = list(range(current_idx, current_idx + batch_size))
                all_filepaths.extend([filepaths_in_order[i] for i in batch_indices]); current_idx += batch_size

    all_targets = np.array(all_targets); all_predictions = np.array(all_predictions)
    all_probas = np.array(all_probas) if num_classes == 2 else None
    return all_targets, all_predictions, all_probas, all_filepaths

# --- 3. Data Loading for Evaluation ---
# Use transforms consistent with GrowingCNN training
eval_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)), # 299x299
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = []
try:
    test_dataset = datasets.ImageFolder(TEST_DATA_DIR, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    class_names = test_dataset.classes
    print(f"Loaded TEST dataset from: {TEST_DATA_DIR} ({len(test_dataset)} samples)")
    print(f"Test classes: {class_names}")
    if len(class_names) != NUM_CLASSES: print("Warning: Class mismatch in test set!")
except Exception as e: print(f"Error loading TEST dataset: {e}"); raise

# --- 4. Determine Number of Layers from Checkpoint or Filename ---
num_extra_layers_for_model = -1 # Initialize with invalid value

# Option A: Parse from filename (less robust if naming changes)
print(f"\n--- Determining architecture for model: {FINAL_MODEL_FILENAME} ---")
match = re.search(r'_layers_(\d+)\.pth$', FINAL_MODEL_FILENAME)
if match:
    try:
        num_extra_layers_for_model = int(match.group(1))
        print(f"  Inferred {num_extra_layers_for_model} extra layers from filename.")
    except ValueError:
        print(f"  Warning: Could not parse layer count from filename '{FINAL_MODEL_FILENAME}'.")

# Option B: Load latest checkpoint to get layer count (more robust)
# This is preferred if Option A fails or filename doesn't contain layer info
if num_extra_layers_for_model == -1:
    print("  Attempting to load latest checkpoint to determine layer count...")
    latest_gen_index = -1
    latest_checkpoint_path = None
    try:
        checkpoint_files = [f for f in os.listdir(MODEL_SAVE_DIR) if f.startswith(CHECKPOINT_BASENAME) and f.endswith(CHECKPOINT_SUFFIX)]
        for filename in checkpoint_files:
            match_cp = re.search(r'_gen_(\d+)\.pth$', filename)
            if match_cp:
                try:
                    gen_index = int(match_cp.group(1))
                    if gen_index > latest_gen_index: latest_gen_index = gen_index; latest_checkpoint_path = os.path.join(MODEL_SAVE_DIR, filename)
                except ValueError: continue
        if latest_checkpoint_path:
            print(f"  Loading latest checkpoint: {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False) # Use weights_only=False
            num_extra_layers_for_model = checkpoint.get('num_extra_conv_layers', -1) # Get saved layer count
            if num_extra_layers_for_model != -1:
                 print(f"  Determined {num_extra_layers_for_model} extra layers from checkpoint.")
            else:
                 print("  Warning: 'num_extra_conv_layers' key not found in checkpoint.")
        else:
             print("  No checkpoint files found to determine layer count.")
    except Exception as e:
         print(f"  Error loading checkpoint to determine layer count: {e}")

# Check if we successfully determined the layer count
if num_extra_layers_for_model == -1:
    print("\nError: Could not determine the number of extra layers for the model.")
    print("Please ensure the filename contains '_layers_N.pth' or that a valid checkpoint exists.")
    exit()


# --- 5. Load Final Model and Evaluate ---
print(f"\n--- Evaluating Final GrowingCNN Model ---")
print(f"Model path: {FINAL_MODEL_PATH}")
print(f"Architecture: {num_extra_layers_for_model} extra conv layers")

# Instantiate the model with the correct number of layers
final_model = GrowingCNN(num_classes=NUM_CLASSES, num_extra_conv_layers=num_extra_layers_for_model).to(device)

# Load the state dict from the final model file
if not os.path.exists(FINAL_MODEL_PATH):
    print(f"Error: Final model file not found at '{FINAL_MODEL_PATH}'. Cannot evaluate.")
else:
    try:
        # Load the saved state dictionary from the FINAL model file
        # weights_only can usually be True here if it's just the state_dict,
        # but set to False if you saved other things with the final model too.
        final_model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=device))
        print("Final model state loaded successfully.")

        # --- Call Detailed Evaluation ---
        test_targets, test_predictions_default, test_probas, test_filepaths = evaluate_model_detailed(
            final_model, test_loader, device, NUM_CLASSES
        )

        # --- Run Analysis Code (Copied and adapted slightly from ResNet script) ---
        if test_targets is None or test_predictions_default is None or test_probas is None :
             print("Evaluation failed, cannot produce detailed metrics.")
        else:
             # Ensure class_names is available
             if not class_names: class_names = [f"Class {i}" for i in range(NUM_CLASSES)] # Generate default names if needed

             print("\n--- Test Set Performance (Default 0.5 Threshold) ---")
             print(classification_report(test_targets, test_predictions_default, target_names=class_names, zero_division=0))
             test_auc_score = roc_auc_score(test_targets, test_probas) if test_probas is not None and len(np.unique(test_targets)) > 1 else np.nan
             print(f"Test AUC: {test_auc_score:.4f}" if not np.isnan(test_auc_score) else "Test AUC: N/A (requires >1 class)")

             # Confusion Matrix (Default Threshold)
             cm_default = confusion_matrix(test_targets, test_predictions_default)
             plt.figure(figsize=(8, 6)); sns.heatmap(cm_default, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
             plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title('Confusion Matrix (Test Set, Default Threshold)'); plt.show()

             # ROC Curve
             if not np.isnan(test_auc_score):
                 fpr, tpr, _ = roc_curve(test_targets, test_probas)
                 plt.figure(figsize=(8, 6)); plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {test_auc_score:.4f})')
                 plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
                 plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve (Test Set)'); plt.legend(loc="lower right"); plt.grid(True); plt.show()
             else: print("Skipping ROC curve plot (AUC is N/A).")

             # --- Threshold Analysis & Precision-Recall Curve ---
             if len(np.unique(test_targets)) > 1:
                 print("\n--- Threshold Analysis ---")
                 precisions, recalls, thresholds_pr = precision_recall_curve(test_targets, test_probas)
                 pr_auc = auc(recalls, precisions)
                 plt.figure(figsize=(8, 6)); plt.plot(recalls, precisions, marker='.', label=f'PR curve (area = {pr_auc:.4f})')
                 plt.xlabel('Recall (Sensitivity)'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve (Test Set)'); plt.legend(); plt.grid(True); plt.show()

                 if len(precisions) > 1 and len(recalls) > 1:
                     precisions_calc = precisions[:-1]; recalls_calc = recalls[:-1]; thresholds_calc = thresholds_pr
                     f1_scores = np.zeros_like(precisions_calc); valid_f1_indices = (precisions_calc + recalls_calc) > 1e-9
                     f1_scores[valid_f1_indices] = (2*precisions_calc[valid_f1_indices]*recalls_calc[valid_f1_indices]) / (precisions_calc[valid_f1_indices]+recalls_calc[valid_f1_indices])
                     if len(f1_scores) > 0:
                          best_f1_idx = np.argmax(f1_scores)
                          if best_f1_idx < len(thresholds_calc):
                               best_threshold_f1 = thresholds_calc[best_f1_idx]; best_f1 = f1_scores[best_f1_idx]
                               print(f"Threshold maximizing F1-Score: {best_threshold_f1:.4f} (F1={best_f1:.4f})")
                               print(f"  Precision at this threshold: {precisions_calc[best_f1_idx]:.4f}"); print(f"  Recall at this threshold: {recalls_calc[best_f1_idx]:.4f}")
                               test_predictions_best_f1 = (test_probas >= best_threshold_f1).astype(int)
                               print("\n--- Test Set Performance (Best F1 Threshold) ---")
                               print(classification_report(test_targets, test_predictions_best_f1, target_names=class_names, zero_division=0))
                               cm_best_f1 = confusion_matrix(test_targets, test_predictions_best_f1)
                               plt.figure(figsize=(8, 6)); sns.heatmap(cm_best_f1, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names)
                               plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title(f'Confusion Matrix (Test Set, Threshold={best_threshold_f1:.3f})'); plt.show()
                          else: print("Warning: Index for best F1 threshold out of bounds.")
                     else: print("Could not calculate valid F1 scores.")
                 else: print("Not enough points on PR curve to calculate F1.")
             else: print("Skipping PR Curve and F1 threshold analysis (requires >1 class).")

             # --- Visualize Misclassified Images ---
             print("\n--- Analyzing Misclassified Images ---")
             incorrect_indices = np.where(test_targets != test_predictions_default)[0] # Using default threshold predictions
             print(f"Found {len(incorrect_indices)} incorrect predictions (default threshold) out of {len(test_targets)}.")

             # Check if file paths are valid before proceeding
             if test_filepaths and len(test_filepaths) == len(test_targets):
                false_positives = []
                false_negatives = []
                for i in incorrect_indices:
                    true_label = test_targets[i]; pred_label = test_predictions_default[i];
                    filepath = test_filepaths[i]; proba = test_probas[i] if test_probas is not None else -1

                    # Ensure class indices are valid before accessing class_names
                    if 0 <= true_label < len(class_names) and 0 <= pred_label < len(class_names):
                        if true_label == 0 and pred_label == 1: false_positives.append({'path': filepath, 'true': true_label, 'pred': pred_label, 'prob': proba})
                        elif true_label == 1 and pred_label == 0: false_negatives.append({'path': filepath, 'true': true_label, 'pred': pred_label, 'prob': proba})
                    else:
                        print(f"Warning: Invalid label index found at index {i}. True: {true_label}, Pred: {pred_label}. Skipping.")


                print(f"  False Positives: {len(false_positives)}")
                print(f"  False Negatives: {len(false_negatives)}")

                # --- CORRECTED: Use a standard FOR loop for displaying ---
                print("\n--- Sample False Positives ---")
                for i, fp in enumerate(false_positives[:5]): # Loop through the first 5 FPs
                    prob_str = f", Prob(Pos): {fp['prob']:.3f}" if fp['prob'] != -1 else '' # Formatted prob string
                    # Create title using f-string (make sure class_names is valid)
                    title = (f"FP {i+1}: ...{fp['path'][-40:]}\n"
                             f"True: {class_names[fp['true']]}, Pred: {class_names[fp['pred']]}"
                             f"{prob_str}")
                    display_image_from_path(fp['path'], title) # Call the display function

                # --- CORRECTED: Use a standard FOR loop for displaying ---
                print("\n--- Sample False Negatives ---")
                for i, fn in enumerate(false_negatives[:5]): # Loop through the first 5 FNs
                    prob_str = f", Prob(Pos): {fn['prob']:.3f}" if fn['prob'] != -1 else '' # Formatted prob string
                     # Create title using f-string (make sure class_names is valid)
                    title = (f"FN {i+1}: ...{fn['path'][-40:]}\n"
                             f"True: {class_names[fn['true']]}, Pred: {class_names[fn['pred']]}"
                             f"{prob_str}")
                    display_image_from_path(fn['path'], title) # Call the display function
                # --- End of corrected section ---

             else:
                print("File paths unavailable or mismatch length, cannot display misclassified images by path.")

        # --- Plot Feature Maps (ADAPTED NAMES) ---
        print(f"\n--- Plotting Feature Maps from Final GrowingCNN Model ---")
        if test_loader and len(test_loader.dataset) > 0:
             try:
                 sample_data = next(iter(test_loader))
                 sample_image_tensor, sample_label = sample_data
                 image_to_plot = sample_image_tensor[0].unsqueeze(0)
                 print(f"Visualizing feature maps for a sample test image (True Label: {class_names[sample_label[0].item()]}).")
                 # Use names from GrowingCNN
                 plot_feature_maps(final_model, 'conv1', image_to_plot, device, max_maps=16)
                 # Plot first extra conv layer if it exists
                 if num_extra_layers_for_model > 0:
                     plot_feature_maps(final_model, 'extra_conv_0', image_to_plot, device, max_maps=16) # Tries to access extra_conv_layers[0]
                 # Plot last extra conv layer if it exists
                 if num_extra_layers_for_model > 0:
                      plot_feature_maps(final_model, 'extra_conv_last', image_to_plot, device, max_maps=32)
                 # Plot fc1 output (before ReLU) - Requires modifying the plot function slightly or the model
                 # plot_feature_maps(final_model, 'fc1', image_to_plot, device, max_maps=1) # This won't work directly as FC outputs are 1D

             except StopIteration: print("Could not get a sample batch from test loader.")
             except Exception as e: print(f"An error occurred during feature map visualization: {e}")
        else: print("Test loader is empty or not available. Cannot plot feature maps.")


        # --- Plot Augmented Training Image ---
        # This requires access to the original training dataset with *training* transforms
        print("\n--- Plotting a Sample (Original) Training Image with Eval Transforms ---")
        try:
            # Need to reload train dataset with eval transforms to show *what model sees*
            temp_train_dataset_eval = datasets.ImageFolder(TRAIN_DATA_DIR, transform=eval_transform)
            temp_train_loader = DataLoader(temp_train_dataset_eval, batch_size=1, shuffle=True)
            eval_img_tensor, eval_label = next(iter(temp_train_loader))
            # Show the *evaluation* transformed image, not augmented one
            show_augmented_image(eval_img_tensor.squeeze(0), title=f"Sample Eval-Transformed Train Image (Class: {class_names[eval_label.item()]})")
            del temp_train_loader, temp_train_dataset_eval
        except Exception as e: print(f"Could not display sample training image: {e}")

    except RuntimeError as e: print(f"\nError loading final GrowingCNN model state dict: {e}\nCannot perform final steps.");
    except Exception as e: print(f"Unexpected error during final steps: {e}\nCannot perform final steps.");

print("\n--- Evaluation Script Finished ---")

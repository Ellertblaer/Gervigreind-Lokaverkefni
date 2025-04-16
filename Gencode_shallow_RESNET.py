import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from torch.optim import lr_scheduler # For Learning Rate Scheduling

import numpy as np
import copy
import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.metrics import (roc_auc_score, roc_curve, auc, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, classification_report)
from sklearn.utils.class_weight import compute_class_weight
import time
import warnings

# Filter warnings for cleaner output (optional)
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')
warnings.filterwarnings('ignore', category=FutureWarning)


# --- Google Drive Integration ---
from google.colab import drive
import shutil

# --- Configuration ---
NUM_GENERATIONS = 10
MODELS_PER_GENERATION = 5
EPOCHS_PER_GENERATION = 8 # Increased epochs for ResNet fine-tuning
BATCH_SIZE = 16          # Reduced batch size for ResNet memory
LEARNING_RATE = 0.0005   # Adjusted LR for fine-tuning
LR_SCHEDULER_PATIENCE = 2 # Patience for ReduceLROnPlateau (epochs)
LR_SCHEDULER_FACTOR = 0.2 # Factor to reduce LR by

GENERATION_DATA_SEED = 42

# --- Data Paths ---
TRAIN_DATA_DIR = '/content/Training_images'
VAL_DATA_DIR = '/content/Validation_images'
TEST_DATA_DIR = '/content/Test_images'

# --- Model Parameters ---
IMG_HEIGHT, IMG_WIDTH = 224, 224 # Standard ResNet size often works well
NUM_CLASSES = 2
MODEL_NAME = "ResNet50_Cropped" # Specify model for checkpoints/logging

# --- Cropping Parameters ---
CROP_THRESHOLD = 10 # Pixel intensity threshold for background
CROP_MARGIN = 10    # Margin around foreground

# --- Checkpoint Configuration ---
DRIVE_MOUNT_POINT = '/content/drive'
CHECKPOINT_DIR = os.path.join(DRIVE_MOUNT_POINT, 'MyDrive', 'Colab_Checkpoints', f'GenerationalTraining_{MODEL_NAME}')
CHECKPOINT_FILENAME = f'generational_checkpoint_{MODEL_NAME}.pth'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)

# --- Setup Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Mount Google Drive ---
print("Mounting Google Drive...")
try:
    drive.mount(DRIVE_MOUNT_POINT, force_remount=True)
    print("Google Drive mounted successfully.")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Checkpoint directory ensured: {CHECKPOINT_DIR}")
except Exception as e:
    print(f"Error mounting Google Drive: {e}")
    print("Checkpoints will not be saved/loaded from Drive.")
    CHECKPOINT_PATH = f'./generational_checkpoint_{MODEL_NAME}.pth'

# --- 1. Custom Preprocessing Transform ---
class CropBackground:
    """Crops black background from a PIL Image."""
    def __init__(self, threshold=10, margin=5):
        self.threshold = threshold
        self.margin = margin

    def __call__(self, img):
        if not isinstance(img, Image.Image): raise TypeError("Input must be a PIL Image.")
        try:
            img_gray = ImageOps.grayscale(img)
            img_array = np.array(img_gray)
            coords = np.argwhere(img_array > self.threshold)
            if coords.size == 0: return img # Return original if empty
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            x0 = max(0, x0 - self.margin)
            y0 = max(0, y0 - self.margin)
            x1 = min(img.width - 1, x1 + self.margin)
            y1 = min(img.height - 1, y1 + self.margin)
            if y1 <= y0 or x1 <= x0: return img # Return original if box invalid
            cropped_img = img.crop((x0, y0, x1 + 1, y1 + 1))
            return cropped_img
        except Exception as e:
            print(f"Error during cropping: {e}. Returning original image.")
            return img

    def __repr__(self):
        return self.__class__.__name__ + f'(threshold={self.threshold}, margin={self.margin})'

# --- 2. Data Loading and Preparation ---
crop_transform = CropBackground(threshold=CROP_THRESHOLD, margin=CROP_MARGIN)

# More aggressive augmentation for training
train_transform = transforms.Compose([
    crop_transform, # Crop first
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=25), # Increased rotation
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2), shear=15), # Increased affine
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1), # Increased jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
])

# Validation/Test: Crop, Resize, Normalize only
val_test_transform = transforms.Compose([
    crop_transform, # Crop first
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = []
try:
    # Create datasets directly from folders
    train_dataset_orig = datasets.ImageFolder(TRAIN_DATA_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DATA_DIR, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(TEST_DATA_DIR, transform=val_test_transform)

    class_names = train_dataset_orig.classes
    print(f"Detected classes: {class_names}")
    if len(class_names) != NUM_CLASSES:
         print(f"Warning: Mismatch between detected classes ({len(class_names)}) and NUM_CLASSES ({NUM_CLASSES}).")
         # Optionally adjust NUM_CLASSES = len(class_names)

    # Check class consistency
    if val_dataset.classes != class_names or test_dataset.classes != class_names:
        print("Warning: Class mismatch between training, validation, or test sets!")

except FileNotFoundError as e: print(f"Error: Data directory not found - {e}."); raise
except Exception as e: print(f"Data loading error: {e}"); raise

# DataLoaders (Val/Test)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Data loaded:")
print(f"  Training samples:   {len(train_dataset_orig)} (from {TRAIN_DATA_DIR})")
print(f"  Validation samples: {len(val_dataset)} (from {VAL_DATA_DIR})")
print(f"  Test samples:       {len(test_dataset)} (from {TEST_DATA_DIR})")

# Function to create train loader (used inside generational loop)
def create_generation_train_loader(dataset, batch_size, generation_seed=None):
    g = None
    if generation_seed is not None: g = torch.Generator(); g.manual_seed(generation_seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g, num_workers=2, pin_memory=True)


# --- 3. Model Definition (ResNet) ---
def create_resnet_model(num_classes=NUM_CLASSES, pretrained=True):
    """Loads a pre-trained ResNet50 model and replaces the final classifier."""
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# --- 4. Helper Functions (Plotting, etc.) ---

def show_augmented_image(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], title="Sample Augmented Training Image"):
    """Displays a single augmented image tensor after reversing normalization."""
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

def plot_feature_maps(model, layer_name, image_tensor, device, max_maps=16):
    """Plots feature maps from a specific layer."""
    model.eval(); feature_maps = None; hook_handle = None
    def hook_fn(module, input, output): nonlocal feature_maps; feature_maps = output.detach().clone()
    target_layer = None; found = False
    for name, layer in model.named_modules():
        if name == layer_name: target_layer = layer; found = True; break
    if not found: print(f"Error: Layer '{layer_name}' not found."); return
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


# --- 5. Training & Evaluation Functions ---
def train_one_epoch(model, train_loader, optimizer, criterion, device, num_classes):
    """Trains the model for one epoch."""
    model.train(); running_loss = 0.0; epoch_correct = 0; epoch_total = 0
    all_targets = []; all_preds_proba = []
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(); outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward(); optimizer.step(); running_loss += loss.item() * data.size(0)
        pred_prob = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(pred_prob.data, 1)
        epoch_total += target.size(0); epoch_correct += (predicted == target).sum().item()
        if num_classes == 2:
            all_targets.extend(target.cpu().numpy())
            proba_for_auc = pred_prob[:, 1]
            all_preds_proba.extend(proba_for_auc.detach().cpu().numpy())
    avg_epoch_loss = running_loss / epoch_total if epoch_total > 0 else 0
    avg_epoch_acc = 100.0 * epoch_correct / epoch_total if epoch_total > 0 else 0
    epoch_auc = np.nan
    if num_classes == 2 and len(all_preds_proba) > 0 and len(np.unique(all_targets)) > 1:
        try: epoch_auc = roc_auc_score(all_targets, all_preds_proba)
        except ValueError: pass
    return avg_epoch_loss, avg_epoch_acc, epoch_auc

def evaluate_model(model, data_loader, criterion, device, num_classes):
    """Evaluates model, returns loss, accuracy, AUC."""
    model.eval(); val_loss = 0.0; correct = 0; total = 0
    all_targets = []; all_preds_proba = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device); outputs = model(data)
            loss = criterion(outputs, target)
            val_loss += loss.item() * data.size(0)
            pred_prob = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(pred_prob.data, 1)
            total += target.size(0); correct += (predicted == target).sum().item()
            if num_classes == 2:
                all_targets.extend(target.cpu().numpy())
                all_preds_proba.extend(pred_prob[:, 1].cpu().numpy())
    avg_loss = val_loss / total if total > 0 else 0
    accuracy = 100.0 * correct / total if total > 0 else 0
    val_auc = np.nan
    if num_classes == 2 and len(all_preds_proba) > 0 and len(np.unique(all_targets)) > 1:
        try: val_auc = roc_auc_score(all_targets, all_preds_proba)
        except ValueError: pass
    return avg_loss, accuracy, val_auc

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
        # Assumes data_loader.dataset is the original ImageFolder dataset
        if isinstance(data_loader.dataset, datasets.ImageFolder):
             filepaths_in_order = [item[0] for item in data_loader.dataset.samples]
        else:
             print("Warning: Cannot get file paths, dataset is not an ImageFolder.")
    except Exception as e:
        print(f"Warning: Error getting file paths: {e}")

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target_np = target.cpu().numpy() # Keep target on CPU initially
            outputs = model(data)
            pred_prob = torch.softmax(outputs, dim=1)
            proba_pos_class = pred_prob[:, 1].cpu().numpy() if num_classes == 2 else None
            _, predicted = torch.max(pred_prob.data, 1)
            predicted_np = predicted.cpu().numpy()

            all_targets.extend(target_np)
            all_predictions.extend(predicted_np)
            if proba_pos_class is not None: all_probas.extend(proba_pos_class)

            if filepaths_in_order is not None:
                batch_size = data.size(0)
                batch_indices = list(range(current_idx, current_idx + batch_size))
                all_filepaths.extend([filepaths_in_order[i] for i in batch_indices])
                current_idx += batch_size

    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    all_probas = np.array(all_probas) if num_classes == 2 else None

    return all_targets, all_predictions, all_probas, all_filepaths


# --- 6. Calculate Class Weights for Loss Function ---
print("Calculating class weights for training loss...")
try:
    # Ensure train_dataset_orig.targets is accessible
    if hasattr(train_dataset_orig, 'targets'):
        train_labels = train_dataset_orig.targets
    elif hasattr(train_dataset_orig, 'labels'): # Some datasets use 'labels'
         train_labels = train_dataset_orig.labels
    else:
        # Fallback: iterate through dataset (can be slow)
        print("Getting training labels by iterating (may be slow)...")
        train_labels = [label for _, label in train_dataset_orig]

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Using Class Weights: {class_weights_tensor.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
except Exception as e:
    print(f"Error calculating class weights: {e}. Using standard CrossEntropyLoss.")
    criterion = nn.CrossEntropyLoss()


# --- 7. Load Checkpoint / Initialize State ---
start_generation = 0
best_model_state_dict = None
all_best_val_metrics = [] # Store tuples: (gen, acc, auc)

print(f"\n--- Checking for {MODEL_NAME} checkpoint at: {CHECKPOINT_PATH} ---")
if os.path.exists(CHECKPOINT_PATH):
    try:
        print("Checkpoint file found. Loading...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        saved_config = checkpoint.get('config', {})
        saved_model_class = saved_config.get('model_class', 'Unknown')

        if saved_model_class != MODEL_NAME:
            raise ValueError(f"Checkpoint model mismatch: expected {MODEL_NAME}, found {saved_model_class}")

        start_generation = checkpoint['generation'] + 1
        best_model_state_dict = checkpoint['model_state_dict']
        all_best_val_metrics = checkpoint.get('all_best_val_metrics', [])

        # Config check (optional)
        # ...

        if start_generation >= NUM_GENERATIONS: print("Target generations reached."); exit()
        else: print(f"Resuming {MODEL_NAME} training from Gen {start_generation + 1}/{NUM_GENERATIONS}")

    except Exception as e:
        print(f"Error loading {MODEL_NAME} checkpoint or compatibility issue: {e}. Starting fresh.")
        start_generation = 0; best_model_state_dict = None; all_best_val_metrics = []
else:
    print(f"No {MODEL_NAME} checkpoint found. Starting fresh.")
    start_generation = 0; best_model_state_dict = None; all_best_val_metrics = []


# --- 8. Generational Training Loop ---
print(f"\n--- Starting {MODEL_NAME} Generational Training (Gens {start_generation + 1} to {NUM_GENERATIONS}) ---")

for generation in range(start_generation, NUM_GENERATIONS):
    current_gen_number = generation + 1
    print(f"\n--- Generation {current_gen_number}/{NUM_GENERATIONS} ---")

    # 1. Determine Initial State
    if best_model_state_dict is None:
        print(f"  Initializing {MODEL_NAME} model (Pretrained=True) for first generation.")
        try:
            initial_model = create_resnet_model(num_classes=NUM_CLASSES, pretrained=True).to(device)
            initial_state_dict = copy.deepcopy(initial_model.state_dict()); del initial_model
        except Exception as e: print(f"Error creating initial {MODEL_NAME} model: {e}"); raise
    else:
        initial_state_dict = best_model_state_dict

    # 2. Prepare Training Data Loader for this generation
    current_gen_seed = GENERATION_DATA_SEED + generation if GENERATION_DATA_SEED is not None else None
    train_loader_gen = create_generation_train_loader(train_dataset_orig, BATCH_SIZE, generation_seed=current_gen_seed)

    generation_model_states = []
    generation_val_accuracies = []
    generation_val_aucs = []
    generation_val_losses = [] # Track val loss for scheduler

    # 3. Train and Evaluate Models within the Generation
    print(f"  Training {MODELS_PER_GENERATION} {MODEL_NAME} models...")
    best_val_auc_this_gen = -1.0 # Track best AUC within this generation for selection

    for model_run in range(MODELS_PER_GENERATION):
        print(f"    Model Run {model_run + 1}/{MODELS_PER_GENERATION}")
        current_model = create_resnet_model(num_classes=NUM_CLASSES, pretrained=True).to(device)
        try:
            current_model.load_state_dict(copy.deepcopy(initial_state_dict))
        except RuntimeError as e:
             print(f"    Error loading state dict into model {model_run+1}: {e}"); generation_model_states = []; break # Skip gen
        except Exception as e: print(f"    Unexpected error loading state dict: {e}"); continue # Skip run

        optimizer = optim.Adam(current_model.parameters(), lr=LEARNING_RATE)
        # Define LR scheduler for this model instance
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=LR_SCHEDULER_FACTOR,
                                                   patience=LR_SCHEDULER_PATIENCE, verbose=True)

        run_val_auc = -1 # Track best val auc for *this specific run*

        # Epoch loop for this model run
        for epoch in range(EPOCHS_PER_GENERATION):
            epoch_start_time = time.time()
            train_loss, train_acc, train_auc = train_one_epoch(
                current_model, train_loader_gen, optimizer, criterion, device, NUM_CLASSES
            )
            # Evaluate on validation set after each epoch for LR scheduling
            val_loss, val_accuracy, val_auc = evaluate_model(
                current_model, val_loader, criterion, device, NUM_CLASSES
            )
            epoch_end_time = time.time()

            print(f"      Epoch {epoch+1}/{EPOCHS_PER_GENERATION} [{epoch_end_time - epoch_start_time:.1f}s] - "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, AUC: {train_auc if not np.isnan(train_auc) else 'N/A':.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.2f}%, AUC: {val_auc if not np.isnan(val_auc) else 'N/A':.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.1e}")

            # LR Scheduler Step (using validation AUC)
            current_epoch_val_auc = val_auc if not np.isnan(val_auc) else 0.0 # Use 0 if NaN
            scheduler.step(current_epoch_val_auc)

            # Track the best validation AUC achieved during this run's epochs
            if current_epoch_val_auc > run_val_auc:
                 run_val_auc = current_epoch_val_auc


        # After all epochs for this run, store final state and metrics
        # Evaluate one last time to get final metrics for this run
        final_val_loss, final_val_accuracy, final_val_auc = evaluate_model(
                 current_model, val_loader, criterion, device, NUM_CLASSES
             )

        generation_model_states.append(copy.deepcopy(current_model.state_dict()))
        generation_val_losses.append(final_val_loss)
        generation_val_accuracies.append(final_val_accuracy)
        generation_val_aucs.append(final_val_auc) # Store the final AUC

        del current_model, optimizer, scheduler
        if device.type == 'cuda': torch.cuda.empty_cache()

    if not generation_model_states: print(f"Warning: Gen {current_gen_number} skipped due to model loading error."); continue

    # 4. Select Best Model based on HIGHEST VALIDATION AUC from this generation
    valid_auc_indices = [i for i, auc_val in enumerate(generation_val_aucs) if not np.isnan(auc_val)]
    if not valid_auc_indices:
        print("Warning: No valid AUC scores in this generation. Cannot select best model. Skipping checkpoint.")
        continue # Keep the previous best_model_state_dict

    best_model_index_this_gen = valid_auc_indices[np.argmax([generation_val_aucs[i] for i in valid_auc_indices])]
    best_accuracy_this_gen = generation_val_accuracies[best_model_index_this_gen]
    best_auc_this_gen = generation_val_aucs[best_model_index_this_gen]
    best_model_state_dict = generation_model_states[best_model_index_this_gen] # Prepare for next gen

    # Store metrics history for the best model of each generation
    all_best_val_metrics.append({'gen': current_gen_number, 'acc': best_accuracy_this_gen, 'auc': best_auc_this_gen})

    print(f"--- Best Gen {current_gen_number} ({MODEL_NAME}): Model Run {best_model_index_this_gen + 1} selected ---")
    print(f"    Val Acc: {best_accuracy_this_gen:.2f}%, Val AUC: {best_auc_this_gen:.4f}")


    # 5. Save Checkpoint
    checkpoint_data = {
        'generation': generation, 'model_state_dict': best_model_state_dict,
        'all_best_val_metrics': all_best_val_metrics,
        'config': {'model_class': MODEL_NAME, 'num_classes': NUM_CLASSES, 'img_height': IMG_HEIGHT, 'img_width': IMG_WIDTH, 'learning_rate': LEARNING_RATE}
    }
    try: torch.save(checkpoint_data, CHECKPOINT_PATH)
    except Exception as e: print(f"  Error saving checkpoint: {e}")


# --- 9. Final Steps after ALL Generations Complete ---
print(f"\n--- {MODEL_NAME} Generational Training Complete ---")

if best_model_state_dict is not None:
    final_model = create_resnet_model(num_classes=NUM_CLASSES, pretrained=False).to(device)
    try:
        final_model.load_state_dict(best_model_state_dict)
        print(f"Final {MODEL_NAME} model loaded from the best state found across generations.")

        print("\n--- Generational Validation Performance Summary ---")
        if all_best_val_metrics:
             for metric_entry in all_best_val_metrics:
                 print(f"  Gen {metric_entry['gen']}: Best Val Acc: {metric_entry['acc']:.2f}%, Best Val AUC: {metric_entry['auc']:.4f}")
        else:
             print("  No validation metric history saved.")


        # --- Evaluate final model on Test Set for Detailed Analysis ---
        print(f"\n--- Evaluating Final {MODEL_NAME} on TEST Set ---")
        test_targets, test_predictions_default, test_probas, test_filepaths = evaluate_model_detailed(
            final_model, test_loader, device, NUM_CLASSES
        )

        if test_targets is None or test_predictions_default is None or test_probas is None :
             print("Evaluation failed, cannot produce detailed metrics.")
        else:
             print("\n--- Test Set Performance (Default 0.5 Threshold) ---")
             print(classification_report(test_targets, test_predictions_default, target_names=class_names))
             test_auc_score = roc_auc_score(test_targets, test_probas) if test_probas is not None else np.nan
             print(f"Test AUC: {test_auc_score:.4f}" if not np.isnan(test_auc_score) else "Test AUC: N/A")

             # Confusion Matrix (Default Threshold)
             cm_default = confusion_matrix(test_targets, test_predictions_default)
             plt.figure(figsize=(8, 6)); sns.heatmap(cm_default, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
             plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title('Confusion Matrix (Test Set, Default Threshold)'); plt.show()

             # ROC Curve
             fpr, tpr, _ = roc_curve(test_targets, test_probas)
             plt.figure(figsize=(8, 6)); plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {test_auc_score:.4f})')
             plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
             plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve (Test Set)'); plt.legend(loc="lower right"); plt.grid(True); plt.show()

             # --- Threshold Analysis & Precision-Recall Curve ---
             print("\n--- Threshold Analysis ---")
             precisions, recalls, thresholds_pr = precision_recall_curve(test_targets, test_probas)
             # Exclude the last precision/recall value corresponding to threshold=1
             precisions = precisions[:-1]
             recalls = recalls[:-1]

             # Plot PR curve
             pr_auc = auc(recalls, precisions)
             plt.figure(figsize=(8, 6))
             plt.plot(recalls, precisions, marker='.', label=f'PR curve (area = {pr_auc:.4f})')
             plt.xlabel('Recall (Sensitivity)'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve (Test Set)'); plt.legend(); plt.grid(True); plt.show()

             # Find threshold that maximizes F1 score
             f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
             best_f1_idx = np.argmax(f1_scores)
             best_threshold_f1 = thresholds_pr[best_f1_idx]
             best_f1 = f1_scores[best_f1_idx]
             print(f"Threshold maximizing F1-Score: {best_threshold_f1:.4f} (F1={best_f1:.4f})")
             print(f"  Precision at this threshold: {precisions[best_f1_idx]:.4f}")
             print(f"  Recall at this threshold: {recalls[best_f1_idx]:.4f}")

             # Apply the 'best F1' threshold
             test_predictions_best_f1 = (test_probas >= best_threshold_f1).astype(int)
             print("\n--- Test Set Performance (Best F1 Threshold) ---")
             print(classification_report(test_targets, test_predictions_best_f1, target_names=class_names))
             cm_best_f1 = confusion_matrix(test_targets, test_predictions_best_f1)
             plt.figure(figsize=(8, 6)); sns.heatmap(cm_best_f1, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names)
             plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title(f'Confusion Matrix (Test Set, Threshold={best_threshold_f1:.3f})'); plt.show()

             # --- Visualize Misclassified Images ---
             print("\n--- Analyzing Misclassified Images ---")
             incorrect_indices = np.where(test_targets != test_predictions_default)[0] # Using default threshold predictions
             print(f"Found {len(incorrect_indices)} incorrect predictions (default threshold) out of {len(test_targets)}.")

             false_positives = []
             false_negatives = []

             if test_filepaths and len(test_filepaths) == len(test_targets):
                for i in incorrect_indices:
                    true_label = test_targets[i]
                    pred_label = test_predictions_default[i]
                    filepath = test_filepaths[i]
                    proba = test_probas[i] if test_probas is not None else -1

                    if true_label == 0 and pred_label == 1: false_positives.append({'path': filepath, 'true': true_label, 'pred': pred_label, 'prob': proba})
                    elif true_label == 1 and pred_label == 0: false_negatives.append({'path': filepath, 'true': true_label, 'pred': pred_label, 'prob': proba})

                print(f"  False Positives: {len(false_positives)}")
                print(f"  False Negatives: {len(false_negatives)}")

                print("\n--- Sample False Positives (Predicted CAD, Actual No CAD) ---")
                for i, fp in enumerate(false_positives[:5]):
                    prob_str = f", Prob(CAD): {fp['prob']:.3f}" if fp['prob'] != -1 else ""
                    title = (f"FP {i+1}: ...{fp['path'][-40:]}\n"
                             f"True: {class_names[fp['true']]}, Pred: {class_names[fp['pred']]}"
                             f"{prob_str}")
                    display_image_from_path(fp['path'], title)

                print("\n--- Sample False Negatives (Predicted No CAD, Actual CAD) ---")
                for i, fn in enumerate(false_negatives[:5]):
                    prob_str = f", Prob(CAD): {fn['prob']:.3f}" if fn['prob'] != -1 else ""
                    title = (f"FN {i+1}: ...{fn['path'][-40:]}\n"
                             f"True: {class_names[fn['true']]}, Pred: {class_names[fn['pred']]}"
                             f"{prob_str}")
                    display_image_from_path(fn['path'], title)
             else:
                print("File paths unavailable or mismatch length, cannot display misclassified images by path.")


        # --- Plot Feature Maps ---
        print(f"\n--- Plotting Feature Maps from Final {MODEL_NAME} Model ---")
        if test_loader and len(test_loader.dataset) > 0:
             try:
                 sample_data = next(iter(test_loader))
                 sample_image_tensor, sample_label = sample_data
                 image_to_plot = sample_image_tensor[0].unsqueeze(0)
                 print(f"Visualizing feature maps for a sample test image (True Label: {class_names[sample_label[0].item()]}).")
                 plot_feature_maps(final_model, 'conv1', image_to_plot, device, max_maps=16)
                 plot_feature_maps(final_model, 'layer1', image_to_plot, device, max_maps=16)
                 plot_feature_maps(final_model, 'layer2', image_to_plot, device, max_maps=32)
                 plot_feature_maps(final_model, 'layer4', image_to_plot, device, max_maps=32)
             except StopIteration: print("Could not get a sample batch from test loader.")
             except Exception as e: print(f"An error occurred during feature map visualization: {e}")
        else: print("Test loader is empty or not available. Cannot plot feature maps.")


        # --- Plot Augmented Training Image ---
        print("\n--- Plotting a Sample Augmented Training Image ---")
        try:
            temp_train_loader = DataLoader(train_dataset_orig, batch_size=1, shuffle=True)
            aug_image_tensor, aug_label = next(iter(temp_train_loader))
            show_augmented_image(aug_image_tensor.squeeze(0), title=f"Sample Augmented Image (Class: {class_names[aug_label.item()]})")
            del temp_train_loader
        except Exception as e: print(f"Could not display augmented image: {e}")


        # --- Optional: Save Final Model ---
        final_model_save_path = os.path.join(CHECKPOINT_DIR, f'final_generational_model_{MODEL_NAME}_complete.pth')
        print(f"\nSaving final trained {MODEL_NAME} model to: {final_model_save_path}")
        try: torch.save(final_model.state_dict(), final_model_save_path); print("Final model saved successfully.")
        except Exception as e: print(f"Error saving final model: {e}")

    except RuntimeError as e: print(f"\nError loading final {MODEL_NAME} model state dict: {e}\nCannot perform final steps.");
    except Exception as e: print(f"Unexpected error during final steps: {e}\nCannot perform final steps.");

else:
    print("No valid model state available after training loop.")

print("\n--- Script Finished ---")



#Grafík byrjar

from sklearn.metrics import precision_recall_curve, precision_score, recall_score

# Calculate precision, recall for various thresholds
precisions, recalls, thresholds = precision_recall_curve(test_targets, test_probas)

# Plot PR curve
plt.figure()
plt.plot(recalls, precisions, marker='.')
plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()

# Find threshold that maximizes F1 (example)
f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9) # Add epsilon for stability
best_threshold_idx = np.argmax(f1_scores[:-1]) # Exclude last threshold value
best_threshold = thresholds[best_threshold_idx]
best_f1 = f1_scores[best_threshold_idx]
print(f"Best Threshold (for F1): {best_threshold:.4f} with F1: {best_f1:.4f}")

# Apply the new threshold to get new predictions
new_predictions = (test_probas >= best_threshold).astype(int)
new_cm = confusion_matrix(test_targets, new_predictions)
new_precision = precision_score(test_targets, new_predictions)
new_recall = recall_score(test_targets, new_predictions)

print("\nMetrics with new threshold:")
print(f" Threshold: {best_threshold:.4f}")
print(f" Confusion Matrix:\n{new_cm}")
print(f" Precision: {new_precision:.4f}")
print(f" Recall (Sensitivity): {new_recall:.4f}")
# Plot new CM etc.



def evaluate_model_detailed(model, data_loader, criterion, device, num_classes): # Renamed for clarity
    model.eval()
    all_targets = []
    all_predictions = []
    all_probas = []
    all_filepaths = [] # <<< Store file paths

    # Get file paths from the dataset associated with the loader
    # Ensure the loader uses the original dataset, not a subset, or adapt this part
    try:
        # Assumes data_loader.dataset is the original ImageFolder dataset
        filepaths_in_order = [item[0] for item in data_loader.dataset.samples]
        # Note: This assumes shuffle=False in the DataLoader, which is correct for test/val
        current_idx = 0
    except AttributeError:
        print("Warning: Could not directly get file paths from data_loader.dataset.samples.")
        filepaths_in_order = None


    with torch.no_grad():
        batch_num = 0
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            # loss = criterion(outputs, target) # Loss calculation not needed for just getting preds

            pred_prob = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(pred_prob.data, 1)

            # Store results
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            if num_classes == 2:
                 all_probas.extend(pred_prob[:, 1].cpu().numpy()) # Positive class proba

            # <<< Store corresponding file paths >>>
            if filepaths_in_order is not None:
                batch_size = data.size(0)
                batch_indices = list(range(current_idx, current_idx + batch_size))
                all_filepaths.extend([filepaths_in_order[i] for i in batch_indices])
                current_idx += batch_size
            # If filepaths couldn't be retrieved, all_filepaths will remain empty

            batch_num += 1

    # Convert to numpy arrays
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    all_probas = np.array(all_probas) if num_classes == 2 else None

    # Calculate overall metrics (optional here, can be done outside)
    # accuracy = accuracy_score(all_targets, all_predictions)
    # cm = confusion_matrix(all_targets, all_predictions)
    # auc_score_val = roc_auc_score(all_targets, all_probas) if all_probas is not None else np.nan

    # Return detailed results
    return all_targets, all_predictions, all_probas, all_filepaths # Return paths

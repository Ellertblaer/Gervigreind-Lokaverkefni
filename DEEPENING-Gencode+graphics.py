#Boiler-plate skipanir sem eru nauðsynlegar til að importa gagnasettinu og fá það til að virka.

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.io import loadmat
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

#Importa gagnasettinu
!wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/fk6rys63h9-1.zip

! unzip fk6rys63h9-1.zip

! unzip Test_images.zip
! unzip Training_images.zip
! unzip Validation_images.zip


# Iterate-ar yfir allt gagnasettið, ekki bara nýja. Hitt kallast layer freezing eða transfer learning fine-tuning og ég er ekki að nota það.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset, Dataset
from torchvision import datasets, transforms
import numpy as np
import copy
import os
import re
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns # For confusion matrix plotting
import time
import math # For feature map grid calculation

# --- Google Drive Integration ---
from google.colab import drive
import shutil

# --- Configuration ---
NUM_GENERATIONS = 10  # Total desired generations
MODELS_PER_GENERATION = 5 # Models with the *same* architecture trained within a generation
LAYERS_TO_ADD_PER_GENERATION = 2 # How many Conv->ReLU layers to add each time
EPOCHS_PER_GENERATION = 5 # Epochs to train the *current* architecture each generation
FINAL_TRAINING_EPOCHS = 10 # <<<--- NEW: Epochs for final history training/plotting
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GENERATION_DATA_SEED = 42
VAL_SPLIT = 0.1
# --- New Configuration for Plotting ---
CONFUSION_MATRIX_THRESHOLD = 0.975 # Example custom threshold
FEATURE_MAP_LAYER_NAME = 'conv1' # Layer to visualize feature maps from (e.g., 'conv1', 'relu1', 'extra_conv_layers.0')
FEATURE_MAP_SAMPLE_INDEX = 0 # Index of the image in the test set to use for feature maps

# --- Data Paths ---
TRAIN_DATA_DIR = '/content/Training_images'
VAL_DATA_DIR = '/content/Validation_images'

# --- Model Parameters ---
IMG_HEIGHT, IMG_WIDTH = 299, 299 # Keep consistent for input
NUM_CLASSES = 2
BASE_CONV_OUT_CHANNELS = 16 # Channels after first conv layer
EXTRA_CONV_CHANNELS = 32 # Number of channels for the added conv layers

# --- Checkpoint Configuration ---
DRIVE_MOUNT_POINT = '/content/drive'
CHECKPOINT_DIR = os.path.join(DRIVE_MOUNT_POINT, 'MyDrive', 'Gervigreind - verkefni', 'LayerGrowth') # Specific subdir
CHECKPOINT_BASENAME = 'layer_growth_checkpoint_gen_'
CHECKPOINT_SUFFIX = '.pth'

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
    print(f"Error mounting Google Drive: {e}. Exiting.")
    exit()

# --- 1. Data Loading and Preparation (Identical) ---
train_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)), transforms.RandomRotation(20),
    transforms.RandomResizedCrop((IMG_HEIGHT, IMG_WIDTH), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
val_test_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
try:
    full_train_dataset_train_tf = datasets.ImageFolder(TRAIN_DATA_DIR, transform=train_transform)
    full_train_dataset_val_tf = datasets.ImageFolder(TRAIN_DATA_DIR, transform=val_test_transform)
    test_dataset_folder = datasets.ImageFolder(VAL_DATA_DIR, transform=val_test_transform)
    class_names = full_train_dataset_train_tf.classes
    print(f"Class Names: {class_names}")
    if len(class_names) != NUM_CLASSES: print(f"Warning: Class mismatch")
except Exception as e: print(f"Data loading error: {e}"); raise
num_train_val = len(full_train_dataset_train_tf); indices = list(range(num_train_val))
split = int(np.floor(VAL_SPLIT * num_train_val)); np.random.seed(42); np.random.shuffle(indices)
train_idx, val_idx = indices[split:], indices[:split]
# Keep original datasets for final training
train_dataset_subset = Subset(full_train_dataset_train_tf, train_idx) # Use subset for generational training
val_dataset_subset = Subset(full_train_dataset_val_tf, val_idx)
val_loader = DataLoader(val_dataset_subset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset_folder, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print(f"Data loaded: {len(train_dataset_subset)} train, {len(val_dataset_subset)} validation, {len(test_dataset_folder)} test samples.")

def create_generation_train_loader(dataset, batch_size, generation_seed=None):
    g = None
    if generation_seed is not None: g = torch.Generator(); g.manual_seed(generation_seed)
    print(f"  Shuffling training data for generation with seed: {generation_seed if generation_seed else 'Random'}")
    # Use the subset for generational training
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g, num_workers=2, pin_memory=True)

# --- 2. Model Definition (GrowingCNN - Identical) ---
class GrowingCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, num_extra_conv_layers=0):
        super(GrowingCNN, self).__init__()
        self.num_extra_conv_layers = num_extra_conv_layers
        # print(f"  Instantiating GrowingCNN with {num_extra_conv_layers} extra conv layers.") # Less verbose

        # --- Base Convolutional Block ---
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=BASE_CONV_OUT_CHANNELS, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Extra Convolutional Layers ---
        self.extra_conv_layers = nn.ModuleList()
        current_channels = BASE_CONV_OUT_CHANNELS
        for i in range(num_extra_conv_layers):
            conv_layer = nn.Conv2d(in_channels=current_channels, out_channels=EXTRA_CONV_CHANNELS,
                                   kernel_size=3, stride=1, padding=1)
            self.extra_conv_layers.append(conv_layer)
            self.extra_conv_layers.append(nn.ReLU())
            current_channels = EXTRA_CONV_CHANNELS

        # --- Second Pooling Layer ---
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Dynamically Calculate Flattened Size ---
        if not list(self.parameters()): self._dummy_param = nn.Parameter(torch.empty(0))
        self.flattened_size = self._get_conv_output_size((3, IMG_HEIGHT, IMG_WIDTH))
        if self.flattened_size <= 0: raise ValueError(f"Calculated flattened size is {self.flattened_size}.")
        # print(f"    Calculated flattened size: {self.flattened_size}") # Less verbose

        # --- Fully Connected Layers ---
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def _get_conv_output_size(self, input_shape):
        current_device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        dummy_input = torch.randn(1, *input_shape).to(current_device)
        try:
            with torch.no_grad():
                x = self.pool1(self.relu1(self.conv1(dummy_input)))
                for layer in self.extra_conv_layers: x = layer(x)
                x = self.pool2(x)
                output_size = int(np.prod(x.shape[1:]))
            return output_size
        except Exception as e:
             print(f"Error during dummy forward pass for size calculation: {e}")
             h = IMG_HEIGHT // 2 // 2; w = IMG_WIDTH // 2 // 2
             final_channels = EXTRA_CONV_CHANNELS if self.num_extra_conv_layers > 0 else BASE_CONV_OUT_CHANNELS
             estimated_size = final_channels * h * w
             print(f"Warning: Using estimated flattened size: {estimated_size}")
             return estimated_size

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        for layer in self.extra_conv_layers: x = layer(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu_fc1(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 3. Training & Evaluation Functions (Identical to previous version) ---
def train_one_epoch(model, train_loader, optimizer, criterion, device, num_classes):
    model.train(); running_loss = 0.0; epoch_correct = 0; epoch_total = 0
    all_targets = []; all_preds_proba = []
    for data, target in train_loader:
        data, target = data.to(device), target.to(device); optimizer.zero_grad(); outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward(); optimizer.step()
        running_loss += loss.item() * data.size(0); epoch_total += target.size(0)
        pred_prob = torch.softmax(outputs, dim=1); _, predicted = torch.max(pred_prob.data, 1)
        epoch_correct += (predicted == target).sum().item()
        all_targets.extend(target.cpu().numpy())
        if num_classes == 2: all_preds_proba.extend(pred_prob[:, 1].detach().cpu().numpy())
    avg_epoch_loss = running_loss / epoch_total if epoch_total > 0 else 0
    avg_epoch_acc = 100.0 * epoch_correct / epoch_total if epoch_total > 0 else 0; epoch_auc = np.nan
    if num_classes == 2 and len(all_preds_proba) > 0:
        try:
            if len(np.unique(all_targets)) > 1: epoch_auc = roc_auc_score(all_targets, all_preds_proba)
        except Exception: epoch_auc = np.nan # Simplified error handling
    return avg_epoch_loss, avg_epoch_acc, epoch_auc

def evaluate_model(model, data_loader, criterion, device, num_classes):
    model.eval(); val_loss = 0.0; correct = 0; total = 0
    all_targets = []; all_preds_proba_positive = []; all_preds_label = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device); outputs = model(data)
            loss = criterion(outputs, target); val_loss += loss.item() * data.size(0); total += target.size(0)
            pred_prob = torch.softmax(outputs, dim=1); _, predicted = torch.max(pred_prob.data, 1)
            correct += (predicted == target).sum().item(); all_targets.extend(target.cpu().numpy())
            all_preds_label.extend(predicted.cpu().numpy())
            if num_classes == 2: all_preds_proba_positive.extend(pred_prob[:, 1].cpu().numpy())
    avg_loss = val_loss / total if total > 0 else 0; accuracy = 100.0 * correct / total if total > 0 else 0; val_auc = np.nan
    if num_classes == 2 and len(all_preds_proba_positive) > 0:
        try:
            if len(np.unique(all_targets)) > 1: val_auc = roc_auc_score(all_targets, all_preds_proba_positive)
        except Exception: val_auc = np.nan # Simplified error handling
    return avg_loss, accuracy, val_auc, all_targets, all_preds_proba_positive, all_preds_label


# --- 4. Plotting Functions ---

# <<<--- NEW: History Plotting Function --- >>>
def plot_history(history, epochs):
    epochs_range = range(1, epochs + 1) # Start epochs from 1
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    # Accuracy Plot
    axs[0].plot(epochs_range, history['train_acc'], 'o-', label='Training Accuracy')
    axs[0].plot(epochs_range, history['val_acc'], 'o-', label='Validation Accuracy')
    axs[0].set_title('Model Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy (%)')
    axs[0].legend()
    axs[0].grid(True)

    # Loss Plot
    axs[1].plot(epochs_range, history['train_loss'], 'o-', label='Training Loss')
    axs[1].plot(epochs_range, history['val_loss'], 'o-', label='Validation Loss')
    axs[1].set_title('Model Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    axs[1].grid(True)

    # AUC Plot (only if AUC data is available and not all NaN)
    train_auc_valid = [auc for auc in history.get('train_auc', []) if auc is not None and not np.isnan(auc)]
    val_auc_valid = [auc for auc in history.get('val_auc', []) if auc is not None and not np.isnan(auc)]

    plotted_auc = False
    if train_auc_valid:
        axs[2].plot(epochs_range, history['train_auc'], 'o-', label='Training AUC') # Plot even if some are NaN
        plotted_auc = True
    if val_auc_valid:
        axs[2].plot(epochs_range, history['val_auc'], 'o-', label='Validation AUC') # Plot even if some are NaN
        plotted_auc = True

    if plotted_auc:
        axs[2].set_title('Model AUC')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('AUC')
        axs[2].legend()
        axs[2].grid(True)
        axs[2].set_ylim(bottom=0.0) # Ensure AUC plot starts at 0
    else:
        axs[2].set_title('AUC Data Unavailable')
        axs[2].text(0.5, 0.5, 'No valid AUC data to plot', horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes)

    plt.suptitle('Final Model Training History', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- Other Plotting Functions (Identical to previous version) ---
def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', cmap=plt.cm.Oranges, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    fmt = '.2f' if normalize else 'd'
    if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print(f"{title}:\n{cm}") # Print matrix values if desired
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, xticklabels=classes, yticklabels=classes, annot_kws={"size": 12})
    plt.title(title, fontsize=14); plt.ylabel('True Label', fontsize=12); plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(fontsize=10); plt.yticks(fontsize=10); plt.tight_layout(); plt.show()

def plot_roc_curve(y_true, y_score, title='Receiver Operating Characteristic (ROC) Curve'):
    fpr, tpr, _ = roc_curve(y_true, y_score); roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6)); plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12); plt.ylabel('True Positive Rate', fontsize=12); plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=11); plt.grid(True); plt.show()

def plot_precision_recall_curve(y_true, y_score, title='Precision-Recall Curve'):
    precision, recall, _ = precision_recall_curve(y_true, y_score); average_precision = average_precision_score(y_true, y_score)
    plt.figure(figsize=(8, 6)); plt.step(recall, precision, where='post', color='blue', lw=2, label=f'Precision-Recall curve (AP = {average_precision:.3f})')
    plt.xlabel('Recall', fontsize=12); plt.ylabel('Precision', fontsize=12); plt.ylim([0.0, 1.05]); plt.xlim([0.0, 1.0])
    plt.title(title, fontsize=14); plt.legend(loc="upper right", fontsize=11); plt.grid(True); plt.show()

def get_layer(model, layer_name):
    if '.' in layer_name:
        parts = layer_name.split('.'); obj = model
        for part in parts[:-1]:
            if part.isdigit() and isinstance(obj, (nn.ModuleList, nn.Sequential)): obj = obj[int(part)]
            else: obj = getattr(obj, part)
        last_part = parts[-1]
        if last_part.isdigit() and isinstance(obj, (nn.ModuleList, nn.Sequential)): return obj[int(last_part)]
        else: return getattr(obj, last_part)
    else: return getattr(model, layer_name)

activation = {}
def get_activation_hook(name):
    def hook(model, input, output): activation[name] = output.detach()
    return hook

def plot_feature_maps(model, layer_name, sample_image_tensor, grid_size=None, device='cpu'):
    model.eval(); target_layer = None; hook_handle = None
    try:
        target_layer = get_layer(model, layer_name)
        if not isinstance(target_layer, nn.Module): print(f"Error: Target '{layer_name}' not nn.Module."); return
        hook_handle = target_layer.register_forward_hook(get_activation_hook(layer_name))
        if sample_image_tensor.dim() == 3: sample_image_tensor = sample_image_tensor.unsqueeze(0)
        sample_image_tensor = sample_image_tensor.to(device)
        with torch.no_grad(): _ = model(sample_image_tensor)
        act = activation.get(layer_name)
        if act is None: print(f"Error: No activation for '{layer_name}'."); return
        if act.dim() != 4: print(f"Error: Activation shape {act.shape} != 4D."); return
        act = act.squeeze(0).cpu(); num_maps = act.shape[0]
        if num_maps == 0: print(f"Error: 0 channels in activation '{layer_name}'."); return
        if grid_size is None: cols = int(math.ceil(math.sqrt(num_maps))); rows = int(math.ceil(num_maps / cols)); grid_size = (rows, cols)
        else: rows, cols = grid_size; # Simplified: Assume valid grid if provided
        print(f"Plotting {num_maps} feature maps from layer '{layer_name}' in a {grid_size} grid...")
        fig, axes = plt.subplots(rows, cols, figsize=(max(3, cols*1.5), max(3, rows*1.5)))
        axes = axes.flatten()
        for i in range(num_maps):
            ax = axes[i]; feature_map = act[i]
            if feature_map.dim() != 2: print(f"Warning: Map {i} shape {feature_map.shape} not 2D."); ax.axis('off'); continue
            im = ax.imshow(feature_map, cmap='viridis'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f'Map {i+1}', fontsize=8)
        for j in range(num_maps, len(axes)): axes[j].axis('off')
        fig.suptitle(f'Feature Maps: Layer "{layer_name}"', fontsize=14); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()
    except Exception as e: print(f"Feature map plotting error: {e}")
    finally:
        if hook_handle: hook_handle.remove()
        activation.clear()

def imshow_unnormalized(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img_tensor = img_tensor.cpu().clone()
    for t, m, s in zip(img_tensor, mean, std): t.mul_(s).add_(m)
    img_tensor = img_tensor.clamp(0, 1); plt.imshow(img_tensor.permute(1, 2, 0)); plt.axis('off')


# --- 5. Load LATEST Checkpoint / Initialize State (Identical) ---
start_generation = 0; current_num_extra_layers = 0; best_model_state_dict = None
all_best_accuracies = []; all_best_aucs = []; latest_gen_index = -1; latest_checkpoint_path = None
print(f"\n--- Checking for latest checkpoint in: {CHECKPOINT_DIR} ---")
try:
    # Condensed checkpoint finding logic
    gen_files = {}
    for f in os.listdir(CHECKPOINT_DIR):
        match = re.search(rf'{CHECKPOINT_BASENAME}(\d+){CHECKPOINT_SUFFIX}$', f)
        if match: gen_files[int(match.group(1))] = os.path.join(CHECKPOINT_DIR, f)
    if gen_files:
        latest_gen_index = max(gen_files.keys())
        latest_checkpoint_path = gen_files[latest_gen_index]
        print(f"Latest checkpoint found for completed generation {latest_gen_index + 1}: {latest_checkpoint_path}")
        print("Loading...")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
        current_num_extra_layers = checkpoint.get('num_extra_conv_layers', 0)
        # loaded_gen_index = checkpoint.get('generation', -1) # Redundant check removed
        start_generation = latest_gen_index + 1
        best_model_state_dict = checkpoint['model_state_dict']
        all_best_accuracies = checkpoint.get('all_best_accuracies', [])
        all_best_aucs = checkpoint.get('all_best_aucs', [])
        if start_generation >= NUM_GENERATIONS: print(f"Checkpoint indicates all {NUM_GENERATIONS} gen completed. Exiting."); exit()
        else: print(f"Resuming from Gen {start_generation + 1}/{NUM_GENERATIONS} ({current_num_extra_layers} extra layers). History: {len(all_best_accuracies)} entries.")
    else: print("No valid checkpoint files found. Starting from scratch (Generation 1).")
except FileNotFoundError: print(f"Checkpoint directory not found: {CHECKPOINT_DIR}. Starting fresh.")
except Exception as e:
    print(f"Error finding/loading checkpoint: {e}. Starting fresh.")
    start_generation = 0; current_num_extra_layers = 0; best_model_state_dict = None; all_best_accuracies = []; all_best_aucs = []

# --- 6. Generational Training Loop (Identical structure) ---
criterion = nn.CrossEntropyLoss()
print(f"\n--- Starting Generational Training (Gens {start_generation + 1} to {NUM_GENERATIONS}) ---")
for generation in range(start_generation, NUM_GENERATIONS):
    current_gen_number = generation + 1
    print(f"\n--- Generation {current_gen_number}/{NUM_GENERATIONS} ---")
    next_num_extra_layers = current_num_extra_layers + LAYERS_TO_ADD_PER_GENERATION
    print(f"  Architecture: {next_num_extra_layers} extra conv layers.")
    initial_state_dict_for_gen = best_model_state_dict
    current_gen_seed = GENERATION_DATA_SEED + generation if GENERATION_DATA_SEED is not None else None
    # Use the subset dataset for generational training loader
    train_loader_gen = create_generation_train_loader(train_dataset_subset, BATCH_SIZE, generation_seed=current_gen_seed)
    generation_model_states = []; generation_val_accuracies = []; generation_val_aucs = []

    print(f"  Training {MODELS_PER_GENERATION} models...")
    for model_run in range(MODELS_PER_GENERATION):
        print(f"    Model Run {model_run + 1}/{MODELS_PER_GENERATION}")
        current_model = GrowingCNN(num_classes=NUM_CLASSES, num_extra_conv_layers=next_num_extra_layers).to(device)
        if initial_state_dict_for_gen:
            try:
                missing, unexpected = current_model.load_state_dict(initial_state_dict_for_gen, strict=False)
                # Optional: More detailed logging
                # if missing: print(f"      Loaded weights. New layers: {missing}")
                # if unexpected: print(f"      Warning: Unexpected keys: {unexpected}")
            except Exception as e: print(f"      Error loading state dict: {e}. Initializing randomly.")
        # else: print("      Initializing randomly.") # Only log if needed

        optimizer = optim.Adam(current_model.parameters(), lr=LEARNING_RATE)
        for epoch in range(EPOCHS_PER_GENERATION):
             _, _, _ = train_one_epoch(current_model, train_loader_gen, optimizer, criterion, device, NUM_CLASSES) # Ignore epoch metrics here
        val_loss, val_acc, val_auc, _, _, _ = evaluate_model(current_model, val_loader, criterion, device, NUM_CLASSES)
        print(f"    Eval -> Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, AUC: {val_auc if not np.isnan(val_auc) else 'N/A':.4f}")
        generation_model_states.append(copy.deepcopy(current_model.state_dict())); generation_val_accuracies.append(val_acc); generation_val_aucs.append(val_auc)
        del current_model; torch.cuda.empty_cache() if device.type == 'cuda' else None

    if not generation_val_accuracies: print(f"Warning: No models evaluated in Gen {current_gen_number}. Skipping."); continue

    best_model_index_this_gen = np.argmax(generation_val_accuracies)
    best_accuracy_this_gen = generation_val_accuracies[best_model_index_this_gen]
    best_auc_this_gen = generation_val_aucs[best_model_index_this_gen]
    best_model_state_dict = generation_model_states[best_model_index_this_gen]
    current_num_extra_layers = next_num_extra_layers # Update layer count *after* finding best model
    all_best_accuracies.append(best_accuracy_this_gen); all_best_aucs.append(best_auc_this_gen)
    print(f"--- Best Gen {current_gen_number}: Model {best_model_index_this_gen + 1} ({current_num_extra_layers} layers) -> Acc: {best_accuracy_this_gen:.2f}%, AUC: {best_auc_this_gen if not np.isnan(best_auc_this_gen) else 'N/A':.4f} ---")

    checkpoint_filename_gen = f"{CHECKPOINT_BASENAME}{generation}{CHECKPOINT_SUFFIX}"; checkpoint_path_gen = os.path.join(CHECKPOINT_DIR, checkpoint_filename_gen)
    print(f"  Saving checkpoint Gen {current_gen_number} to {checkpoint_path_gen}...")
    checkpoint_data = {'generation': generation, 'model_state_dict': best_model_state_dict, 'num_extra_conv_layers': current_num_extra_layers,
                       'all_best_accuracies': all_best_accuracies, 'all_best_aucs': all_best_aucs,
                       'config': {'num_classes': NUM_CLASSES, 'img_height': IMG_HEIGHT, 'img_width': IMG_WIDTH, 'learning_rate': LEARNING_RATE}}
    try: torch.save(checkpoint_data, checkpoint_path_gen); print(f"  Checkpoint saved.")
    except Exception as e: print(f"  Error saving checkpoint: {e}")


# --- 7. MODIFIED Final Steps after ALL Generations Complete ---
print("\n--- Generational Training Complete ---")
if best_model_state_dict is not None:
    print(f"Final best architecture has {current_num_extra_layers} extra conv layers.")
    # Instantiate the final model structure
    final_model = GrowingCNN(num_classes=NUM_CLASSES, num_extra_conv_layers=current_num_extra_layers).to(device)
    try:
        # Load the best state dict found during generational training
        final_model.load_state_dict(best_model_state_dict)
        print(f"Loaded best state dict from generational training.")
        print("Best validation accuracies per generation:", [f"{acc:.2f}%" for acc in all_best_accuracies])
        print("Corresponding best validation AUCs:", [f"{auc:.4f}" if not np.isnan(auc) else "N/A" for auc in all_best_aucs])

        # --- <<< NEW: Final Training Run for History Plotting >>> ---
        print(f"\n--- Performing Final Training for {FINAL_TRAINING_EPOCHS} Epochs (for History Plot) ---")
        final_history = {'train_loss': [], 'train_acc': [], 'train_auc': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}
        # Use the same training subset and validation loader as during generations
        final_train_loader = DataLoader(train_dataset_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        final_optimizer = optim.Adam(final_model.parameters(), lr=LEARNING_RATE / 10) # Optionally use smaller LR for fine-tuning

        for epoch in range(FINAL_TRAINING_EPOCHS):
            epoch_start_time = time.time()
            # Train one epoch
            train_loss, train_acc, train_auc = train_one_epoch(
                final_model, final_train_loader, final_optimizer, criterion, device, NUM_CLASSES
            )
            # Evaluate on validation set
            val_loss, val_acc, val_auc, _, _, _ = evaluate_model(
                final_model, val_loader, criterion, device, NUM_CLASSES
            )
            epoch_end_time = time.time()

            # Store history
            final_history['train_loss'].append(train_loss)
            final_history['train_acc'].append(train_acc)
            final_history['train_auc'].append(train_auc)
            final_history['val_loss'].append(val_loss)
            final_history['val_acc'].append(val_acc)
            final_history['val_auc'].append(val_auc)

            print(f"  Epoch {epoch + 1}/{FINAL_TRAINING_EPOCHS} "
                  f"({epoch_end_time - epoch_start_time:.1f}s) - "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, AUC: {train_auc if not np.isnan(train_auc) else 'N/A':.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, AUC: {val_auc if not np.isnan(val_auc) else 'N/A':.4f}")

        print("--- Final Training Complete ---")

        # --- <<< NEW: Plot Final Model History >>> ---
        print("\n--- Plotting Final Model Training History ---")
        plot_history(final_history, FINAL_TRAINING_EPOCHS)


        # --- FINAL EVALUATION AND PLOTTING (using the model *after* final training) ---
        print("\n--- Evaluating Final Model on TEST Set ---")
        test_loss, test_accuracy, test_auc, test_targets, test_proba_positive, test_pred_labels = evaluate_model(
            final_model, test_loader, criterion, device, NUM_CLASSES
        )
        print(f"Final Model Performance - Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%, Test AUC: {test_auc if not np.isnan(test_auc) else 'N/A':.4f}")

        # --- Generate and Show Plots (CM, ROC, PR, Features) ---
        print("\n--- Generating Test Set Visualizations ---")
        # 1. Confusion Matrix (Default Threshold ~0.5)
        if test_targets and test_pred_labels: plot_confusion_matrix(test_targets, test_pred_labels, classes=class_names, title=f'Confusion Matrix (Test Set, Default Threshold)', cmap=plt.cm.Blues)
        else: print("Skipping default CM plot.")
        # 2. Confusion Matrix (Custom Threshold)
        if test_targets and test_proba_positive: custom_preds = [1 if prob >= CONFUSION_MATRIX_THRESHOLD else 0 for prob in test_proba_positive]; plot_confusion_matrix(test_targets, custom_preds, classes=class_names, title=f'Confusion Matrix (Test Set, Threshold={CONFUSION_MATRIX_THRESHOLD})', cmap=plt.cm.Oranges)
        else: print("Skipping custom CM plot.")
        # 3. ROC Curve
        if test_targets and test_proba_positive and not np.isnan(test_auc): plot_roc_curve(test_targets, test_proba_positive, title='ROC Curve (Test Set)')
        elif np.isnan(test_auc): print("Skipping ROC plot: AUC NaN.")
        else: print("Skipping ROC plot: Missing data.")
        # 4. Precision-Recall Curve
        if test_targets and test_proba_positive and len(np.unique(test_targets)) > 1: plot_precision_recall_curve(test_targets, test_proba_positive, title='Precision-Recall Curve (Test Set)')
        elif not (test_targets and test_proba_positive): print("Skipping PR plot: Missing data.")
        else: print("Skipping PR plot: Only one class.")
        # 5. Feature Maps
        print(f"\n--- Generating Feature Maps for Layer '{FEATURE_MAP_LAYER_NAME}' ---")
        if FEATURE_MAP_SAMPLE_INDEX < len(test_dataset_folder):
            try:
                sample_image_raw, sample_label_idx = test_dataset_folder[FEATURE_MAP_SAMPLE_INDEX]; sample_label = class_names[sample_label_idx]
                plt.figure(figsize=(4,4)); imshow_unnormalized(sample_image_raw); plt.title(f"Input Image for Feature Maps\nIndex: {FEATURE_MAP_SAMPLE_INDEX}, True Label: {sample_label}"); plt.show()
                plot_feature_maps(final_model, FEATURE_MAP_LAYER_NAME, sample_image_raw, device=device)
            except Exception as e: print(f"Error generating feature maps: {e}")
        else: print(f"Error: Sample index {FEATURE_MAP_SAMPLE_INDEX} OOB for test dataset (size {len(test_dataset_folder)}).")


        # --- Save final model state ---
        # Use the generation variable from the end of the loop for naming consistency
        final_model_save_path = os.path.join(CHECKPOINT_DIR, f'final_model_trained_gen_{generation}_layers_{current_num_extra_layers}.pth')
        print(f"\nSaving final trained model state dict to: {final_model_save_path}")
        try:
            torch.save(final_model.state_dict(), final_model_save_path)
            print("Final model state dict saved successfully.")
        except Exception as e: print(f"Error saving final model state dict: {e}")

    except Exception as e:
        print(f"Error during final steps (loading, final training, plotting, saving): {e}")
        import traceback
        traceback.print_exc()

else:
    print("No valid best_model_state_dict available after generational training loop. Cannot perform final steps.")

print("\n--- Script Finished ---")

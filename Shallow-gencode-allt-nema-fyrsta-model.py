#Þetta er kóðinn sem við notuðum fyrir alla hina þjálfunina á CNN generational módelunum sem komu á eftir því fyrsta, en dýpkuðu ekki.



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


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset, Dataset
from torchvision import datasets, transforms
import numpy as np
import copy
import os
import re # Import regular expressions for parsing filenames
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import time

# --- Google Drive Integration ---
from google.colab import drive
import shutil

# --- Configuration ---
NUM_GENERATIONS = 10  # Total desired generations
MODELS_PER_GENERATION = 5
EPOCHS_PER_GENERATION = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GENERATION_DATA_SEED = 42
VAL_SPLIT = 0.1

# --- Data Paths ---
TRAIN_DATA_DIR = '/content/Training_images'
VAL_DATA_DIR = '/content/Validation_images'

# --- Model Parameters ---
IMG_HEIGHT, IMG_WIDTH = 299, 299
NUM_CLASSES = 2

# --- Checkpoint Configuration ---
DRIVE_MOUNT_POINT = '/content/drive'
# **IMPORTANT:** Set to your desired folder in Google Drive
CHECKPOINT_DIR = os.path.join(DRIVE_MOUNT_POINT, 'MyDrive', 'Gervigreind - verkefni') # Example Path
CHECKPOINT_BASENAME = 'generational_checkpoint_gen_' # Base name for generation files
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
    print(f"Error mounting Google Drive: {e}")
    print("Checkpoints will not be saved/loaded from Drive.")
    # Fallback or exit if Drive is critical
    CHECKPOINT_DIR = './checkpoints_local' # Example local fallback
    # os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Using local checkpoint directory: {CHECKPOINT_DIR}")
    #exit() # Exit if Drive is essential

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
    if len(class_names) != NUM_CLASSES: print(f"Warning: Class mismatch")
except Exception as e: print(f"Data loading error: {e}"); raise
num_train_val = len(full_train_dataset_train_tf); indices = list(range(num_train_val))
split = int(np.floor(VAL_SPLIT * num_train_val)); np.random.seed(42); np.random.shuffle(indices)
train_idx, val_idx = indices[split:], indices[:split]
train_dataset = Subset(full_train_dataset_train_tf, train_idx); val_dataset = Subset(full_train_dataset_val_tf, val_idx)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset_folder, batch_size=BATCH_SIZE, shuffle=False)
print(f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset_folder)} test samples.")

def create_generation_train_loader(dataset, batch_size, generation_seed=None):
    g = None
    if generation_seed is not None: g = torch.Generator(); g.manual_seed(generation_seed)
    print(f"  Shuffling training data for generation with seed: {generation_seed if generation_seed else 'Random'}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g, num_workers=2, pin_memory=True)

# --- 2. Model Definition (Identical) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU(); self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU(); self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._calculate_flattened_size()
        self.fc1 = nn.Linear(self.flattened_size, 128); self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    def _calculate_flattened_size(self):
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH)
                dummy_output = self.pool2(self.relu2(self.conv2(self.pool1(self.relu1(self.conv1(dummy_input))))))
                self.flattened_size = int(np.prod(dummy_output.shape[1:]))
        except Exception as e:
            print(f"Warning: Error calculating flattened size ({e}). Using fallback.")
            self.flattened_size = 32 * 74 * 74 # ADJUST FALLBACK IF NEEDED
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x))); x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1); x = self.relu3(self.fc1(x)); x = self.fc2(x)
        return x

# --- 3. Training & Evaluation Functions (Identical) ---
def train_one_epoch(model, train_loader, optimizer, criterion, device, num_classes):
    model.train(); running_loss = 0.0; epoch_correct = 0; epoch_total = 0
    all_targets = []; all_preds_proba = []
    for data, target in train_loader:
        data, target = data.to(device), target.to(device); optimizer.zero_grad(); outputs = model(data)
        if num_classes == 2: # CE Loss
            loss = criterion(outputs, target); pred_prob = torch.softmax(outputs, dim=1); proba_for_auc = pred_prob[:, 1]; _, predicted = torch.max(pred_prob.data, 1)
        else: # Multiclass
            loss = criterion(outputs, target); pred_prob = torch.softmax(outputs, dim=1); _, predicted = torch.max(outputs.data, 1); proba_for_auc = pred_prob
        loss.backward(); optimizer.step(); running_loss += loss.item(); epoch_total += target.size(0); epoch_correct += (predicted == target).sum().item()
        all_targets.extend(target.cpu().numpy())
        if num_classes == 2: all_preds_proba.extend(proba_for_auc.detach().cpu().numpy())
    avg_epoch_loss = running_loss / len(train_loader); avg_epoch_acc = 100.0 * epoch_correct / epoch_total; epoch_auc = np.nan
    if num_classes == 2 and len(all_preds_proba) > 0:
        try:
            if len(np.unique(all_targets)) > 1: epoch_auc = roc_auc_score(all_targets, all_preds_proba)
        except Exception: pass
    return avg_epoch_loss, avg_epoch_acc, epoch_auc
def evaluate_model(model, val_loader, criterion, device, num_classes):
    model.eval(); val_loss = 0.0; correct = 0; total = 0; all_targets = []; all_preds_proba = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device); outputs = model(data)
            if num_classes == 2: # CE Loss
                loss = criterion(outputs, target); pred_prob = torch.softmax(outputs, dim=1); proba_for_auc = pred_prob[:, 1]; _, predicted = torch.max(pred_prob.data, 1)
            else: # Multiclass
                loss = criterion(outputs, target); pred_prob = torch.softmax(outputs, dim=1); _, predicted = torch.max(outputs.data, 1); proba_for_auc = pred_prob
            val_loss += loss.item(); total += target.size(0); correct += (predicted == target).sum().item(); all_targets.extend(target.cpu().numpy())
            if num_classes == 2: all_preds_proba.extend(proba_for_auc.detach().cpu().numpy())
    avg_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0; accuracy = 100.0 * correct / total if total > 0 else 0; val_auc = np.nan
    if num_classes == 2 and len(all_preds_proba) > 0:
        try:
            if len(np.unique(all_targets)) > 1: val_auc = roc_auc_score(all_targets, all_preds_proba)
        except Exception: pass
    return avg_loss, accuracy, val_auc

# --- 4. Plotting Function (Optional, Identical) ---
def plot_history(history, epochs):
    epochs_range = range(epochs); fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    axs[0].plot(epochs_range, history['train_acc'], label='Training Acc'); axs[0].plot(epochs_range, history['val_acc'], label='Validation Acc'); axs[0].set_title('Accuracy'); axs[0].set_xlabel('Epoch'); axs[0].set_ylabel('Accuracy (%)'); axs[0].legend(); axs[0].grid(True)
    axs[1].plot(epochs_range, history['train_loss'], label='Training Loss'); axs[1].plot(epochs_range, history['val_loss'], label='Validation Loss'); axs[1].set_title('Loss'); axs[1].set_xlabel('Epoch'); axs[1].set_ylabel('Loss'); axs[1].legend(); axs[1].grid(True)
    valid_train_auc = [x for x in history.get('train_auc',[]) if x is not None and not np.isnan(x)]; valid_val_auc = [x for x in history.get('val_auc',[]) if x is not None and not np.isnan(x)]
    train_idx = [i for i, x in enumerate(history.get('train_auc',[])) if x is not None and not np.isnan(x)]; val_idx = [i for i, x in enumerate(history.get('val_auc',[])) if x is not None and not np.isnan(x)]
    plotted_auc = False
    if valid_train_auc: axs[2].plot([epochs_range[i] for i in train_idx], valid_train_auc, label='Training AUC'); plotted_auc = True
    if valid_val_auc: axs[2].plot([epochs_range[i] for i in val_idx], valid_val_auc, label='Validation AUC'); plotted_auc = True
    if plotted_auc: axs[2].set_title('AUC'); axs[2].set_xlabel('Epoch'); axs[2].set_ylabel('AUC'); axs[2].legend(); axs[2].grid(True)
    else: axs[2].set_title('AUC Unavailable')
    plt.tight_layout(); plt.show()

# --- 5. Load LATEST Checkpoint / Initialize State ---
start_generation = 0
best_model_state_dict = None
all_best_accuracies = []
all_best_aucs = []
latest_gen_index = -1
latest_checkpoint_path = None

print(f"\n--- Checking for latest checkpoint in: {CHECKPOINT_DIR} ---")
try:
    # Find all checkpoint files matching the pattern
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR)
                        if f.startswith(CHECKPOINT_BASENAME) and f.endswith(CHECKPOINT_SUFFIX)]

    # Parse generation numbers and find the latest
    for filename in checkpoint_files:
        match = re.search(r'_gen_(\d+)\.pth$', filename)
        if match:
            try:
                gen_index = int(match.group(1))
                if gen_index > latest_gen_index:
                    latest_gen_index = gen_index
                    latest_checkpoint_path = os.path.join(CHECKPOINT_DIR, filename)
            except ValueError:
                print(f"  Warning: Could not parse generation number from '{filename}'. Skipping.")
                continue # Skip files with non-integer generation numbers

    if latest_checkpoint_path:
        print(f"Latest checkpoint found for completed generation {latest_gen_index + 1}: {latest_checkpoint_path}")
        print("Loading...")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)

        # Load state from the latest checkpoint
        # Ensure the loaded generation index matches expectations
        loaded_gen_index = checkpoint.get('generation', -1)
        if loaded_gen_index != latest_gen_index:
             print(f"  Warning: Filename generation index ({latest_gen_index}) does not match index saved inside checkpoint ({loaded_gen_index}). Using filename index.")
             # Decide if this is critical. Using filename index is usually safer if consistent.

        start_generation = latest_gen_index + 1 # Resume from the *next* generation index
        best_model_state_dict = checkpoint['model_state_dict']
        all_best_accuracies = checkpoint.get('all_best_accuracies', [])
        all_best_aucs = checkpoint.get('all_best_aucs', [])

        if start_generation >= NUM_GENERATIONS:
             print(f"Checkpoint indicates generation {latest_gen_index + 1} was already completed.")
             print("Target number of generations reached. Nothing more to train.")
             exit()
        else:
            print(f"Resuming training from Generation {start_generation + 1}/{NUM_GENERATIONS}")
            print(f"Loaded initial state from best model of Gen {latest_gen_index + 1}.")
            print(f"Accuracy history loaded: {len(all_best_accuracies)} entries.")

    else:
        print("No valid checkpoint files found matching the pattern.")
        print("Starting training from scratch (Generation 1).")

except FileNotFoundError:
    print(f"Checkpoint directory not found: {CHECKPOINT_DIR}")
    print("Starting training from scratch (Generation 1).")
except Exception as e:
    print(f"Error finding or loading latest checkpoint: {e}")
    print("Starting training from scratch (Generation 1).")
    start_generation = 0
    best_model_state_dict = None
    all_best_accuracies = []
    all_best_aucs = []


# --- 6. Modified Generational Training Loop ---
criterion = nn.CrossEntropyLoss()

print(f"\n--- Starting Generational Training (Gens {start_generation + 1} to {NUM_GENERATIONS}) ---")

for generation in range(start_generation, NUM_GENERATIONS):
    current_gen_number = generation + 1
    print(f"\n--- Generation {current_gen_number}/{NUM_GENERATIONS} ---")

    # 1. Determine Initial State
    if best_model_state_dict is None:
        print("  Initializing models randomly for the first generation.")
        try:
            initial_model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
            initial_state_dict = copy.deepcopy(initial_model.state_dict()); del initial_model
        except Exception as e: print(f"Error creating initial model: {e}"); raise
    else:
        print("  Initializing models from the best model of the previous generation (or checkpoint).")
        initial_state_dict = best_model_state_dict

    # 2. Prepare Data Loader
    current_gen_seed = None
    if GENERATION_DATA_SEED is not None: current_gen_seed = GENERATION_DATA_SEED + generation
    train_loader_gen = create_generation_train_loader(train_dataset, BATCH_SIZE, generation_seed=current_gen_seed)

    generation_model_states = []; generation_val_accuracies = []; generation_val_aucs = []

    # 3. Train and Evaluate Models
    print(f"  Starting training for {MODELS_PER_GENERATION} models in Gen {current_gen_number}...")
    for model_run in range(MODELS_PER_GENERATION):
        current_model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
        try:
            current_model.load_state_dict(copy.deepcopy(initial_state_dict))
        except Exception as e: print(f"Error loading initial state dict into model {model_run+1}: {e}"); continue
        optimizer = optim.Adam(current_model.parameters(), lr=LEARNING_RATE)
        for epoch in range(EPOCHS_PER_GENERATION): _, _, _ = train_one_epoch(current_model, train_loader_gen, optimizer, criterion, device, NUM_CLASSES)
        val_loss, val_accuracy, val_auc = evaluate_model(current_model, val_loader, criterion, device, NUM_CLASSES)
        print(f"    Model {model_run + 1} Eval -> Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.2f}%, AUC: {val_auc:.4f}") # Keep this eval print
        generation_model_states.append(copy.deepcopy(current_model.state_dict())); generation_val_accuracies.append(val_accuracy); generation_val_aucs.append(val_auc)
        del current_model;
        if device.type == 'cuda': torch.cuda.empty_cache()

    if not generation_val_accuracies: print(f"Warning: No models evaluated in Gen {current_gen_number}. Skipping checkpoint."); continue

    # 4. Select Best Model
    best_model_index_this_gen = np.argmax(generation_val_accuracies)
    best_accuracy_this_gen = generation_val_accuracies[best_model_index_this_gen]
    valid_aucs = [auc for auc in generation_val_aucs if auc is not None and not np.isnan(auc)]
    best_auc_this_gen = generation_val_aucs[best_model_index_this_gen] if not np.isnan(generation_val_aucs[best_model_index_this_gen]) else (max(valid_aucs) if valid_aucs else np.nan)
    best_model_state_dict = generation_model_states[best_model_index_this_gen]
    all_best_accuracies.append(best_accuracy_this_gen); all_best_aucs.append(best_auc_this_gen)
    print(f"--- Best model in Gen {current_gen_number}: Model {best_model_index_this_gen + 1} (Val Acc: {best_accuracy_this_gen:.2f}%, Val AUC: {best_auc_this_gen:.4f}) ---")

    # --- 5. Save INDIVIDUAL Checkpoint to Google Drive ---
    # Construct the specific filename for this generation
    checkpoint_filename_gen = f"{CHECKPOINT_BASENAME}{generation}{CHECKPOINT_SUFFIX}" # e.g., _gen_0.pth, _gen_1.pth
    checkpoint_path_gen = os.path.join(CHECKPOINT_DIR, checkpoint_filename_gen)
    print(f"  Saving checkpoint for completed Generation {current_gen_number} to {checkpoint_path_gen}...")

    checkpoint_data = {
        'generation': generation, # Save the index of the *completed* generation (0, 1, 2...)
        'model_state_dict': best_model_state_dict,
        'all_best_accuracies': all_best_accuracies,
        'all_best_aucs': all_best_aucs,
        'config': {'num_classes': NUM_CLASSES, 'img_height': IMG_HEIGHT, 'img_width': IMG_WIDTH, 'learning_rate': LEARNING_RATE}
    }
    try:
        torch.save(checkpoint_data, checkpoint_path_gen) # Save to the generation-specific path
        print(f"  Checkpoint successfully saved.")
    except Exception as e:
        print(f"  Error saving checkpoint to Google Drive: {e}")

# --- 7. Final Steps after ALL Generations Complete ---
print("\n--- Generational Training Complete ---")
if best_model_state_dict is not None:
    final_model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    try:
        final_model.load_state_dict(best_model_state_dict)
        print("Final model loaded from the best state found across all trained generations.")
        print("Best validation accuracies per generation:", [f"{acc:.2f}%" for acc in all_best_accuracies])
        print("Corresponding best validation AUCs:", [f"{auc:.4f}" if not np.isnan(auc) else "N/A" for auc in all_best_aucs])

        # Evaluate final model
        print("\nEvaluating final model on the TEST set...")
        test_loss, test_accuracy, test_auc = evaluate_model(final_model, test_loader, criterion, device, NUM_CLASSES)
        print(f"Final Model Performance - Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%, Test AUC: {test_auc:.4f}")

        # --- UNCOMMENT BELOW TO ENABLE FINAL PLOTTING ---
        print("\nCollecting training history for the final model...")
        history_model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
        history_model.load_state_dict(best_model_state_dict)
        history_optimizer = optim.Adam(history_model.parameters(), lr=LEARNING_RATE)
        final_gen_seed = None
        if GENERATION_DATA_SEED is not None: final_gen_seed = GENERATION_DATA_SEED + (NUM_GENERATIONS - 1)
        history_train_loader = create_generation_train_loader(train_dataset, BATCH_SIZE, generation_seed=final_gen_seed)
        history = {'train_loss': [], 'train_acc': [], 'train_auc': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}
        num_history_epochs = EPOCHS_PER_GENERATION
        print(f"Collecting history over {num_history_epochs} epochs...")
        for epoch in range(num_history_epochs):
            train_loss, train_acc, train_auc = train_one_epoch(history_model, history_train_loader, history_optimizer, criterion, device, NUM_CLASSES)
            val_loss, val_acc, val_auc = evaluate_model(history_model, val_loader, criterion, device, NUM_CLASSES)
            history['train_loss'].append(train_loss); history['train_acc'].append(train_acc); history['train_auc'].append(train_auc)
            history['val_loss'].append(val_loss); history['val_acc'].append(val_acc); history['val_auc'].append(val_auc)
            print(f"  History Epoch {epoch+1}/{num_history_epochs}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}%, AUC={train_auc if not np.isnan(train_auc) else 'N/A':.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%, AUC={val_auc if not np.isnan(val_auc) else 'N/A':.4f}")
        print("\nPlotting training history of the final model...")
        plot_history(history, num_history_epochs)
        # --- END OF OPTIONAL PLOTTING SECTION ---

        # Save final model
        final_model_save_path = os.path.join(CHECKPOINT_DIR, 'final_generational_model_complete.pth')
        print(f"\nSaving final trained model to: {final_model_save_path}")
        try:
            torch.save(final_model.state_dict(), final_model_save_path)
            print("Final model saved successfully.")
        except Exception as e: print(f"Error saving final model: {e}")
    except Exception as e: print(f"Error loading final model state dict: {e}")
else: print("No valid model state available after training loop.")
print("\n--- Script Finished ---")

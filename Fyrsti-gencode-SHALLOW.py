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


#Þetta er kóðinn sem við þjálfuðum dæmið á, nema hvað þá var gen=5.
#Að auki sé ég (Ellert) nú að kóðinn er kolrangur, ég held að hann hafi
#ekki bætt við nýjum layerum á gamla, heldur bara endurþjálfað þau endalaust.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset, Dataset
from torchvision import datasets, transforms
import numpy as np
import copy
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import time # For potential delays if needed

# --- Google Drive Integration ---
from google.colab import drive
import shutil # For copying files if needed (though torch.save can write directly)

# --- Configuration ---
NUM_GENERATIONS = 10  # Total desired generations
MODELS_PER_GENERATION = 5
EPOCHS_PER_GENERATION = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GENERATION_DATA_SEED = 42
VAL_SPLIT = 0.1

# --- Data Paths ---
TRAIN_DATA_DIR = '/content/Training_images' # Example
VAL_DATA_DIR = '/content/Validation_images' # Example

# --- Model Parameters ---
IMG_HEIGHT, IMG_WIDTH = 299, 299
NUM_CLASSES = 2 # Assuming binary classification

# --- Checkpoint Configuration ---
DRIVE_MOUNT_POINT = '/content/drive'
# **IMPORTANT:** Choose a folder in your Google Drive for checkpoints
CHECKPOINT_DIR = os.path.join(DRIVE_MOUNT_POINT, 'MyDrive', 'Colab_Checkpoints', 'GenerationalTraining')
CHECKPOINT_FILENAME = 'generational_checkpoint.pth'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)

# --- Setup Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Mount Google Drive ---
print("Mounting Google Drive...")
try:
    drive.mount(DRIVE_MOUNT_POINT, force_remount=True) # force_remount can be helpful
    print("Google Drive mounted successfully.")
    # Create checkpoint directory if it doesn't exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Checkpoint directory ensured: {CHECKPOINT_DIR}")
except Exception as e:
    print(f"Error mounting Google Drive: {e}")
    print("Checkpoints will not be saved/loaded from Drive.")
    # Optionally, you could fallback to local saving or exit
    CHECKPOINT_PATH = './generational_checkpoint.pth' # Example fallback
    # exit() # Or raise an error

# --- 1. Data Loading and Preparation (Identical - keep your preferred setup) ---
# (Assuming the data loading code from previous answers is here)
# ... (transforms, ImageFolder, Subset, DataLoader definitions) ...
train_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop((IMG_HEIGHT, IMG_WIDTH), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_test_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
try:
    full_train_dataset_train_tf = datasets.ImageFolder(TRAIN_DATA_DIR, transform=train_transform)
    full_train_dataset_val_tf = datasets.ImageFolder(TRAIN_DATA_DIR, transform=val_test_transform)
    test_dataset_folder = datasets.ImageFolder(VAL_DATA_DIR, transform=val_test_transform)
    class_names = full_train_dataset_train_tf.classes
    if len(class_names) != NUM_CLASSES: print(f"Warning: Class mismatch")
except Exception as e: print(f"Data loading error: {e}"); raise
num_train_val = len(full_train_dataset_train_tf)
indices = list(range(num_train_val)); split = int(np.floor(VAL_SPLIT * num_train_val))
np.random.seed(42); np.random.shuffle(indices); train_idx, val_idx = indices[split:], indices[:split]
train_dataset = Subset(full_train_dataset_train_tf, train_idx)
val_dataset = Subset(full_train_dataset_val_tf, val_idx)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset_folder, batch_size=BATCH_SIZE, shuffle=False)
print(f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset_folder)} test samples.")

def create_generation_train_loader(dataset, batch_size, generation_seed=None):
    g = None
    if generation_seed is not None: g = torch.Generator(); g.manual_seed(generation_seed)
    print(f"  Shuffling training data for generation with seed: {generation_seed if generation_seed else 'Random'}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g, num_workers=2, pin_memory=True)


# --- 2. Model Definition (Identical) ---
# (Make sure this class definition matches the one used for potential checkpoint)
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
            with torch.no_grad(): # No need for grads during size calculation
                dummy_input = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH)
                dummy_output = self.pool2(self.relu2(self.conv2(self.pool1(self.relu1(self.conv1(dummy_input))))))
                self.flattened_size = int(np.prod(dummy_output.shape[1:]))
        except Exception as e:
            print(f"Warning: Error calculating flattened size ({e}). Using fallback if available, else Linear layer might fail.")
            self.flattened_size = 32 * 74 * 74 # Example fallback based on 299x299 -> 74x74 * 32 channels - ADJUST IF NEEDED

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1); x = self.relu3(self.fc1(x)); x = self.fc2(x)
        return x

# --- 3. Training & Evaluation Functions (Identical) ---
# (train_one_epoch and evaluate_model definitions go here)
# ... train_one_epoch function ...
def train_one_epoch(model, train_loader, optimizer, criterion, device, num_classes):
    model.train(); running_loss = 0.0; epoch_correct = 0; epoch_total = 0
    all_targets = []; all_preds_proba = []
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(); outputs = model(data)
        if num_classes == 2: # Assuming CE Loss
            loss = criterion(outputs, target); pred_prob = torch.softmax(outputs, dim=1)
            proba_for_auc = pred_prob[:, 1]; _, predicted = torch.max(pred_prob.data, 1)
        else: # Multiclass
            loss = criterion(outputs, target); pred_prob = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1); proba_for_auc = pred_prob
        loss.backward(); optimizer.step(); running_loss += loss.item()
        epoch_total += target.size(0); epoch_correct += (predicted == target).sum().item()
        all_targets.extend(target.cpu().numpy())
        if num_classes == 2: all_preds_proba.extend(proba_for_auc.detach().cpu().numpy())
    avg_epoch_loss = running_loss / len(train_loader); avg_epoch_acc = 100.0 * epoch_correct / epoch_total
    epoch_auc = np.nan
    if num_classes == 2 and len(all_preds_proba) > 0:
        try:
            if len(np.unique(all_targets)) > 1: epoch_auc = roc_auc_score(all_targets, all_preds_proba)
        except Exception: pass # Keep NaN
    return avg_epoch_loss, avg_epoch_acc, epoch_auc

# ... evaluate_model function ...
def evaluate_model(model, val_loader, criterion, device, num_classes):
    model.eval(); val_loss = 0.0; correct = 0; total = 0
    all_targets = []; all_preds_proba = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device); outputs = model(data)
            if num_classes == 2: # Assuming CE Loss
                loss = criterion(outputs, target); pred_prob = torch.softmax(outputs, dim=1)
                proba_for_auc = pred_prob[:, 1]; _, predicted = torch.max(pred_prob.data, 1)
            else: # Multiclass
                loss = criterion(outputs, target); pred_prob = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1); proba_for_auc = pred_prob
            val_loss += loss.item(); total += target.size(0); correct += (predicted == target).sum().item()
            all_targets.extend(target.cpu().numpy())
            if num_classes == 2: all_preds_proba.extend(proba_for_auc.detach().cpu().numpy())
    avg_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
    accuracy = 100.0 * correct / total if total > 0 else 0
    val_auc = np.nan
    if num_classes == 2 and len(all_preds_proba) > 0:
        try:
            if len(np.unique(all_targets)) > 1: val_auc = roc_auc_score(all_targets, all_preds_proba)
        except Exception: pass # Keep NaN
    return avg_loss, accuracy, val_auc

# --- 4. Plotting Function (Optional, Identical) ---
# (plot_history definition can go here if you want plotting at the end)
# ... plot_history function ...
def plot_history(history, epochs):
    epochs_range = range(epochs); fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    # Accuracy
    axs[0].plot(epochs_range, history['train_acc'], label='Training Acc'); axs[0].plot(epochs_range, history['val_acc'], label='Validation Acc')
    axs[0].set_title('Accuracy'); axs[0].set_xlabel('Epoch'); axs[0].set_ylabel('Accuracy (%)'); axs[0].legend(); axs[0].grid(True)
    # Loss
    axs[1].plot(epochs_range, history['train_loss'], label='Training Loss'); axs[1].plot(epochs_range, history['val_loss'], label='Validation Loss')
    axs[1].set_title('Loss'); axs[1].set_xlabel('Epoch'); axs[1].set_ylabel('Loss'); axs[1].legend(); axs[1].grid(True)
    # AUC
    valid_train_auc = [x for x in history.get('train_auc',[]) if x is not None and not np.isnan(x)]
    valid_val_auc = [x for x in history.get('val_auc',[]) if x is not None and not np.isnan(x)]
    train_idx = [i for i, x in enumerate(history.get('train_auc',[])) if x is not None and not np.isnan(x)]
    val_idx = [i for i, x in enumerate(history.get('val_auc',[])) if x is not None and not np.isnan(x)]
    plotted_auc = False
    if valid_train_auc: axs[2].plot([epochs_range[i] for i in train_idx], valid_train_auc, label='Training AUC'); plotted_auc = True
    if valid_val_auc: axs[2].plot([epochs_range[i] for i in val_idx], valid_val_auc, label='Validation AUC'); plotted_auc = True
    if plotted_auc: axs[2].set_title('AUC'); axs[2].set_xlabel('Epoch'); axs[2].set_ylabel('AUC'); axs[2].legend(); axs[2].grid(True)
    else: axs[2].set_title('AUC Unavailable')
    plt.tight_layout(); plt.show()


# --- 5. Load Checkpoint / Initialize State ---
start_generation = 0
best_model_state_dict = None
all_best_accuracies = []
# Add other metrics you want to resume, e.g., AUCs
all_best_aucs = [] # Example

print(f"\n--- Checking for checkpoint at: {CHECKPOINT_PATH} ---")
if os.path.exists(CHECKPOINT_PATH):
    try:
        print("Checkpoint file found. Loading...")
        # Use map_location to load correctly regardless of saved device
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

        # Load saved state
        start_generation = checkpoint['generation'] + 1 # Resume from the *next* generation
        best_model_state_dict = checkpoint['model_state_dict']
        all_best_accuracies = checkpoint.get('all_best_accuracies', []) # Use .get for backward compatibility
        all_best_aucs = checkpoint.get('all_best_aucs', []) # Load saved AUCs if present

        # Validate generation number
        if start_generation >= NUM_GENERATIONS:
             print(f"Checkpoint indicates generation {start_generation-1} was completed.")
             print("Target number of generations already reached. Nothing more to train.")
             # Optionally exit or just let the loop below not run
             exit() # Exit cleanly
        else:
            print(f"Resuming training from Generation {start_generation + 1}/{NUM_GENERATIONS}")
            print(f"Loaded initial state from best model of Gen {start_generation}.")
            print(f"Accuracy history loaded: {len(all_best_accuracies)} entries.")

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting training from scratch (Generation 1).")
        start_generation = 0
        best_model_state_dict = None
        all_best_accuracies = []
        all_best_aucs = []
else:
    print("No checkpoint file found.")
    print("Starting training from scratch (Generation 1).")

# --- 6. Modified Generational Training Loop ---
criterion = nn.CrossEntropyLoss() # Or BCEWithLogitsLoss

print(f"\n--- Starting Generational Training (Gens {start_generation + 1} to {NUM_GENERATIONS}) ---")

# Loop from the determined start_generation up to NUM_GENERATIONS
for generation in range(start_generation, NUM_GENERATIONS):
    current_gen_number = generation + 1
    print(f"\n--- Generation {current_gen_number}/{NUM_GENERATIONS} ---")

    # 1. Determine Initial State for this Generation
    if best_model_state_dict is None: # Should only happen for the very first run (generation 0)
        print("  Initializing models randomly for the first generation.")
        # Need to instantiate the model to get an initial random state
        try:
            initial_model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
            initial_state_dict = copy.deepcopy(initial_model.state_dict())
            del initial_model # Free memory
        except Exception as e:
            print(f"Error creating initial model: {e}. Check SimpleCNN definition and parameters.")
            raise # Stop if we can't even create the initial model
    else:
        print("  Initializing models from the best model of the previous generation (or checkpoint).")
        initial_state_dict = best_model_state_dict # Already loaded or from previous gen

    # 2. Prepare Data Loader
    current_gen_seed = None
    if GENERATION_DATA_SEED is not None:
        current_gen_seed = GENERATION_DATA_SEED + generation # Consistent seed per gen
    train_loader_gen = create_generation_train_loader(
        train_dataset, BATCH_SIZE, generation_seed=current_gen_seed
    )

    generation_model_states = []
    generation_val_accuracies = []
    generation_val_aucs = []

    # 3. Train and Evaluate Models within the Generation
    print(f"  Starting training for {MODELS_PER_GENERATION} models in Gen {current_gen_number}...")
    for model_run in range(MODELS_PER_GENERATION):
        # print(f"    Training model {model_run + 1}/{MODELS_PER_GENERATION}...") # Verbose
        current_model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
        try:
            current_model.load_state_dict(copy.deepcopy(initial_state_dict))
        except Exception as e:
             print(f"Error loading initial state dict into model {model_run+1}: {e}")
             print("Check if model definition changed or checkpoint is incompatible.")
             # Decide how to handle: skip model, stop, etc.
             continue # Skip this model run

        optimizer = optim.Adam(current_model.parameters(), lr=LEARNING_RATE)

        # Train this model instance
        for epoch in range(EPOCHS_PER_GENERATION):
             _, _, _ = train_one_epoch(
                 current_model, train_loader_gen, optimizer, criterion, device, NUM_CLASSES
             )

        # Evaluate the trained model
        val_loss, val_accuracy, val_auc = evaluate_model(
            current_model, val_loader, criterion, device, NUM_CLASSES
        )
        #OPTIONAL
        print(f"      Model {model_run + 1} Eval -> Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.2f}%, AUC: {val_auc:.4f}") # Verbose

        generation_model_states.append(copy.deepcopy(current_model.state_dict()))
        generation_val_accuracies.append(val_accuracy)
        generation_val_aucs.append(val_auc) # Store AUC (will be NaN if calculation failed)

        del current_model # Clean up memory
        if device.type == 'cuda': torch.cuda.empty_cache()


    # Check if any models were successfully trained and evaluated
    if not generation_val_accuracies:
        print(f"Warning: No models were successfully evaluated in Generation {current_gen_number}. Skipping selection and checkpointing.")
        continue # Skip to the next generation

    # 4. Select the Best Model from the Generation (based on Val Accuracy)
    best_model_index_this_gen = np.argmax(generation_val_accuracies)
    best_accuracy_this_gen = generation_val_accuracies[best_model_index_this_gen]
    # Handle potential NaN in AUCs when finding the best AUC
    valid_aucs = [auc for auc in generation_val_aucs if auc is not None and not np.isnan(auc)]
    best_auc_this_gen = generation_val_aucs[best_model_index_this_gen] if not np.isnan(generation_val_aucs[best_model_index_this_gen]) else (max(valid_aucs) if valid_aucs else np.nan)

    best_model_state_dict = generation_model_states[best_model_index_this_gen] # Prepare for next gen or final save

    # Append history (only append if this gen ran successfully)
    all_best_accuracies.append(best_accuracy_this_gen)
    all_best_aucs.append(best_auc_this_gen) # Append the best AUC (might be NaN)

    print(f"--- Best model in Gen {current_gen_number}: Model {best_model_index_this_gen + 1} (Val Acc: {best_accuracy_this_gen:.2f}%, Val AUC: {best_auc_this_gen:.4f}) ---")

    # --- 5. Save Checkpoint to Google Drive ---
    print(f"  Saving checkpoint for completed Generation {current_gen_number}...")
    checkpoint_data = {
        'generation': generation, # Save the index of the *completed* generation
        'model_state_dict': best_model_state_dict,
        'all_best_accuracies': all_best_accuracies,
        'all_best_aucs': all_best_aucs,
        # Add any other state you want to save, e.g., LEARNING_RATE if it changes
        'config': { # Example of saving config used
            'num_classes': NUM_CLASSES,
            'img_height': IMG_HEIGHT,
            'img_width': IMG_WIDTH,
            'learning_rate': LEARNING_RATE
        }
    }
    try:
        # Save directly to the Drive path
        torch.save(checkpoint_data, CHECKPOINT_PATH)
        print(f"  Checkpoint successfully saved to: {CHECKPOINT_PATH}")
    except Exception as e:
        print(f"  Error saving checkpoint to Google Drive: {e}")
        # Consider alternative saving or logging the error


# --- 7. Final Steps after ALL Generations Complete ---
print("\n--- Generational Training Complete ---")

# Ensure best_model_state_dict holds the state from the very last successful generation
if best_model_state_dict is not None:
    final_model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    try:
        final_model.load_state_dict(best_model_state_dict)
        print("Final model loaded from the best state found across all trained generations.")
        print("Best validation accuracies per generation:", [f"{acc:.2f}%" for acc in all_best_accuracies])
        print("Corresponding best validation AUCs:", [f"{auc:.4f}" if not np.isnan(auc) else "N/A" for auc in all_best_aucs])

        # --- Evaluate final model on Test Set ---
        print("\nEvaluating final model on the TEST set...")
        test_loss, test_accuracy, test_auc = evaluate_model(final_model, test_loader, criterion, device, NUM_CLASSES)
        print(f"Final Model Performance - Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%, Test AUC: {test_auc:.4f}")

        # --- History Collection & Plotting for the final model --- << UNCOMMENTED SECTION
        print("\nCollecting training history for the final model...")

        # Reload the best state dict into a fresh model instance for clean history collection
        history_model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
        history_model.load_state_dict(best_model_state_dict) # Load the final best state
        history_optimizer = optim.Adam(history_model.parameters(), lr=LEARNING_RATE) # New optimizer

        # Use the same data loader setup as the *last* generation for consistency
        final_gen_seed = None
        if GENERATION_DATA_SEED is not None:
            final_gen_seed = GENERATION_DATA_SEED + (NUM_GENERATIONS - 1) # Seed used in last gen
        history_train_loader = create_generation_train_loader(
            train_dataset, BATCH_SIZE, generation_seed=final_gen_seed
        )

        # Store history metrics epoch by epoch
        history = {
            'train_loss': [], 'train_acc': [], 'train_auc': [],
            'val_loss': [], 'val_acc': [], 'val_auc': []
        }

        # Run for the same number of epochs as within a generation to get comparable history length
        num_history_epochs = EPOCHS_PER_GENERATION
        print(f"Collecting history over {num_history_epochs} epochs...")

        for epoch in range(num_history_epochs):
            # Train one epoch and get metrics
            train_loss, train_acc, train_auc = train_one_epoch(
                history_model, history_train_loader, history_optimizer, criterion, device, NUM_CLASSES
            )
            # Evaluate on validation set and get metrics
            val_loss, val_acc, val_auc = evaluate_model(
                history_model, val_loader, criterion, device, NUM_CLASSES
            )

            # Record metrics for plotting
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_auc'].append(train_auc) # Will be NaN if calculation failed
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_auc'].append(val_auc) # Will be NaN if calculation failed

            print(f"  History Epoch {epoch+1}/{num_history_epochs}: "
                  f"Train Loss={train_loss:.4f}, Acc={train_acc:.2f}%, AUC={train_auc if not np.isnan(train_auc) else 'N/A':.4f} | "
                  f"Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%, AUC={val_auc if not np.isnan(val_auc) else 'N/A':.4f}")

        # Call the plotting function << UNCOMMENTED LINE
        print("\nPlotting training history of the final model...")
        plot_history(history, num_history_epochs)
        # --- End of uncommented plotting section ---


        # --- Optional: Save the final model separately ---
        final_model_save_path = os.path.join(CHECKPOINT_DIR, 'final_generational_model_complete.pth')
        print(f"\nSaving final trained model to: {final_model_save_path}")
        try:
            torch.save(final_model.state_dict(), final_model_save_path)
            print("Final model saved successfully.")
        except Exception as e:
            print(f"Error saving final model: {e}")

    except Exception as e:
         print(f"Error loading final model state dict: {e}")
         print("Cannot perform final evaluation or saving.")

else:
    print("No valid model state available after training loop (possibly no generations were run or training failed).")


print("\n--- Script Finished ---")

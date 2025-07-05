import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import os
import random
import copy  # <-- Import copy for deep copying the model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures reproducibility

# Define a neural network model
class SimpleNN(nn.Module):
    def __init__(self, num_layers, hidden_size, dim_x=1, dim_y=1, activation=nn.SiLU()):   # num_layers: number of hidden layers
        super(SimpleNN, self).__init__()
        layers = [nn.Linear(dim_x, hidden_size)]
        if activation is not None:
            layers.append(activation)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if activation is not None:
                layers.append(activation)
        layers.append(nn.Linear(hidden_size, dim_y))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class EarlyStopping:
    def __init__(self, patience=50, fraction=0.0005):
        """
        Early stopping with a relative threshold.

        Parameters:
        - patience (int): Number of epochs to wait for improvement.
        - fraction (float): Minimum percentage decrease required to reset patience.
        """
        self.patience = patience
        self.fraction = fraction
        self.best_loss = float("inf")
        self.wait = 0

    def step(self, val_loss):
        """Check if training should stop."""
        if val_loss < self.best_loss * (1 - self.fraction):  # ✅ Relative improvement
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1

        return self.wait >= self.patience

def find_max_batch_size(model, dataset, device, start=32, step=2):
    """
    Dynamically find the largest batch size that fits in memory.

    Parameters:
    - model: PyTorch model
    - dataset: Dataset to test
    - device: "cuda" or "cpu"
    - start: Initial batch size to test
    - step: Factor to increase batch size (default: double each step)

    Returns:
    - Largest batch size that fits in memory
    """
    batch_size = start
    best_batch = batch_size
    max_batch = len(dataset)  # Limit to dataset size

    print(f"🔹 Starting batch size search on {device}...")

    # ✅ **1st Attempt: Try the largest possible batch size**
    try:
        loader = DataLoader(dataset, batch_size=max_batch)
        x, y = next(iter(loader))
        x, y = x.to(device), y.to(device)
        model.to(device)(x)  # Check if model can process batch
        print(f"✅ Batch Size {max_batch} fits in memory on {device}.")
        return max_batch
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "memory" in str(e).lower():
            print(f"❌ Batch Size {max_batch} is too large, reducing...")
        else:
            print(f"⚠ Unexpected error: {e}")

    # ✅ **2nd Attempt: Find the largest batch size incrementally**
    while batch_size <= max_batch:
        try:
            # Create DataLoader with current batch size
            loader = DataLoader(dataset, batch_size=batch_size)
            
            # Try multiple batches to ensure stability
            for _ in range(3):  
                x, y = next(iter(loader))
                x, y = x.to(device), y.to(device)
                model.to(device)(x)  # Check model compatibility

            print(f"✅ Batch Size {batch_size} fits in memory on {device}. Trying larger size...")
            best_batch = batch_size
            batch_size *= step  # Increase batch size

        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "memory" in str(e).lower():
                print(f"❌ Batch Size {batch_size} is too large, stopping search.")
                break  # Stop increasing when OOM occurs
            else:
                print(f"⚠ Unexpected error: {e}")
                break  # Stop on unknown errors

    print(f"🎯 Optimal Batch Size Found: {best_batch}")
    return best_batch

def train_NN(num_layers, hidden_size, train_x, train_y, val_x=None, val_y=None, decay=0, epochs=1000, lr=0.1, device='cuda', save_model=False, model_path='model.pth', activation=nn.SiLU(), lgk=None, zero_centering=False, L2_reg=True, initial_model=None, random_seed=42, mean_std=None, train_loss_lower=0):
    set_seed(random_seed) # Set seed for reproducibility

    center_x = None
    center_y = None
    if zero_centering: # x and y
        center_x = train_x.mean(dim=0, keepdim=True)
        train_x = train_x - center_x

        # check if x includes y_LF 
        # if train_x.shape[1] > train_y.shape[1]:
        #     center_y = center_x[:, -train_y.shape[1]:]
        # else:
        #     center_y = train_y.mean(dim=0, keepdim=True)
        center_y = train_y.mean(dim=0, keepdim=True)

        train_y = train_y - center_y

        if val_x is not None and val_y is not None:
            val_x = val_x - center_x
            val_y = val_y - center_y

    # lgk is not used for training, but saved for later use

    if initial_model is not None:
        model = initial_model
    else:
        # Create the model with the given hyperparameters
        model = SimpleNN(num_layers=num_layers, hidden_size=hidden_size, dim_x=train_x.shape[1], dim_y=train_y.shape[1], activation=activation).to(device)

    criterion = nn.MSELoss()
    penalty = 0

    def penalty():
        if L2_reg:
            l2_norm = sum(torch.sum(param ** 2) for param in model.parameters())
            penalty = decay * l2_norm
            return penalty
        else:
            return 0

        
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0 if L2_reg else decay)  # Use weight decay for L2 regularization


    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=60)

    # **Check if full-batch training is possible**
    # use_full_batch = best_batch >= len(train_dataset)
    use_full_batch = True
    best_batch = len(train_x) # enforce full-batch training for now (small dataset)

    # if not device yet, move to device
    if str(device)[:3] != str(train_x.device)[:3] and val_x is not None and val_y is not None:
        val_x, val_y = val_x.to(device), val_y.to(device)

    if use_full_batch:
        print(f"🔹 Using full-batch training (batch_size={best_batch})")
        # Convert to PyTorch tensors if not already
        if str(train_x.device)[:3] != str(device)[:3]:  # uss [:3] because device is cuda:0 or cuda when using GPU
            print("Converting to device tensors...")
            train_x, train_y = train_x.to(device), train_y.to(device)
    else:
        print(f"🔹 Using mini-batch training (batch_size={best_batch})")
        # Create PyTorch dataset
        train_x, train_y = train_x.cpu(), train_y.cpu()  # ✅ Ensure CPU tensors for DataLoader
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=best_batch, shuffle=True, num_workers=2, pin_memory=True)

    # Usage in training loop
    early_stopping = EarlyStopping(patience=300)

    # Training loop with mini-batches
    for epoch in range(epochs):
        model.train()
        
        if use_full_batch:
            optimizer.zero_grad()
            y_pred = model(train_x)
            train_loss = criterion(y_pred, train_y)
            loss = train_loss + penalty()
            loss.backward()
            optimizer.step()
        else:
            for batch_x, batch_y in train_loader:  # Loop over mini-batches
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                y_pred = model(batch_x)
                loss_batch = criterion(y_pred, batch_y) + penalty()
                loss_batch.backward()
                optimizer.step()

        # Training and Validation losses
        model.eval()
        with torch.no_grad():
            if not use_full_batch:
                train_pred = model(train_x.to(device))
                train_loss = criterion(train_pred, train_y.to(device)).item()
            else:
                train_loss = train_loss.item()
            if val_x is not None and val_y is not None:
                val_pred = model(val_x)
                val_loss = criterion(val_pred, val_y).item()
            else:
                val_loss = train_loss

        scheduler.step(val_loss+loss.item())

        # Check early stopping condition
        if early_stopping.step(val_loss+loss.item()) or train_loss < train_loss_lower:
           # sum of train and val loss (should be more stable; if the training loss is lower than the provided lower bound, stop training to avoid overfitting
            print(f"Stopping early at epoch {epoch}")
            print(f"Epoch {epoch}, Train Loss: {train_loss:.6e}, Val Loss: {val_loss:.6e}, Train loss with L2: {loss.item():.6e}, LR: {optimizer.param_groups[0]['lr']:.6e}")
            break

        if epoch==0 or (epoch+1) % 100 == 0 or epoch == epochs-1:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.6e}, Val Loss: {val_loss:.6e}, Train loss with L2: {loss.item():.6e}, LR: {optimizer.param_groups[0]['lr']:.6e}")

        # if epoch reached the maximum number of epochs, warn the user that the model is not converged
        if epoch == epochs - 1:
            print("⚠ Maximum number of epochs reached. The model may not have converged.\n")
    # print('type of pca_components:', type(pca_components))
    if save_model:
        torch.save({
            'num_layers': num_layers,
            'hidden_size': hidden_size,
            'activation': activation.__class__.__name__ if activation is not None else 'None', # save activation function as string
            'decay': decay,
            'lgk': lgk,
            'training_loss': train_loss,
            'center_x': center_x.cpu().numpy() if center_x is not None else None,  # Ensure it's a NumPy array
            'center_y': center_y.cpu().numpy() if center_y is not None else None,  # Convert before saving
            'state_dict': model.state_dict(),
            'mean_std': mean_std,
        }, model_path)
        print(f"Model saved to {model_path}\n")

    return train_loss, val_loss, model, optimizer.param_groups[0]['lr'], loss.item()  # Return the loss with L2 regularization

def train_model_kfold_2r_old(num_layers, hidden_size, x_data, y_data, decay=0, k=5, epochs=None, 
                      epochs_neuron=10, lr=0.1, model_dir='./', save_kf_model=False, 
                      device='cuda', shuffle=False, activation=nn.SiLU(), zero_centering=False, 
                      lgk=None, test_folds=None, num_trials=1, mean_std=None, trials_k1=None):
    """
    Train model using K-Fold Cross-Validation with an option to specify test folds.

    Parameters:
        num_layers: int - Number of layers in the NN.
        hidden_size: int - Number of neurons per layer.
        x_data: np.array - Input data.
        y_data: np.array - Target data.
        decay: float - Weight decay.
        k: int - Number of folds.
        epochs: int or None - Number of training epochs.
        epochs_neuron: int - Epochs per neuron.
        lr: float - Learning rate.
        model_dir: str - Directory to save models.
        save_kf_model: bool - Whether to save models.
        device: str - Device for computation ('cuda' or 'cpu').
        shuffle: bool - Shuffle data before splitting.
        activation: nn.Module - Activation function.
        zero_centering: bool - Whether to zero-center data.
        lgk: any - Additional parameter.
        test_folds: list or None - List of fold indices to test. If None, all folds are tested.

    Returns:
        tuple: (avg_train_loss, avg_val_loss)
    """
    epochs = epochs if epochs is not None else epochs_neuron * hidden_size * num_layers
    kf = KFold(n_splits=k, shuffle=True, random_state=42) if shuffle else KFold(n_splits=k)
    fold_results = []

    if test_folds is None:
        test_folds = list(range(k))  # Use all folds if not specified

    total_folds_to_test = len(test_folds)  # Total number of folds to test
    tested_count = 0  # Counter for completed folds

    # first round of training: independent training for each fold

    # exclude test folds from training x_data and y_data
    # x_data and y_data are PyTorch tensors
    mask = torch.ones(len(x_data), dtype=torch.bool)  # Create a mask of all True values
    mask[test_folds] = False  # Set test fold indices to False

    # Apply the mask to exclude the points we want to test against
    inds = np.arange(len(x_data))
    inds_1 = inds[mask]  # Indices of the points we will use in the first round of training
    x_data_1 = x_data[mask]
    y_data_1 = y_data[mask]

    # print excluded folds
    print(f"🔹 Excluded the {total_folds_to_test} target test points from the first round of training 🔹")

    # the number of folds in the first round of training
    k1 = k - len(test_folds)
    kf_1 = KFold(n_splits=k1, shuffle=True, random_state=42) if shuffle else KFold(n_splits=k1)

    trials_k1 = trials_k1 if trials_k1 is not None else total_folds_to_test

    print(f"🔹 Starting Round 1 of K-Fold Training: the first {trials_k1} folds from the {k1} folds will be used 🔹")

    for fold, (train_idx, val_idx) in enumerate(kf_1.split(x_data_1)):
        if tested_count == trials_k1: # for now, only test the first few folds
            break

        tested_count += 1
        print(f"🔹 Fold {tested_count}/{trials_k1}: Testing fold index {fold}/{k1-1} (point {inds_1[fold]}/{k-1}) 🔹")

        train_x, train_y = x_data_1[train_idx], y_data_1[train_idx]
        val_x, val_y = x_data_1[val_idx], y_data_1[val_idx]
        
        train_loss, val_loss, model, lr_fine, _ = train_fold_multiple_times(num_layers, hidden_size, train_x, train_y, val_x, val_y,
                 num_trials=num_trials, decay=decay, epochs=epochs, lr=lr, device=device, 
                 activation=activation, zero_centering=zero_centering)
        
        fold_results.append((train_loss, val_loss, model, lr_fine))
    # find the best model fold index
    idx_best = np.argmin([train_loss + val_loss for train_loss, val_loss, _, _ in fold_results])
    # best_model = fold_results[idx_best][2]
    best_model = copy.deepcopy(fold_results[idx_best][2])
    lr_best = fold_results[idx_best][3]

    # print the best model fold
    print(f"\n✅ Best Model Selected: model fold {idx_best}")

    fold_results = []  # Reset fold results for second round
    tested_count = 0  # Reset tested count for second round

    # second round of training: retrain using the best model's weights
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_data)):
        if fold not in test_folds:
            continue  # Skip unselected folds

        tested_count += 1
        print(f"🔹 Fold {tested_count}/{total_folds_to_test}: Testing fold index {fold}/{k-1} 🔹")

        train_x, train_y = x_data[train_idx], y_data[train_idx]
        val_x, val_y = x_data[val_idx], y_data[val_idx]

        kf_model_path = os.path.join(model_dir, f"model_fold{fold}.pth")

        train_loss, val_loss, model, _, _ = train_NN(num_layers, hidden_size, train_x, train_y, val_x, val_y, 
                                        decay=decay, epochs=epochs, lr=lr_best, device=device, 
                                        save_model=save_kf_model, model_path=kf_model_path, 
                                        activation=activation, zero_centering=zero_centering, lgk=lgk, initial_model=copy.deepcopy(best_model), mean_std=mean_std)

        fold_results.append((train_loss, val_loss, model))
    
    # best model
    best_model = min(fold_results, key=lambda x: x[0] + x[1])[2]

    if fold_results:
        avg_val_loss = np.mean([val_loss for _, val_loss, _ in fold_results])
        avg_train_loss = np.mean([train_loss for train_loss, _, _ in fold_results])
        print(f"✅ Average Loss Across Selected Folds: training: {avg_train_loss:.6e}, validation: {avg_val_loss:.6e}, mean(training,validation): {.5*(avg_train_loss+avg_val_loss):.6e}\n")
        return avg_train_loss, avg_val_loss, best_model, lr_best
    else:
        print("⚠️ No folds were selected for testing. Returning None.")
        return None, None, None, None

# update 2-round training: no k-fold training in the first round, but only multiple trials with distinct random seeds
# and then use the best model to train on the selected folds in the second round
def train_model_kfold_2r(num_layers, hidden_size, x_data, y_data, decay=0, k=5, epochs=None, 
                      epochs_neuron=10, lr=0.1, model_dir='./', save_kf_model=False, 
                      device='cuda', shuffle=False, activation=nn.SiLU(), zero_centering=False, 
                      lgk=None, test_folds=None, num_trials=None, mean_std=None):
    """
    Train model using K-Fold Cross-Validation with an option to specify test folds.

    Parameters:
        num_layers: int - Number of layers in the NN.
        hidden_size: int - Number of neurons per layer.
        x_data: np.array - Input data.
        y_data: np.array - Target data.
        decay: float - Weight decay.
        k: int - Number of folds.
        epochs: int or None - Number of training epochs.
        epochs_neuron: int - Epochs per neuron.
        lr: float - Learning rate.
        model_dir: str - Directory to save models.
        save_kf_model: bool - Whether to save models.
        device: str - Device for computation ('cuda' or 'cpu').
        shuffle: bool - Shuffle data before splitting.
        activation: nn.Module - Activation function.
        zero_centering: bool - Whether to zero-center data.
        lgk: any - Additional parameter.
        test_folds: list or None - List of fold indices to test. If None, all folds are tested.

    Returns:
        tuple: (avg_train_loss, avg_val_loss)
    """

    epochs = epochs if epochs is not None else epochs_neuron * hidden_size * num_layers
    kf = KFold(n_splits=k, shuffle=True, random_state=42) if shuffle else KFold(n_splits=k)

    if test_folds is None:
        test_folds = list(range(k))  # Use all folds if not specified

    if num_trials is None:
        print("⚠️ No trials specified. Setting num_trials to the number of test points by default.")
        num_trials = len(test_folds)  # Number of trials in the first round

    total_folds_to_test = len(test_folds)  # Total number of folds to test
    tested_count = 0  # Counter for completed folds

    # first round of training: train on the data excluding the test points

    mask = torch.ones(len(x_data), dtype=torch.bool)  # Create a mask of all True values
    mask[test_folds] = False  # Set test fold indices to False

    # Apply the mask to exclude the points we want to test against
    inds = np.arange(len(x_data))
    # inds_1 = inds[mask]  # Indices of the points we will use in the first round of training
    x_data_1 = x_data[mask]
    y_data_1 = y_data[mask]

    # use the excluded test points as the validation set in the first round
    x_data_1_val = x_data[test_folds]
    y_data_1_val = y_data[test_folds]

    # print training and validation data
    print(f"🔹 Excluded the {total_folds_to_test} target test points from training and test on them  🔹")

    print(f"🔹 Starting Round 1 of Training: searching for a good minimum 🔹")
    # no real validation loss here, just training
    train_loss, val_loss, model, lr_fine, _ = train_fold_multiple_times(num_layers, hidden_size, x_data_1, y_data_1, x_data_1_val, y_data_1_val,
                 num_trials=num_trials, decay=decay, epochs=epochs, lr=lr, device=device, 
                 activation=activation, zero_centering=zero_centering)
        
    # best_model = fold_results[idx_best][2]
    best_model = copy.deepcopy(model)
    lr_best = lr_fine

    fold_results = []  # Reset fold results for second round
    tested_count = 0  # Reset tested count for second round

    # second round of training: retrain using the best model's weights
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_data)):
        if fold not in test_folds:
            continue  # Skip unselected folds

        tested_count += 1
        print(f"🔹 Fold {tested_count}/{total_folds_to_test}: Testing fold index {fold}/{k-1} 🔹")

        train_x, train_y = x_data[train_idx], y_data[train_idx]
        val_x, val_y = x_data[val_idx], y_data[val_idx]

        kf_model_path = os.path.join(model_dir, f"model_fold{fold}.pth")

        train_loss, val_loss, model, _, reg_loss = train_NN(num_layers, hidden_size, train_x, train_y, val_x, val_y, 
                                        decay=decay, epochs=epochs, lr=lr_best, device=device, 
                                        save_model=save_kf_model, model_path=kf_model_path, 
                                        activation=activation, zero_centering=zero_centering, lgk=lgk, initial_model=copy.deepcopy(best_model), mean_std=mean_std)

        fold_results.append((train_loss, val_loss, model, reg_loss))
    
    # best model
    # best_model = min(fold_results, key=lambda x: x[0] + x[1])[2]

    # print the best model fold
    # idx_best = np.argmin([train_loss + val_loss for train_loss, val_loss, _, _ in fold_results])
    # should select the best model fold based on regularized loss instead of train_loss + val_loss, because individual val loss is highly dependent on the tested point, reg loss is more stable
    # choose the model with the regularized loss that is closest to the mean of the regularized loss
    del_loss = np.abs(np.array([reg_loss for _, _, _, reg_loss in fold_results]) - np.mean([reg_loss for _, _, _, reg_loss in fold_results]))
    idx_best = np.argmin(del_loss)
    best_model = copy.deepcopy(fold_results[idx_best][2])
    print(f"\n✅ Best Model Selected: model fold {test_folds[idx_best]} (with regularized loss closest to the mean)")

    if fold_results:
        avg_val_loss = np.mean([val_loss for _, val_loss, _, _ in fold_results])
        avg_train_loss = np.mean([train_loss for train_loss, _, _, _ in fold_results])
        print(f"✅ Average Loss Across Selected Folds: training: {avg_train_loss:.6e}, validation: {avg_val_loss:.6e}, mean(training,validation): {.5*(avg_train_loss+avg_val_loss):.6e}\n")
        return avg_train_loss, avg_val_loss, best_model, lr_best
    else:
        print("⚠️ No folds were selected for testing. Returning None.")
        return None, None, None, None
    

def train_fold_multiple_times(num_layers, hidden_size, train_x, train_y, val_x=None, val_y=None, 
                              num_trials=3, decay=0, epochs=1000, lr=0.1, device='cuda', 
                              activation=nn.SiLU(), zero_centering=False, save_model=False,
                              model_path='model.pth', lgk=None, mean_std=None):
    """
    Train a single fold multiple times with different seeds and return the best model.
    
    Parameters:
        num_trials: int - Number of times to train with different random seeds.
        
    Returns:
        best_model - The best-performing model for this fold.
    """
    val_provided = True  # Check if validation data is provided
    if val_x is None or val_y is None:
        val_provided = False
        print(f"⚠️ No validation data provided. Regularized loss will be used to select the best model.")

    best_model = None
    best_summed_loss = float("inf")

    for trial in range(num_trials):
        seed = 42 + trial  # Change seed for each trial
        
        print(f"🔄 Training fold with seed {seed} (Trial {trial+1}/{num_trials})...")
        
        train_loss, val_loss, model, lr_fine, reg_loss = train_NN(num_layers, hidden_size, train_x, train_y, val_x, val_y, 
                                                  decay=decay, epochs=epochs, lr=lr, device=device, 
                                                  activation=activation, zero_centering=zero_centering, random_seed=seed)

        # use regularized loss when no separate validation set is provided
        summed_loss = val_loss + train_loss if val_provided else reg_loss
            
        if summed_loss < best_summed_loss:
            best_model = model
            best_summed_loss = summed_loss
            best_train_loss = train_loss
            best_val_loss = val_loss
            best_reg_loss = reg_loss
            lr_best = lr_fine
            seed_best = seed
            # print(f"✅ Best model selected for this fold (Validation Loss + Training Loss: {best_summed_loss:.6e})")
    #retrain and save the best model

    print(f"✅ Best model selected mean(Validation Loss,Training Loss): {(best_train_loss+best_val_loss)/2:.6e}")
    if save_model and best_model is not None:
        print(f"🔄 Retraining the best model with seed {seed_best}... (usually leads to a slightly better model)")
        # retrain and save the best model
        train_NN(num_layers, hidden_size, train_x, train_y, val_x, val_y,
                 decay=decay, epochs=epochs, lr=lr_best, device=device, 
                 activation=activation, zero_centering=zero_centering, random_seed=seed_best, save_model=save_model, model_path=model_path, lgk=lgk, initial_model=best_model, mean_std=mean_std)

    return best_train_loss, best_val_loss, best_model, lr_best, best_reg_loss  # Return the loss with L2 regularization

def train_model_kfold(num_layers, hidden_size, x_data, y_data, decay=0, k=5, epochs=None, 
                      epochs_neuron=10, lr=0.1, model_dir='./', save_kf_model=False, 
                      device='cuda', shuffle=True, activation=nn.SiLU(), zero_centering=False, 
                      lgk=None, test_folds=None, num_trials=1, mean_std=None, trials_k1=None):  # trials_k1 is not used here
    """
    Train model using K-Fold Cross-Validation with an option to specify test folds.

    Parameters:
        num_layers: int - Number of layers in the NN.
        hidden_size: int - Number of neurons per layer.
        x_data: np.array - Input data.
        y_data: np.array - Target data.
        decay: float - Weight decay.
        k: int - Number of folds.
        epochs: int or None - Number of training epochs.
        epochs_neuron: int - Epochs per neuron.
        lr: float - Learning rate.
        model_dir: str - Directory to save models.
        save_kf_model: bool - Whether to save models.
        device: str - Device for computation ('cuda' or 'cpu').
        shuffle: bool - Shuffle data before splitting.
        activation: nn.Module - Activation function.
        zero_centering: bool - Whether to zero-center data.
        lgk: any - Additional parameter.
        test_folds: list or None - List of fold indices to test. If None, all folds are tested.

    Returns:
        tuple: (avg_train_loss, avg_val_loss)
    """
    epochs = epochs if epochs is not None else epochs_neuron * hidden_size * num_layers
    kf = KFold(n_splits=k, shuffle=True, random_state=42) if shuffle else KFold(n_splits=k)
    fold_results = []

    if test_folds is None:
        test_folds = list(range(k))  # Use all folds if not specified

    total_folds_to_test = len(test_folds)  # Total number of folds to test
    tested_count = 0  # Counter for completed folds


    for fold, (train_idx, val_idx) in enumerate(kf.split(x_data)):
        if fold not in test_folds:
            continue  # Skip unselected folds

        tested_count += 1
        print(f"🔹 Fold {tested_count}/{total_folds_to_test}: Testing fold index {fold}/{k-1} 🔹")

        train_x, train_y = x_data[train_idx], y_data[train_idx]
        val_x, val_y = x_data[val_idx], y_data[val_idx]

        kf_model_path = os.path.join(model_dir, f"model_fold{fold}.pth")

        train_loss, val_loss, model, lr_fine, reg_loss = train_fold_multiple_times(num_layers, hidden_size, train_x, train_y, val_x, val_y,
                                                                decay=decay, epochs=epochs, lr=lr, device=device, 
                                                                activation=activation, zero_centering=zero_centering, num_trials=num_trials, save_model=save_kf_model, model_path=kf_model_path, lgk=lgk, mean_std=mean_std)

        fold_results.append((train_loss, val_loss, model, lr_fine, reg_loss))
    # find the best model fold index based on regularized loss
    # idx_best = np.argmin([reg_loss for _, _, _, _, reg_loss in fold_results])
    del_loss = np.abs(np.array([reg_loss for _, _, _, _, reg_loss in fold_results]) - np.mean([reg_loss for _, _, _, _, reg_loss in fold_results]))
    idx_best = np.argmin(del_loss)
    # best_model = fold_results[idx_best][2]
    best_model = copy.deepcopy(fold_results[idx_best][2])
    lr_best = fold_results[idx_best][3]

    # print the best model fold
    print(f"\n✅ Best Model Selected: model fold {idx_best} (with regularized loss closest to the mean)")

    if fold_results:
        avg_val_loss = np.mean([val_loss for _, val_loss, _, _, _ in fold_results])
        avg_train_loss = np.mean([train_loss for train_loss, _, _, _, _ in fold_results])
        print(f"✅ Average Loss Across Selected Folds: training: {avg_train_loss:.6e}, validation: {avg_val_loss:.6e}, mean(training,validation): {.5*(avg_train_loss+avg_val_loss):.6e}\n")
        return avg_train_loss, avg_val_loss, best_model, lr_best
    else:
        print("⚠️ No folds were selected for testing. Returning None.")
        return None, None, None, None
    
def train_model_kfold_with_initial(num_layers, hidden_size, x_data, y_data, decay=0, k=5, epochs=None, 
                      epochs_neuron=10, lr=0.1, model_dir='./', save_kf_model=False, 
                      device='cuda', shuffle=True, activation=nn.SiLU(), zero_centering=False, 
                      lgk=None, test_folds=None, initial_model=None):

    epochs = epochs if epochs is not None else epochs_neuron * hidden_size * num_layers
    kf = KFold(n_splits=k, shuffle=True, random_state=42) if shuffle else KFold(n_splits=k)
    fold_results = []

    if test_folds is None:
        test_folds = list(range(k))  # Use all folds if not specified

    total_folds_to_test = len(test_folds)  # Total number of folds to test
    tested_count = 0  # Counter for completed folds


    for fold, (train_idx, val_idx) in enumerate(kf.split(x_data)):
        if fold not in test_folds:
            continue  # Skip unselected folds

        tested_count += 1
        print(f"🔹 Fold {tested_count}/{total_folds_to_test}: Testing fold index {fold}/{k-1} 🔹")

        train_x, train_y = x_data[train_idx], y_data[train_idx]
        val_x, val_y = x_data[val_idx], y_data[val_idx]

        kf_model_path = os.path.join(model_dir, f"model_fold{fold}.pth")

        train_loss, val_loss, model, lr_fine, _ = train_NN(num_layers, hidden_size, train_x, train_y, val_x, val_y, 
                                        decay=decay, epochs=epochs, lr=lr, device=device, 
                                        save_model=save_kf_model, model_path=kf_model_path, 
                                        activation=activation, zero_centering=zero_centering, lgk=lgk, initial_model=initial_model)

        fold_results.append((train_loss, val_loss, model, lr_fine))
    # find the best model fold index
    idx_best = np.argmin([train_loss + val_loss for train_loss, val_loss, _, _ in fold_results])
    # best_model = fold_results[idx_best][2]
    best_model = copy.deepcopy(fold_results[idx_best][2])
    lr_best = fold_results[idx_best][3]

    # print the best model fold
    print(f"\n✅ Best Model Selected: model fold {test_folds[idx_best]}")

    if fold_results:
        avg_val_loss = np.mean([val_loss for _, val_loss, _, _ in fold_results])
        avg_train_loss = np.mean([train_loss for train_loss, _, _, _ in fold_results])
        print(f"✅ Average Loss Across Selected Folds: training: {avg_train_loss:.6e}, validation: {avg_val_loss:.6e}, mean(training,validation): {.5*(avg_train_loss+avg_val_loss):.6e}\n")
        return avg_train_loss, avg_val_loss, best_model, lr_best
    else:
        print("⚠️ No folds were selected for testing. Returning None.")
        return None, None, None, None
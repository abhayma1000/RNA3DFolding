# %%
import general_utils

import time

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import random
import pickle
import yaml

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

start_time = time.time()

import GPUtil

GPUs = GPUtil.getGPUs()
if GPUs:
    gpu = GPUs[0]
    print(f"Running on: {gpu.name}")
else:
    print("Running on CPU")

# %% [markdown]
# # 1. CONFIG & SEED

# %%
def set_seed(seed: int):
    """Set a random seed for Python, NumPy, PyTorch (CPU & GPU) to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Example configuration (you can load this from a YAML, JSON, etc.)
config = {
    "seed": 42,
    "cutoff_date": "2020-01-01",
    "test_cutoff_date": "2022-05-01",
    "max_len": 384,
    "batch_size": 1,
    "model_config_path": "ribonanzanet2d-final/configs/pairwise.yaml",
    "max_len_filter": 9999999,
    "min_len_filter": 10,
    
    "ribonanzanet2d-final_path": "ribonanzanet2d-final",
    "train_sequences_path": "stanford-rna-3d-folding/train_sequences.csv",
    "train_labels_path": "stanford-rna-3d-folding/train_labels.csv",
    "test_sequences_path": "stanford-rna-3d-folding/test_sequences.csv",
    "pretrained_weights_path": "ribonanzanet-weights/RibonanzaNet.pt",
    "save_weights_folder": "trained_model_weights",
    "save_weights_name": "RibonanzaNet-3D.pt",
    "save_weights_final": "RibonanzaNet-3D-final.pt",
}

if not os.path.exists(config['save_weights_folder']):
    os.mkdir(config['save_weights_folder'])

# Set the seed for reproducibility
set_seed(config["seed"])

# %% [markdown]
# # 2. DATA LOADING & PREPARATION

# %%
# Load CSVs
train_sequences = pd.read_csv(config["train_sequences_path"])
train_labels = pd.read_csv(config["train_labels_path"])

test_sequences = pd.read_csv(config["test_sequences_path"])

# Create a pdb_id field
train_labels["pdb_id"] = train_labels["ID"].apply(
    lambda x: x.split("_")[0] + "_" + x.split("_")[1]
)

# Collect xyz data for each sequence
all_xyz = []
for pdb_id in tqdm(train_sequences["target_id"], desc="Collecting XYZ data"):
    df = train_labels[train_labels["pdb_id"] == pdb_id]
    xyz = df[["x_1", "y_1", "z_1"]].to_numpy().astype("float32")
    xyz[xyz < -1e17] = float("nan")
    all_xyz.append(xyz)

# %% [markdown]
# # 3. DATA FILTERING

# %%
valid_indices = []
max_len_seen = 0

for i, xyz in enumerate(all_xyz):
    # Track the maximum length
    if len(xyz) > max_len_seen:
        max_len_seen = len(xyz)

    nan_ratio = np.isnan(xyz).mean()
    seq_len = len(xyz)
    # Keep sequence if it meets criteria
    if (nan_ratio <= 0.5) and (config["min_len_filter"] < seq_len < config["max_len_filter"]):
        valid_indices.append(i)

print(f"Longest sequence in train: {max_len_seen}")

# Filter sequences & xyz based on valid_indices
train_sequences = train_sequences.loc[valid_indices].reset_index(drop=True)
all_xyz = [all_xyz[i] for i in valid_indices]

# Prepare final data dictionary
data = {
    "sequence": train_sequences["sequence"].tolist(),
    "temporal_cutoff": train_sequences["temporal_cutoff"].tolist(),
    "description": train_sequences["description"].tolist(),
    "all_sequences": train_sequences["all_sequences"].tolist(),
    "xyz": all_xyz,
}

test_data = {
    "sequence": test_sequences["sequence"].tolist(),
    "target_id": test_sequences["target_id"].tolist(),
}

# %% [markdown]
# # 4. TRAIN / VAL SPLIT

# %%
cutoff_date = pd.Timestamp(config["cutoff_date"])
test_cutoff_date = pd.Timestamp(config["test_cutoff_date"])

train_indices = [i for i, date_str in enumerate(data["temporal_cutoff"]) if pd.Timestamp(date_str) <= cutoff_date]
val_indices = [i for i, date_str in enumerate(data["temporal_cutoff"]) if cutoff_date < pd.Timestamp(date_str) <= test_cutoff_date]

# %% [markdown]
# # 5. DATASET & DATALOADER

# %%
class RNA3D_Dataset(Dataset):
    """
    A PyTorch Dataset for 3D RNA structures.
    """
    def __init__(self, indices, data_dict, max_len=384):
        self.indices = indices
        self.data = data_dict
        self.max_len = max_len
        self.nt_to_idx = {nt: i for i, nt in enumerate("ACGU")}

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        # Convert nucleotides to integer tokens
        sequence = [self.nt_to_idx[nt] for nt in self.data["sequence"][data_idx]]
        sequence = torch.tensor(sequence, dtype=torch.long)
        # Convert xyz to torch tensor
        xyz = torch.tensor(self.data["xyz"][data_idx], dtype=torch.float32)

        # If sequence is longer than max_len, randomly crop
        if len(sequence) > self.max_len:
            crop_start = np.random.randint(len(sequence) - self.max_len)
            crop_end = crop_start + self.max_len
            sequence = sequence[crop_start:crop_end]
            xyz = xyz[crop_start:crop_end]

        return {"sequence": sequence, "xyz": xyz}

class RNA3D_Dataset_Test(Dataset):
    """
    A PyTorch Dataset for 3D RNA structures.
    """
    def __init__(self, data_dict, max_len=384):
        self.data = data_dict
        self.max_len = max_len
        self.nt_to_idx = {nt: i for i, nt in enumerate("ACGU")}

    def __len__(self):
        return len(self.data["sequence"])
    
    def __getitem__(self, idx):
        # Convert nucleotides to integer tokens
        sequence = [self.nt_to_idx[nt] if nt in self.nt_to_idx else 4 for nt in self.data["sequence"][idx]]
        sequence = torch.tensor(sequence, dtype=torch.long)
        # Convert xyz to torch tensor

        # Dont crop and return the full sequence

        return {"sequence": sequence, "target_id": self.data["target_id"][idx]}


train_dataset = RNA3D_Dataset(train_indices, data, max_len=config["max_len"])
val_dataset = RNA3D_Dataset(val_indices, data, max_len=config["max_len"])
test_dataset = RNA3D_Dataset_Test(test_data, max_len=config["max_len"])

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# %% [markdown]
# # 6. MODEL, CONFIG CLASSES & HELPER FUNCTIONS

# %%
sys.path.append(config["ribonanzanet2d-final_path"])

from Network import RibonanzaNet

class Config:
    """Simple Config class that can load from a dict or YAML."""
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries = entries

    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return Config(**cfg)

class FinetunedRibonanzaNet(RibonanzaNet):
    """
    A finetuned version of RibonanzaNet adapted for predicting 3D coordinates.
    """
    def __init__(self, config_obj, pretrained=False, dropout=0.1):
        # Modify config dropout before super init, if needed
        config_obj.dropout = dropout
        super(FinetunedRibonanzaNet, self).__init__(config_obj)

        # Load pretrained weights if requested
        if pretrained:
            self.load_state_dict(
                torch.load(config["pretrained_weights_path"], map_location="cpu")
            )

        self.dropout = nn.Dropout(p=0.0)
        self.xyz_predictor = nn.Linear(256, 3)

    def forward(self, src):
        """Forward pass to predict 3D XYZ coordinates."""
        # get_embeddings returns (sequence_features, *some_other_outputs)
        sequence_features, _ = self.get_embeddings(
            src, torch.ones_like(src).long().to(src.device)
        )
        xyz_pred = self.xyz_predictor(sequence_features)
        return xyz_pred

# Instantiate the model
model_cfg = load_config_from_yaml(config["model_config_path"])
model = FinetunedRibonanzaNet(model_cfg, pretrained=True).cuda()

# %% [markdown]
# # 7. LOSS FUNCTIONS

# %%
def calculate_distance_matrix(X, Y, epsilon=1e-4):
    """
    Calculate pairwise distances between every point in X and every point in Y.
    Shape: (len(X), len(Y))
    """
    return ((X[:, None] - Y[None, :])**2 + epsilon).sum(dim=-1).sqrt()

def dRMSD(pred_x, pred_y, gt_x, gt_y, epsilon=1e-4, Z=10, d_clamp=None):
    """
    Distance-based RMSD.
    pred_x, pred_y: predicted coordinates (usually the same tensor for X and Y).
    gt_x, gt_y: ground truth coordinates.
    """
    pred_dm = calculate_distance_matrix(pred_x, pred_y)
    gt_dm = calculate_distance_matrix(gt_x, gt_y)

    mask = ~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0], device=mask.device).bool()] = False

    diff_sq = (pred_dm[mask] - gt_dm[mask])**2 + epsilon
    if d_clamp is not None:
        diff_sq = diff_sq.clamp(max=d_clamp**2)

    return diff_sq.sqrt().mean() / Z

def local_dRMSD(pred_x, pred_y, gt_x, gt_y, epsilon=1e-4, Z=10, d_clamp=30):
    """
    Local distance-based RMSD, ignoring distances above a clamp threshold.
    """
    pred_dm = calculate_distance_matrix(pred_x, pred_y)
    gt_dm = calculate_distance_matrix(gt_x, gt_y)

    mask = (~torch.isnan(gt_dm)) & (gt_dm < d_clamp)
    mask[torch.eye(mask.shape[0], device=mask.device).bool()] = False

    diff_sq = (pred_dm[mask] - gt_dm[mask])**2 + epsilon
    return diff_sq.sqrt().mean() / Z

def dRMAE(pred_x, pred_y, gt_x, gt_y, epsilon=1e-4, Z=10):
    """
    Distance-based Mean Absolute Error.
    """
    pred_dm = calculate_distance_matrix(pred_x, pred_y)
    gt_dm = calculate_distance_matrix(gt_x, gt_y)

    mask = ~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0], device=mask.device).bool()] = False

    diff = torch.abs(pred_dm[mask] - gt_dm[mask])
    return diff.mean() / Z

def align_svd_mae(input_coords, target_coords, Z=10):
    """
    Align input_coords to target_coords via SVD (Kabsch algorithm) and compute MAE.
    """
    assert input_coords.shape == target_coords.shape, "Input and target must have the same shape"

    # Create mask for valid points
    mask = ~torch.isnan(target_coords.sum(dim=-1))
    input_coords = input_coords[mask]
    target_coords = target_coords[mask]
    
    # Compute centroids
    centroid_input = input_coords.mean(dim=0, keepdim=True)
    centroid_target = target_coords.mean(dim=0, keepdim=True)

    # Center the points
    input_centered = input_coords - centroid_input
    target_centered = target_coords - centroid_target

    # Compute covariance matrix
    cov_matrix = input_centered.T @ target_centered

    # SVD to find optimal rotation
    U, S, Vt = torch.svd(cov_matrix)
    R = Vt @ U.T

    # Ensure a proper rotation (determinant R == 1)
    if torch.det(R) < 0:
        Vt_adj = Vt.clone()   # Clone to avoid in-place modification issues
        Vt_adj[-1, :] = -Vt_adj[-1, :]
        R = Vt_adj @ U.T

    # Rotate input and compute mean absolute error
    aligned_input = (input_centered @ R.T) + centroid_target
    return torch.abs(aligned_input - target_coords).mean() / Z

# %% [markdown]
# # 8. TRAINING LOOP

# %%
def train_model(model, train_dl, val_dl, epochs=50, cos_epoch=35, lr=3e-4, clip=1):
    """Train the model with a CosineAnnealingLR after `cos_epoch` epochs."""
    best_model_path = general_utils.get_next_filename(os.path.join(config['save_weights_folder'], config["save_weights_name"]))

    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.0, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(epochs - cos_epoch) * len(train_dl),
    )

    best_val_loss = float("inf")
    best_preds = None

    for epoch in range(epochs):
        model.train()
        train_pbar = tqdm(train_dl, desc=f"Training Epoch {epoch+1}/{epochs}")
        running_loss = 0.0

        for idx, batch in enumerate(train_pbar):
            sequence = batch["sequence"].cuda()
            gt_xyz = batch["xyz"].squeeze().cuda()

            pred_xyz = model(sequence).squeeze()

            # Combine two distance-based losses
            loss = dRMAE(pred_xyz, pred_xyz, gt_xyz, gt_xyz) + align_svd_mae(pred_xyz, gt_xyz)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()

            if (epoch + 1) > cos_epoch:
                scheduler.step()

            running_loss += loss.item()
            avg_loss = running_loss / (idx + 1)
            train_pbar.set_description(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        with torch.no_grad():
            for idx, batch in enumerate(val_dl):
                sequence = batch["sequence"].cuda()
                gt_xyz = batch["xyz"].squeeze().cuda()

                pred_xyz = model(sequence).squeeze()
                loss = dRMAE(pred_xyz, pred_xyz, gt_xyz, gt_xyz)
                val_loss += loss.item()

                val_preds.append((gt_xyz.cpu().numpy(), pred_xyz.cpu().numpy()))

            val_loss /= len(val_dl)
            print(f"Validation Loss (Epoch {epoch+1}): {val_loss:.4f}")

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_preds = val_preds
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> New best model saved to {best_model_path} at epoch {epoch+1}")

    # Save final model
    torch.save(model.state_dict(), general_utils.get_next_filename(os.path.join(config['save_weights_folder'], config["save_weights_final"])))
    return best_val_loss, best_preds, best_model_path

# %% [markdown]
# # 9. RUN TRAINING

# %%
best_loss, best_predictions, best_model_path = train_model(
    model=model,
    train_dl=train_loader,
    val_dl=val_loader,
    epochs=80,         # or config["epochs"]
    cos_epoch=35,      # or config["cos_epoch"]
    lr=3e-4,
    clip=1
)
print(f"Best Validation Loss: {best_loss:.4f}")

# %%
general_utils.send_email("itam xmqh ngut jhle", "Training complete", f"Model saved to {best_model_path} after {general_utils.get_time_from_start(start_time)}")

# %% [markdown]
# # 10. LOAD MODEL AND EVALUATE

# %%
loaded_model = FinetunedRibonanzaNet(model_cfg, pretrained=True).cuda()
loaded_model.load_state_dict(
    torch.load(best_model_path, map_location="cpu")
)

# %%
val_to_monomer = {i: nt for i, nt in enumerate("ACGU")}
def vals_to_monomers(val):
    return [val_to_monomer[i] for i in val.tolist()]

cols = ["ID","resname","resid","x_1","y_1","z_1","x_2","y_2","z_2","x_3","y_3","z_3","x_4","y_4","z_4","x_5","y_5","z_5"]
preds_pd = pd.DataFrame(columns=cols)

with torch.no_grad():
    loaded_model.eval()

    test_pbar = tqdm(test_loader, desc=f"Generating Predictions")

    test_preds = []
    for idx, batch in enumerate(test_pbar):
        sequence = batch["sequence"].cuda()
        res_names = vals_to_monomers(sequence.squeeze())
        res_ids = [i + 1 for i in range(len(res_names))]
        target_id = batch["target_id"]


        pred_xyz = loaded_model(sequence).squeeze().cpu().numpy()

        new_row = {
            "ID": [batch['target_id'][0] + "_" + str(i) for i in res_ids],
            "resname": res_names,
            "resid": res_ids,
            "x_1": pred_xyz[:, 0],
            "y_1": pred_xyz[:, 1],
            "z_1": pred_xyz[:, 2],
        }
        for i in range(2, 6):
            new_row[f"x_{i}"] = 0.0
            new_row[f"y_{i}"] = 0.0
            new_row[f"z_{i}"] = 0.0
        
        preds_pd = pd.concat([preds_pd, pd.DataFrame(new_row)], ignore_index=True)

preds_pd.to_csv("submission.csv", index=False)
print("Predictions saved to submission.csv")

# %%


# %%




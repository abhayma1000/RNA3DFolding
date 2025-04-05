# %% [markdown]
# # Purpose
# 
# * Take  what I learned in ```ribonanza-3d-finetune-v2.ipynb``` and go further with it
# * Make some improvements, specially to null handling
# * Expand the model to do better and use newer technologies like transfomers
# * Better evaluation metrics
# 

# %% [markdown]
# # Imports

# %%
import general_utils
import utils

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

    # change to kaggle
    "model_config_path": "ribonanzanet2d-final/configs/pairwise.yaml",
    
    "max_len_filter": 9999999,
    "min_len_filter": 10,
    
    # change to kaggle
    "ribonanzanet2d-final_path": "ribonanzanet2d-final",
    "train_sequences_path": "stanford-rna-3d-folding/train_sequences.csv",
    "train_labels_path": "stanford-rna-3d-folding/train_labels.csv",
    "test_sequences_path": "stanford-rna-3d-folding/test_sequences.csv",
    "pretrained_weights_path": "ribonanzanet-weights/RibonanzaNet.pt",


    "save_weights_folder": "trained_model_weights",
    "save_weights_name": "RibonanzaNet-3D.pt",
    "save_weights_final": "RibonanzaNet-3D-final.pt",

    "epochs": 80,
    "cos_epoch": 35,
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
    "target_id": train_sequences["target_id"].tolist(),
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

        return {"sequence": sequence, "xyz": xyz, "target_id": self.data['target_id'][data_idx]}

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

        self.embedding_dim = 256
        self.num_heads = 4
        self.ff_dim = 512
        self.num_layers = 3

        encoder_layer2 = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            dropout=dropout,
            batch_first=True)

        self.transformer_encoder2 = nn.TransformerEncoder(
            encoder_layer2,
            num_layers = self.num_layers)

        self.xyz_predictor = nn.Linear(self.embedding_dim, 3)

    def forward(self, src):
        """Forward pass to predict 3D XYZ coordinates."""
        # get_embeddings returns (sequence_features, *some_other_outputs)
        sequence_features, _ = self.get_embeddings(
            src, torch.ones_like(src).long().to(src.device)
        )
        transformed = self.transformer_encoder2(sequence_features)
        xyz_pred = self.xyz_predictor(transformed)
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

def calculate_tm_score_exact(pred_coords, true_coords):
    """
    Takes in np arrays
    https://www.kaggle.com/code/fernandosr85/rna-3d-fold-hybrid-template-nn-structure#Phase-2:-Quality-Assessment-Model
    Implementation more closely matching US-align with sequence-independent alignment.
    Includes multiple rotation schemes to find the optimal structural alignment.
    """

    mask = ~np.isnan(np.sum(true_coords, axis=-1))
    pred_coords = pred_coords[mask]
    true_coords = true_coords[mask]
    
    Lref = len(true_coords)
    if Lref < 3:
        return 0.0
    
    # Define d0 exactly as in the evaluation formula
    if Lref >= 30:
        d0 = 0.6 * np.sqrt(Lref - 0.5) - 2.5
    elif Lref >= 24:
        d0 = 0.7
    elif Lref >= 20:
        d0 = 0.6
    elif Lref >= 16:
        d0 = 0.5
    elif Lref >= 12:
        d0 = 0.4
    else:
        d0 = 0.3
    
    # Normalize structures
    pred_centered = pred_coords - np.mean(pred_coords, axis=0)
    true_centered = true_coords - np.mean(true_coords, axis=0)
    
    # Try multiple fragment lengths for sequence-independent alignment
    # This mimics US-align's approach to find the best fragment alignment
    best_tm_score = 0.0
    fragment_lengths = [Lref, max(5, Lref//2), max(5, Lref//4)]
    
    for frag_len in fragment_lengths:
        # Try different fragment start positions
        for i in range(0, Lref - frag_len + 1, max(1, frag_len//2)):
            pred_frag = pred_centered[i:i+frag_len]
            
            # Try aligning with different parts of the true structure
            for j in range(0, Lref - frag_len + 1, max(1, frag_len//2)):
                true_frag = true_centered[j:j+frag_len]
                
                # Covariance matrix for optimal rotation
                covariance = np.dot(pred_frag.T, true_frag)
                U, S, Vt = np.linalg.svd(covariance)
                rotation = np.dot(U, Vt)
                
                # Try different rotation schemes - this is the new part
                rotations_to_try = [
                    rotation,  # Original rotation from SVD
                    np.dot(rotation, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])),  # 90 degree Z rotation
                    np.dot(rotation, np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))  # 180 degree Z rotation
                ]
                
                for rot in rotations_to_try:
                    # Apply rotation to the full structure
                    pred_aligned = np.dot(pred_centered, rot)
                    
                    # Calculate distances
                    distances = np.sqrt(np.sum((pred_aligned - true_centered) ** 2, axis=1))
                    
                    # Calculate TM-score terms
                    tm_terms = 1.0 / (1.0 + (distances / d0) ** 2)
                    tm_score = np.sum(tm_terms) / Lref
                    
                    best_tm_score = max(best_tm_score, tm_score)
    
    return float(best_tm_score)

# %% [markdown]
# # 8. TRAINING LOOP

# %%
def train_model(model, train_dl, val_dl, epochs=50, cos_epoch=35, lr=3e-4, clip=1):
    """Train the model with a CosineAnnealingLR after `cos_epoch` epochs."""
    best_model_path, _ = general_utils.get_next_filename(os.path.join(config['save_weights_folder'], config["save_weights_name"]))

    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.0, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(epochs - cos_epoch) * len(train_dl),
    )

    best_val_loss = float("inf")
    best_tm_score = 0.0
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
        val_tm_score = 0.0
        val_preds = []
        with torch.no_grad():
            for idx, batch in enumerate(val_dl):
                sequence = batch["sequence"].cuda()
                gt_xyz = batch["xyz"].squeeze().cuda()

                pred_xyz = model(sequence).squeeze()
                loss = dRMAE(pred_xyz, pred_xyz, gt_xyz, gt_xyz)
                val_loss += loss.item()
                val_tm_score += calculate_tm_score_exact(pred_xyz.cpu().numpy(), gt_xyz.cpu().numpy())

                val_preds.append((gt_xyz.cpu().numpy(), pred_xyz.cpu().numpy()))

            val_loss /= len(val_dl)
            val_tm_score /= len(val_dl)
            print(f"Validation Loss (Epoch {epoch+1}): {val_loss:.4f}")
            print(f"Validation TM Score (Epoch {epoch+1}): {val_tm_score:.4f}")

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_tm_score = val_tm_score
                best_preds = val_preds
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> New best model saved to {best_model_path} at epoch {epoch+1}")

    # Save final model
    final_save_folder, _ = general_utils.get_next_filename(os.path.join(config['save_weights_folder'], config["save_weights_final"]))
    torch.save(model.state_dict(), final_save_folder)
    return best_val_loss, best_tm_score, best_preds, best_model_path

# %% [markdown]
# # 9. RUN TRAINING

# %%
best_loss, best_tm_score, best_predictions, best_model_path_name = train_model(
    model=model,
    train_dl=train_loader,
    val_dl=val_loader,
    # epochs=1,         # or config["epochs"]
    # cos_epoch=1,      # or config["cos_epoch"]
    epochs=config['epochs'],
    cos_epoch=config['cos_epoch'],
    lr=3e-4,
    clip=1
)
print(f"Best Validation Loss: {best_loss:.4f}")
print(f"Best TM Score: {best_tm_score:.4f}")

# %% [markdown]
# # 9.5 Potentially load in a model

# %%
# # model_load_path = best_model_path
# model_load_path = "trained_model_weights/RibonanzaNet-3D_2.pt"
# model = FinetunedRibonanzaNet(model_cfg, pretrained=True).cuda()
# model.load_state_dict(
#     torch.load(model_load_path, map_location="cpu")
# )

# display(model)

# %% [markdown]
# # 10. All Val Data Testing

# %%


all_drmae = []
all_tm_scores = []
all_svd_mae = []

with torch.no_grad():
    model.eval()

    for this_data in tqdm(val_loader, desc="Evaluating on Val dataset"):
        xyz_tensor = this_data['xyz'].squeeze().cuda()
        xyz = xyz_tensor.cpu().numpy()
        sequences = np.array(utils.vals_to_monomers(this_data['sequence'].numpy().squeeze())).astype(object)

        
        actual = {
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
            "sequences": sequences,
            "name": f"{this_data['target_id']}-actual"}
    
    
        pred_sequence = this_data['sequence'].cuda()
        pred_xyz_tensor = model(pred_sequence).squeeze()
        pred_xyz = pred_xyz_tensor.cpu().numpy()

    
        new_pred_xyz = utils.align_structures(pred_xyz, xyz)
        cropped_sequences = sequences[:new_pred_xyz.shape[0]]

        # print(pred_xyz.shape)
        # print(type(pred_xyz))
        # print(sequences.shape)
        # print(pred_sequence.shape)


    
        predicted = {
            "x": new_pred_xyz[:, 0],
            "y": new_pred_xyz[:, 1],
            "z": new_pred_xyz[:, 2],
            "sequences": cropped_sequences,
            "name": f"{this_data['target_id']}-predicted"}
    
        
        # utils.plot_multiple_structures([predicted, actual])

        # print(f"Predicted vs Actual for {this_data['target_id']}")
        drmae = dRMAE(pred_xyz_tensor, pred_xyz_tensor, xyz_tensor, xyz_tensor)
        svd_mae = align_svd_mae(pred_xyz_tensor, xyz_tensor)
        tm_score = calculate_tm_score_exact(pred_xyz, xyz)
        # print(f"drMAE: {drmae}")
        # print(f"align_svd_mae: {svd_mae}")
        # print(f"TM-score: {tm_score}")
        all_drmae.append(drmae.cpu().numpy())
        all_tm_scores.append(tm_score)
        all_svd_mae.append(svd_mae.cpu().numpy())


print("Average drMAE:", np.mean(np.array(all_drmae)))
print("Average TM-score:", np.mean(np.array(all_tm_scores)))
print("Average SVD MAE:", np.mean(np.array(all_svd_mae)))

# %% [markdown]
# # 11. One Val Data Testing (run multiple times)

# %%
with torch.no_grad():
    model.eval()

    this_data = random.choice(list(val_loader))
    xyz_tensor = this_data['xyz'].squeeze().cuda()
    xyz = xyz_tensor.cpu().numpy()
    sequences = np.array(utils.vals_to_monomers(this_data['sequence'].numpy().squeeze())).astype(object)

    
    actual = {
        "x": xyz[:, 0],
        "y": xyz[:, 1],
        "z": xyz[:, 2],
        "sequences": sequences,
        "name": f"{this_data['target_id']}-actual"}


    pred_sequence = this_data['sequence'].cuda()
    pred_xyz_tensor = model(pred_sequence).squeeze()
    pred_xyz = pred_xyz_tensor.cpu().numpy()


    new_pred_xyz = utils.align_structures(pred_xyz, xyz)
    cropped_sequences = sequences[:new_pred_xyz.shape[0]]

    # print(pred_xyz.shape)
    # print(new_pred_xyz.shape)
    # print(sequences.shape)
    # print(pred_sequence.shape)



    predicted = {
        "x": new_pred_xyz[:, 0],
        "y": new_pred_xyz[:, 1],
        "z": new_pred_xyz[:, 2],
        "sequences": cropped_sequences,
        "name": f"{this_data['target_id']}-predicted"}


    print(f"Predicted vs Actual for {this_data['target_id']}")
    print(f"drMAE: {dRMAE(pred_xyz_tensor, pred_xyz_tensor, xyz_tensor, xyz_tensor)}")
    print(f"align_svd_mae: {align_svd_mae(pred_xyz_tensor, xyz_tensor)}")
    print(f"TM-score: {calculate_tm_score_exact(pred_xyz, xyz)}")

    # utils.plot_multiple_structures([predicted, actual])
    

# %% [markdown]
# # Make prediction to submission.csv

# %%
cols = ["ID","resname","resid","x_1","y_1","z_1","x_2","y_2","z_2","x_3","y_3","z_3","x_4","y_4","z_4","x_5","y_5","z_5"]
preds_pd = pd.DataFrame(columns=cols)

with torch.no_grad():
    model.eval()

    test_pbar = tqdm(test_loader, desc=f"Generating Predictions")

    test_preds = []
    for idx, batch in enumerate(test_pbar):
        sequence = batch["sequence"].cuda()
        res_names = utils.vals_to_monomers(sequence.squeeze())
        res_ids = [i + 1 for i in range(len(res_names))]
        target_id = batch["target_id"]


        pred_xyz = model(sequence).squeeze().cpu().numpy()

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
general_utils.send_email("itam xmqh ngut jhle", "Training complete", f"Model saved to {best_model_path_name} after {general_utils.get_time_from_start(start_time)}")

# %%




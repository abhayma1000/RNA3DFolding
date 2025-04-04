import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

def plot_structure(x: np.array, 
                    y: np.array, 
                    z: np.array, 
                    sequences: np.array, 
                    name: str,
                    size=4,
                    lines=True,
                    linewidth=2) -> None:
    # Takes in the raw lists of x, y, z, and sequences and plots the 3D structure of the RNA
    # The sequences are colored by the nucleotide they represent


    # Expecting shapes x: (n, ), y: (n, ), z: (n, ), sequences: (n, )


    # https://www.kaggle.com/code/asarvazyan/interactive-3d-sequence-visualization
    colors = {"A": "red", "G": "blue", "C": "green", "U": "orange"}

    fig = go.Figure()

    for resname, color in colors.items():
        fig.add_trace(go.Scatter3d(
            x=x[sequences == resname],
            y=y[sequences == resname],
            z=z[sequences == resname],
            mode="markers",
            marker=dict(size=size, color=color),
            name=resname
        ))
    
    if lines:
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line=dict(color="black", width=linewidth),
            name="RNA Backbone"
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title_text="X"),
            yaxis=dict(title_text="Y"),
            zaxis=dict(title_text="Z"),
        ),
        title=f"RNA 3D Structure of {name}",
    )

    fig.show()

def plot_multiple_structures(structures: list[dict], size=4, lines=True, linewidth=2) -> None:
    """
    Takes a list of RNA structures and plots them in the same 3D plot.
    Each structure is represented as a dictionary with keys:
    - 'x': np.array of x-coordinates
    - 'y': np.array of y-coordinates
    - 'z': np.array of z-coordinates
    - 'sequences': np.array of nucleotide sequences
    - 'name': str, name of the structure
    """
    base_colors = {"A": "red", "G": "blue", "C": "green", "U": "orange"}
    fig = go.Figure()

    for idx, structure in enumerate(structures):
        x, y, z, sequences, name = structure['x'], structure['y'], structure['z'], structure['sequences'], structure['name']
        
        for resname, base_color in base_colors.items():
            # Adjust color shade based on structure index
            color = f"rgba({idx * 50 % 255}, {idx * 50 % 255}, {idx * 50 % 255}, 0.8)" if resname=="A" else base_color
            fig.add_trace(go.Scatter3d(
                x=x[sequences == resname],
                y=y[sequences == resname],
                z=z[sequences == resname],
                mode="markers",
                marker=dict(size=size, color=color),
                name=f"{name} - {resname}"
            ))
        
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line=dict(color="black", width=linewidth),
            name=f"{name} - Backbone"
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title_text="X"),
            yaxis=dict(title_text="Y"),
            zaxis=dict(title_text="Z"),
        ),
        title="RNA 3D Structures",
    )

    fig.show()


def align_structures(input_coords, target_coords):
    """
    Align input_coords to target_coords using Kabsch algorithm. Note: This is numpy version, not torch.
    Args:
        input_coords (np.array): Input coordinates of shape (N, 3).
        target_coords (np.array): Target coordinates of shape (N, 3).
    """
    assert input_coords.shape == target_coords.shape, "Input and target must have the same shape"

    # Create mask for valid points
    mask = ~np.isnan(np.sum(target_coords, axis=-1))
    input_coords = input_coords[mask]
    target_coords = target_coords[mask]
    
    # Compute centroids
    centroid_input = np.mean(input_coords, axis=0, keepdims=True)
    centroid_target = np.mean(target_coords, axis=0, keepdims=True)

    # Center the points
    input_centered = input_coords - centroid_input
    target_centered = target_coords - centroid_target

    # Compute covariance matrix
    cov_matrix = np.dot(input_centered.T, target_centered)

    # SVD to find optimal rotation
    U, S, Vt = np.linalg.svd(cov_matrix)
    R = np.dot(Vt.T, U.T)

    # Ensure a proper rotation (determinant R == 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Rotate input and compute aligned coordinates
    aligned_input = np.dot(input_centered, R.T) + centroid_target
    return aligned_input



# TODO look at
def calculate_tm_score_exact(pred_coords, true_coords):
    """
    https://www.kaggle.com/code/fernandosr85/rna-3d-fold-hybrid-template-nn-structure#Phase-2:-Quality-Assessment-Model
    Implementation more closely matching US-align with sequence-independent alignment.
    Includes multiple rotation schemes to find the optimal structural alignment.
    """
    # Remove padding
    mask = ~np.all(true_coords == 0, axis=1)
    pred = pred_coords[mask]
    true = true_coords[mask]
    
    Lref = len(true)
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
    pred_centered = pred - np.mean(pred, axis=0)
    true_centered = true - np.mean(true, axis=0)
    
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
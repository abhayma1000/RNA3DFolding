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

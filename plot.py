#!/usr/bin/env python
# coding: utf-8

"""
Created on Friday June 6 2025

@author: Pratik

Command-Line Arguments:
-----------------------
--t      : int   
    Time index for plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

# Argument parsing
parser = argparse.ArgumentParser(description='Plot precipitation data')
parser.add_argument('--t', type=int, default=1, help='Time index for plotting')
args = parser.parse_args()

# Plotting config
sns.set(style="whitegrid")
cmap = sns.color_palette("rocket", as_cmap=True)
plot_dir = "plots"

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Created folder: {plot_dir}")
else:
    print(f"Folder already exists: {plot_dir}")

# Load data
pred = np.load(f"pred/ConvLSTM-pred.npy")

# Extract persistence (t=3) and target (t=6)

true_output = np.load(f"pred/ConvLSTM-test.npy")

# Choose time step to plot
t_idx = args.t

# Function to plot a heatmap and save it
def plot_heatmap(data, title, filename, vmin=None, vmax=None):
    plt.figure(figsize=(2.3, 1.7))
    ax = sns.heatmap(data, cmap="viridis", cbar_kws={
        "orientation": "vertical", "shrink": 0.6, "aspect": 10})
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Longitude (deg)", fontsize=11)
    ax.set_ylabel("Latitude (deg)", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename), dpi=300)
    plt.close()

# Plot persistence image
plot_heatmap(
    pred[t_idx, :, :],
    title=f"Pred, T = {t_idx}",
    filename=f"{t_idx}-pred.png"
)

# Plot true output image
plot_heatmap(
    true_output[t_idx, :, :],
    title=f"True, T = {t_idx}",
    filename=f"{t_idx}-true.png"
)
print(f"Generated plots for time point {args.t}")
#!/usr/bin/env python3
"""
Minimal script to visualize XY model configurations using Wolff algorithm
at specific temperatures for L=20. Based on xy_model.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import logging
from tqdm import tqdm
import numba
from numba import njit, boolean, float64, int64, types

# --- Basic Configuration ---
L_VIS = 20
J_VIS = 1.0
TEMPERATURES_VIS = [0.92, 0.93]
THERMALIZE_SWEEPS_VIS = 4000
RUN_SWEEPS_VIS = 1000
OUTPUT_DIR_VIS = "visualization_results"
OUTPUT_FILENAME = f"xy_model_L{L_VIS}_visualization.png"

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Necessary JIT Functions (Copied from xy_model.py) ---

@njit
def compute_vorticity(spins, L):
    """
    Calculate the vorticity field for the current spin configuration.
    """
    vortices = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            theta1 = spins[i, j]
            theta2 = spins[i, (j + 1) % L]
            theta3 = spins[(i + 1) % L, (j + 1) % L]
            theta4 = spins[(i + 1) % L, j]
            dtheta1 = (theta2 - theta1 + np.pi) % (2 * np.pi) - np.pi
            dtheta2 = (theta3 - theta2 + np.pi) % (2 * np.pi) - np.pi
            dtheta3 = (theta4 - theta3 + np.pi) % (2 * np.pi) - np.pi
            dtheta4 = (theta1 - theta4 + np.pi) % (2 * np.pi) - np.pi
            winding = (dtheta1 + dtheta2 + dtheta3 + dtheta4) / (2 * np.pi)
            vortices[i, j] = np.round(winding)
    return vortices

@njit(types.void(float64[:,:], int64, float64, float64))
def wolff_sweep(spins, L, beta, J):
    """
    Perform one effective Wolff sweep (L*L cluster attempts).
    (Adapted reference implementation from xy_model.py)
    """
    num_sites = L * L
    in_cluster = np.zeros((L, L), dtype=boolean)
    cluster_stack = np.empty((L * L, 2), dtype=np.int64)

    for _ in range(num_sites):
        phi_reflect = np.random.uniform(0, 2.0 * np.pi)
        i0, j0 = np.random.randint(0, L), np.random.randint(0, L)
        in_cluster[:] = False
        stack_idx = 0
        head = 0
        cluster_stack[stack_idx] = (i0, j0)
        stack_idx += 1
        in_cluster[i0, j0] = True

        while head < stack_idx:
            i, j = cluster_stack[head]
            head += 1
            current_theta = spins[i, j]
            spins[i, j] = (2.0 * phi_reflect - current_theta) % (2.0 * np.pi)

            neighbors_coords = [
                ((i - 1 + L) % L, j), ((i + 1) % L, j),
                (i, (j - 1 + L) % L), (i, (j + 1) % L)
            ]

            for ni, nj in neighbors_coords:
                if not in_cluster[ni, nj]:
                    neighbor_theta = spins[ni, nj]
                    energy_diff = -J * (np.cos(current_theta - neighbor_theta) -
                                       np.cos(current_theta - (2.0 * phi_reflect - neighbor_theta)))
                    prob_arg = beta * energy_diff
                    freeze_prob = 0.0
                    if prob_arg < 0:
                        freeze_prob = 1.0 - np.exp(prob_arg)
                        freeze_prob = min(freeze_prob, 1.0)

                    if np.random.random() < freeze_prob:
                        in_cluster[ni, nj] = True
                        if stack_idx < num_sites:
                            cluster_stack[stack_idx] = (ni, nj)
                            stack_idx += 1
                        else: break
            if stack_idx >= num_sites and head < stack_idx: break

# --- Minimal XYModel Class ---

class XYModelVis:
    """Minimal XY model class for visualization."""
    def __init__(self, L: int, J: float = 1.0):
        self.L = L
        self.J = J
        self.spins = np.random.uniform(0, 2*np.pi, (L, L))

    def wolff_sweep(self, beta: float) -> None:
        """Perform one effective Wolff sweep."""
        wolff_sweep(self.spins, self.L, beta, self.J)

    def compute_vorticity(self) -> np.ndarray:
         """Calculate the vorticity field."""
         return compute_vorticity(self.spins, self.L)

# --- Main Visualization Logic ---

def run_and_visualize():
    """Runs simulations for specific temps and generates the plot."""

    if not os.path.exists(OUTPUT_DIR_VIS):
        os.makedirs(OUTPUT_DIR_VIS)
        logger.info(f"Created output directory: {OUTPUT_DIR_VIS}")

    final_states = {} # Dict to store {temp: model}

    logger.info(f"Generating configurations for L={L_VIS} at T={TEMPERATURES_VIS}")

    for temp in TEMPERATURES_VIS:
        logger.info(f"Simulating T = {temp:.4f}")
        model = XYModelVis(L=L_VIS, J=J_VIS)
        beta = 1.0 / temp

        # Thermalization
        logger.info(f"  Thermalizing for {THERMALIZE_SWEEPS_VIS} sweeps...")
        for _ in tqdm(range(THERMALIZE_SWEEPS_VIS), desc=f"  Therm (T={temp:.2f})", leave=False):
            model.wolff_sweep(beta)

        # Run a few more sweeps
        logger.info(f"  Running {RUN_SWEEPS_VIS} sweeps for final configuration...")
        for _ in tqdm(range(RUN_SWEEPS_VIS), desc=f"  Run   (T={temp:.2f})", leave=False):
             model.wolff_sweep(beta)

        final_states[temp] = model
        logger.info(f"  Configuration for T={temp:.4f} generated.")

        # --- Plotting (Inside the loop for individual plots) ---
        logger.info(f"Generating visualization plot for T={temp:.2f}...")
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5), dpi=150) # Create a new figure

        spins = model.spins
        L = model.L # Use L from the model instance

        # Plot spins
        x, y = np.meshgrid(np.arange(0, L), np.arange(0, L))
        u = np.cos(spins)
        v = np.sin(spins)
        ax.quiver(x, y, u, v, pivot='mid', scale=35, scale_units='width',
                  width=0.004, headwidth=3.5, headlength=4, headaxislength=3.5,
                  color='black')

        # --- Plot vortices/antivortices at plaquette centers ---
        vortices = model.compute_vorticity()
        vortex_coords_pos = np.argwhere(vortices > 0.5)
        vortex_coords_neg = np.argwhere(vortices < -0.5)

        # Plaquette center coordinates (x=j+0.5, y=i+0.5)
        if len(vortex_coords_pos) > 0:
            ax.scatter(vortex_coords_pos[:, 1] + 0.5, vortex_coords_pos[:, 0] + 0.5,
                       s=80, c='red', marker='o', alpha=0.6, label='Vortex (+1)', zorder=10)
        if len(vortex_coords_neg) > 0:
            ax.scatter(vortex_coords_neg[:, 1] + 0.5, vortex_coords_neg[:, 0] + 0.5,
                       s=80, c='blue', marker='o', alpha=0.6, label='Antivortex (-1)', zorder=10)

        ax.set_title(f'XY Model Spin Configuration\nL = {L_VIS}, T = {temp:.2f}')
        ax.set_xlim(-1, L) # Use L from model here
        ax.set_ylim(-1, L) # Use L from model here
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        # Add legend if any vortices were plotted
        handles, labels = ax.get_legend_handles_labels()
        if handles: # Check if lists are not empty
            # Remove duplicate labels if both +1 and -1 exist
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc='upper right')

        # Save individual plot
        output_filename_single = f"xy_model_L{L_VIS}_T{temp:.2f}_vis.png"
        save_path = os.path.join(OUTPUT_DIR_VIS, output_filename_single)
        try:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            logger.info(f"Visualization saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save visualization {save_path}: {e}")
        plt.close(fig) # Close the figure

    logger.info("Finished generating all individual visualizations.")


if __name__ == "__main__":
    run_and_visualize() 
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import os
import json

# Configuration
SIM_EXE = r"build\Release\RMDartSim.exe"
MC_EXE = r"build\Release\RMDartMC.exe"
NUM_SIMULATIONS = 10000
MAX_WORKERS = 8

# Target Definitions (Local NED coordinates)
# Based on distances and angles provided:
# Outpost: 15.865m, -6.5 deg
# Base: 25.233m, 7.3 deg
OUTPOST_DIST = 15.865
OUTPOST_AZIMUTH = -6.5
BASE_DIST = 25.233
BASE_AZIMUTH = 7.3

def get_target_pos(dist, azimuth_deg):
    az = np.radians(azimuth_deg)
    # X is North (Forward), Y is East (Right)
    return np.array([dist * np.cos(az), dist * np.sin(az), 0.0])

TARGETS = {
    "Outpost": get_target_pos(OUTPOST_DIST, OUTPOST_AZIMUTH),
    "Base": get_target_pos(BASE_DIST, BASE_AZIMUTH)
}

# Target Area Size (Armor plate approx 15cm x 15cm)
TARGET_RADIUS = 0.10 # 10cm radius for "hit"

# Error Distributions (Standard Deviations)
SIGMA_VEL = 0.3      # m/s (Initial velocity consistency)
SIGMA_PITCH = 0.2    # deg (Mechanical angle error)
SIGMA_YAW = 0.2      # deg (Mechanical angle error)

def run_single_sim(sim_id, v0, pitch, yaw, target_name):
    csv_name = f"mc_sim_{sim_id}.csv"
    # Note: RMDartSim needs to be modified to accept output filename or we rename it
    # For now, let's assume it always writes to rm_dart_trajectory.csv and we rename it
    # To avoid collisions in parallel, we'll run sequentially or use a lock, 
    # but better to modify RMDartSim to take an output path.
    # Let's quickly check if I can modify RMDartSim.
    pass

def analyze_monte_carlo_gpu(target_name):
    target_pos = TARGETS[target_name]
    print(f"\n--- GPU Monte Carlo Analysis for {target_name} ---")
    
    # Run GPU simulation
    # Note: RMDartMC currently has fixed targets in code, but we can pass num_sims
    # and it targets the Base (7.3 deg) by default.
    cmd = [MC_EXE, str(NUM_SIMULATIONS)]
    subprocess.run(cmd, check=True)
    
    # Read results
    df_res = pd.read_csv("rm_dart_mc_results.csv")
    
    # Ensure numeric types
    df_res['X'] = pd.to_numeric(df_res['X'], errors='coerce')
    df_res['Y'] = pd.to_numeric(df_res['Y'], errors='coerce')
    df_res.dropna(subset=['X', 'Y'], inplace=True)
    
    # Calculate errors relative to target
    df_res['err_x'] = df_res['X'] - target_pos[0]
    df_res['err_y'] = df_res['Y'] - target_pos[1]
    df_res['dist_err'] = np.sqrt(df_res['err_x']**2 + df_res['err_y']**2)
    
    hits = df_res[df_res['dist_err'] <= TARGET_RADIUS]
    hit_rate = len(hits) / len(df_res) * 100
    
    cep_50 = np.percentile(df_res['dist_err'], 50)
    cep_95 = np.percentile(df_res['dist_err'], 95)
    
    print(f"Results for {NUM_SIMULATIONS} samples:")
    print(f"Hit Rate (within {TARGET_RADIUS*100:.0f}cm): {hit_rate:.2f}%")
    print(f"CEP (50%): {cep_50:.3f} m")
    print(f"CEP (95%): {cep_95:.3f} m")
    
    # Visualization
    plt.figure(figsize=(10, 8))
    plt.scatter(df_res['err_y'], df_res['err_x'], alpha=0.1, s=1, color='blue', label='Impact Points')
    plt.scatter(0, 0, color='red', marker='X', s=100, label='Target Center')
    
    circle = plt.Circle((0, 0), TARGET_RADIUS, color='r', fill=False, linestyle='--', label='Armor Plate')
    plt.gca().add_patch(circle)
    
    plt.xlabel('Lateral Error (Y) [m]')
    plt.ylabel('Longitudinal Error (X) [m]')
    plt.title(f'GPU Monte Carlo: {target_name}\nSamples={NUM_SIMULATIONS}, Hit Rate={hit_rate:.2f}%')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    
    plt.savefig(f"mc_gpu_{target_name}.png")
    print(f"Plot saved to mc_gpu_{target_name}.png")

if __name__ == "__main__":
    if not os.path.exists(MC_EXE):
        print(f"Error: {MC_EXE} not found. Build the project first.")
    else:
        # Currently RMDartMC targets Base by default
        analyze_monte_carlo_gpu("Base")

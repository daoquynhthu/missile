import subprocess
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SIM_EXE = r"build\Release\DartSim.exe"

def run_comparison_sim(diameter_mm, name):
    # Update aero table for this diameter
    print(f"Evaluating {name} (Diameter: {diameter_mm}mm)...")
    
    # Temporarily modify generate_dart_aero_table.py to use this diameter
    with open("generate_dart_aero_table.py", "r") as f:
        lines = f.readlines()
    
    with open("generate_dart_aero_table.py", "w") as f:
        for line in lines:
            if line.startswith("BODY_RADIUS ="):
                f.write(f"BODY_RADIUS = {diameter_mm / 2000.0}\n")
            else:
                f.write(line)
    
    # Regenerate table
    subprocess.run(["python", "generate_dart_aero_table.py"], capture_output=True)
    
    # Run simulation at 25m/s, 5 deg pitch, 0 yaw
    output_file = f"traj_{name}.csv"
    cmd = [SIM_EXE, "25.0", "5.0", "0.0", output_file]
    subprocess.run(cmd, capture_output=True)
    
    if os.path.exists(output_file):
        return pd.read_csv(output_file)
    return None

if __name__ == "__main__":
    # 1. Compare "Chubby" (50mm) vs "Needle-ish" (25mm)
    df_chubby = run_comparison_sim(50, "Chubby")
    df_needle = run_comparison_sim(25, "Needle")
    
    if df_chubby is not None and df_needle is not None:
        plt.figure(figsize=(12, 6))
        
        # Velocity Plot
        plt.subplot(1, 2, 1)
        plt.plot(df_chubby["Time(s)"], df_chubby["Vel(m/s)"], label="Chubby (50mm)")
        plt.plot(df_needle["Time(s)"], df_needle["Vel(m/s)"], label="Needle (25mm)")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.title("Velocity Decay Comparison")
        plt.legend()
        plt.grid(True)
        
        # Trajectory Plot (X-Z)
        plt.subplot(1, 2, 2)
        plt.plot(df_chubby["X(m)"], df_chubby["Z(m)"], label="Chubby (50mm)")
        plt.plot(df_needle["X(m)"], df_needle["Z(m)"], label="Needle (25mm)")
        plt.xlabel("Distance (m)")
        plt.ylabel("Altitude (m)")
        plt.title("Trajectory Comparison (5 deg Pitch)")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("design_comparison.png")
        
        # Stats
        v_end_c = df_chubby.iloc[-1]["Vel(m/s)"]
        v_end_n = df_needle.iloc[-1]["Vel(m/s)"]
        dist_c = df_chubby.iloc[-1]["X(m)"]
        dist_n = df_needle.iloc[-1]["X(m)"]
        
        print("\n--- Rigorous Comparison Results ---")
        print(f"Chubby (50mm) -> End Vel: {v_end_c:.2f} m/s, Max Range: {dist_c:.2f} m")
        print(f"Needle (25mm) -> End Vel: {v_end_n:.2f} m/s, Max Range: {dist_n:.2f} m")
        print(f"Velocity retention improved by {((v_end_n - v_end_c) / v_end_c * 100):.1f}%")

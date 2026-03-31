import subprocess
import os
import re
import math
from concurrent.futures import ThreadPoolExecutor

# Optimization Script for Qian Xuesen Trajectory
# Goal: Find boost parameters that result in a flat trajectory at burnout (or shortly after)
# to ensure entry into skip-glide phase at ~50km altitude.

EXE_PATH = r"build\bin\Release\AeroSim.exe"

def run_simulation(boost_pitch_start, boost_pitch_rate, boost_pitch_min):
    """
    Run the simulation with given parameters and return the final state or specific metrics.
    We need to capture the output or read the CSV.
    Since the CSV is overwritten, we can't easily parallelize without unique filenames.
    For now, sequential execution or just rely on stdout if possible.
    Actually, let's just run sequentially.
    """
    cmd = [EXE_PATH, str(boost_pitch_start), str(boost_pitch_rate), str(boost_pitch_min)]
    
    try:
        # Run process
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300) # 5 min timeout
        output = result.stdout
        
        # Parse relevant data from stdout or we can read the CSV
        # Let's read the CSV for better accuracy
        if not os.path.exists("simulation_data.csv"):
            return float('inf'), {}

        data = []
        with open("simulation_data.csv", "r") as f:
            lines = f.readlines()
            header = lines[0].strip().split(',')
            for line in lines[1:]:
                if line.strip():
                    parts = line.strip().split(',')
                    row = {k: float(v) for k, v in zip(header, parts)}
                    data.append(row)
        
        if not data:
            return float('inf'), {}

        # Analyze Trajectory
        # Key Metrics:
        # 1. Apogee (Max Altitude)
        # 2. Burnout State (t=32s)
        # 3. Glide Entry (Phase transition to Glide)
        
        max_alt = 0
        burnout_alt = 0
        burnout_gamma = 0 # Flight path angle
        glide_start_idx = -1
        
        for i, row in enumerate(data):
            if row["Alt(m)"] > max_alt:
                max_alt = row["Alt(m)"]
            
            if abs(row["Time(s)"] - 32.0) < 0.1: # Approx burnout
                burnout_alt = row["Alt(m)"]
                # Calculate Gamma? We don't log Gamma directly but we have Vel components?
                # No, we have Vel magnitude and Pitch.
                # Pitch is body attitude. Gamma = Pitch - Alpha.
                # In boost phase, Alpha is small if we turn gravity.
                # Let's use vertical velocity if we can derive it?
                # Wait, we don't log vertical velocity directly.
                # But we can infer from Altitude change? Or just use Max Alt as proxy.
                pass
                
            if int(row["Phase"]) == 2 and glide_start_idx == -1: # Phase 2 is GLIDE (0=BOOST, 1=COAST, 2=GLIDE, 3=TERMINAL)
                # Wait, Phase enum in C++: BOOST=0, COAST=1, GLIDE=2, TERMINAL=3
                glide_start_idx = i

        # Objective Function:
        # 1. Apogee should be around 70-80km (Spacecraft/Missile boundary)
        # 2. Must enter Glide Phase
        
        apogee_target = 75000.0
        apogee_error = abs(max_alt - apogee_target)
        
        penalty = 0
        
        # Penalize excessive apogee heavily (we don't want a ballistic missile)
        if max_alt > 100000:
            penalty += (max_alt - 100000) * 2.0
            
        # Penalize too low apogee (we need height to glide)
        if max_alt < 50000:
            penalty += 1e6
            
        if not (glide_start_idx != -1):
            penalty += 1e7
            
        loss = apogee_error + penalty
        
        metrics = {
            "max_alt": max_alt,
            "burnout_alt": burnout_alt,
            "entered_glide": glide_start_idx != -1
        }
        
        return loss, metrics

    except Exception as e:
        print(f"Error: {e}")
        return float('inf'), {}

def optimize():
    # Grid Search / Random Search / Gradient Descent
    # Parameters:
    # 1. Start Time (0.5 - 10.0 s)
    # 2. Pitch Rate (1.0 - 5.0 deg/s)
    # 3. Min Pitch (-20 - 10 deg)
    
    pitch_start_range = [2.5]
    pitch_rate_range = [3.20, 3.25, 3.30, 3.35, 3.40]
    pitch_min_range = [0.0, -5.0]

    best_loss = float('inf')
    best_params = None
    best_metrics = None

    print(f"{'Start':<10} {'Rate':<10} {'Min':<10} {'Loss':<15} {'Apogee(km)':<15} {'Glide?':<10}")

    for start in pitch_start_range:
        for rate in pitch_rate_range:
            for min_p in pitch_min_range:
                loss, metrics = run_simulation(start, rate, min_p)
                print(f"{start:<10} {rate:<10} {min_p:<10} {loss:<15.1f} {metrics.get('max_alt', 0)/1000:<15.1f} {str(metrics.get('entered_glide', False)):<10}")

                if loss < best_loss:
                    best_loss = loss
                    best_params = (start, rate, min_p)
                    best_metrics = metrics
                    
    print("\nOptimization Complete.")
    print(f"Best Params: Start={best_params[0]}, Rate={best_params[1]}, Min={best_params[2]}")
    print(f"Best Loss: {best_loss}")
    print(f"Metrics: {best_metrics}")

if __name__ == "__main__":
    optimize()

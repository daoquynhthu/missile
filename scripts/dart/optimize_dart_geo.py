import numpy as np
import subprocess
import os
import sys
import json

# Optimization Target: Maximize Stability, Minimize Drag
# Parameters:
# 0: BODY_RADIUS (m)
# 1: FIN_SPAN (m)
# 2: FIN_CHORD_ROOT (m)
# 3: FIN_CHORD_TIP (m)
# 4: FIN_SWEEP_DEG (deg)

BOUNDS = [
    (0.010, 0.020), # Radius (20-40mm dia)
    (0.030, 0.070), # Span
    (0.030, 0.060), # Root Chord
    (0.010, 0.040), # Tip Chord
    (0.0, 45.0)     # Sweep
]

def update_aero_script(params):
    r, span, cr, ct, sweep = params
    with open("generate_dart_aero_table.py", "r") as f:
        lines = f.readlines()
    
    with open("generate_dart_aero_table.py", "w") as f:
        for line in lines:
            if line.startswith("BODY_RADIUS ="): f.write(f"BODY_RADIUS = {r}\n")
            elif line.startswith("FIN_SPAN ="): f.write(f"FIN_SPAN = {span}\n")
            elif line.startswith("FIN_CHORD_ROOT ="): f.write(f"FIN_CHORD_ROOT = {cr}\n")
            elif line.startswith("FIN_CHORD_TIP ="): f.write(f"FIN_CHORD_TIP = {ct}\n")
            elif line.startswith("FIN_SWEEP_DEG ="): f.write(f"FIN_SWEEP_DEG = {sweep}\n")
            else: f.write(line)

def evaluate(params):
    update_aero_script(params)
    
    # Run table generation
    subprocess.run(["python", "generate_dart_aero_table.py"], capture_output=True)
    
    # Read the table to get metrics at Mach 0.1, Alpha 5
    import pandas as pd
    df = pd.read_csv("dart_aero_table.csv")
    
    # Filter for Mach ~0.1, Alpha 5
    sample = df[(df['Mach'] == 0.1) & (df['Alpha'] == 5.0)].iloc[0]
    
    cd = sample['CD']
    cm = sample['Cm'] # Negative is stable
    
    # Static Margin Proxy: SM = -Cm / CL_alpha (approx)
    # We want Cm to be negative and large magnitude for stability
    # We want CD to be small
    
    # Constraints: Tip chord < Root chord
    if params[3] > params[2]: return 1000.0
    
    # Objective: Stability + Velocity
    # Stability: We want Cm < -0.1 (normalized)
    # Drag: We want CD < 0.5
    
    score = cd * 2.0 - cm * 5.0 # Minimize this
    
    # Penalty for unstable
    if cm >= 0: score += 500.0
    
    print(f"Params: R={params[0]*1000:.1f}mm, Span={params[1]*1000:.1f}mm, Sweep={params[4]:.1f} | CD={cd:.3f}, Cm={cm:.3f} | Score={score:.3f}")
    return score

def differential_evolution(pop_size=8, generations=10):
    # Simple DE implementation
    population = []
    for _ in range(pop_size):
        ind = [np.random.uniform(low, high) for low, high in BOUNDS]
        population.append(ind)
    
    scores = [evaluate(ind) for ind in population]
    
    for gen in range(generations):
        print(f"\n--- Generation {gen} ---")
        for i in range(pop_size):
            # Mutate
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = [population[idx] for idx in np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(np.array(a) + 0.8 * (np.array(b) - np.array(c)), 
                             [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
            
            # Crossover
            trial = [mutant[j] if np.random.rand() < 0.7 else population[i][j] for j in range(len(BOUNDS))]
            
            score = evaluate(trial)
            if score < scores[i]:
                scores[i] = score
                population[i] = trial
                
        best_idx = np.argmin(scores)
        print(f"Best Score so far: {scores[best_idx]:.4f}")
        
    return population[best_idx]

if __name__ == "__main__":
    best_params = differential_evolution()
    print("\n=== Optimization Finished ===")
    print(f"Best Radius: {best_params[0]*1000:.2f} mm")
    print(f"Best Fin Span: {best_params[1]*1000:.2f} mm")
    print(f"Best Root Chord: {best_params[2]*1000:.2f} mm")
    print(f"Best Tip Chord: {best_params[3]*1000:.2f} mm")
    print(f"Best Sweep: {best_params[4]:.2f} deg")
    
    # Final update
    update_aero_script(best_params)
    subprocess.run(["python", "generate_dart_aero_table.py"])
    print("Final Aero Table generated with optimized parameters.")

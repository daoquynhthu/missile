import numpy as np
import json
import subprocess
import os
import sys
import random
from generate_cone_waverider import generate_cone_waverider

# Configuration
AERO_CALC_EXE = r"e:\missile\build\bin\Release\AeroCalc.exe"
STL_PATH = "e:/missile/temp_opt_shape.stl"
BEST_STL_PATH = "e:/missile/hgv_model_optimized.stl"

# Design Point for Optimization
# Mach 15, Alpha 10 deg, Beta 0
# We want Max L/D at this condition for efficient glide.
DESIGN_MACH = 15.0
DESIGN_ALPHA = 10.0
DESIGN_BETA = 0.0

# Parameter Bounds
# Realistic HGV Dimensions for IRBM Payload (Mass ~1000-1500kg)
# 1. Body Length (m): [3.5, 5.0]
# 2. Body Width (m): [1.2, 2.2]
# 3. Design Shock Angle (deg): [10.0, 18.0] - Determines Volume/Drag
# 4. Planform Power (n): [0.6, 0.9] - 0.5=Parabolic, 1.0=Delta
BOUNDS = [
    (3.5, 5.0),   # Length
    (1.2, 2.2),   # Width
    (10.0, 18.0), # Shock Angle
    (0.6, 0.9)    # Planform Power
]

def evaluate_design(params):
    length, width, beta, power = params
    
    # Generate STL
    config = {
        'body_length': length,
        'body_width': width,
        'design_mach': DESIGN_MACH, # Optimize for cruise Mach
        'design_beta': beta,
        'planform_power': power,
        'stl_path': STL_PATH,
        'grid_density': 60
    }
    
    try:
        generate_cone_waverider(config)
    except Exception as e:
        return 1000.0 # Penalty
    
    # Run AeroCalc
    # We need reference area. For waverider, Planform Area is approx 2/3 * L * W (for power=0.5)
    # Let's use 1.0 for optimization metric (maximize raw L/D)
    # L/D is independent of reference area.
    
    cmd = [
        AERO_CALC_EXE,
        STL_PATH,
        str(DESIGN_MACH),
        str(DESIGN_ALPHA),
        str(DESIGN_BETA),
        "1.0", "1.0", "1.0"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Parse JSON output
        # Output format: {"CL": ..., "CD": ..., "L_D": ...}
        # Find JSON in stdout
        stdout = result.stdout
        start = stdout.find('{')
        end = stdout.rfind('}') + 1
        if start == -1 or end == 0:
            return 1000.0
            
        data = json.loads(stdout[start:end])
        l_d = data['L_D']
        
        # We want to MAXIMIZE L/D, so minimize NEGATIVE L/D
        return -l_d
    except Exception as e:
        return 1000.0

def optimize():
    print("Starting Shape Optimization (Differential Evolution)...")
    print(f"Target: Max L/D at Mach {DESIGN_MACH}, Alpha {DESIGN_ALPHA}")
    print("Parameters: Length, Width, Shock Angle, Planform Power")
    
    # Differential Evolution Parameters
    pop_size = 5
    mutation_factor = 0.8
    crossover_prob = 0.7
    generations = 3
    
    # Initialize Population with scores
    population = []
    scores = []
    print("Initializing population...")
    for _ in range(pop_size):
        ind = [random.uniform(b[0], b[1]) for b in BOUNDS]
        population.append(ind)
        scores.append(evaluate_design(ind))
        
    best_score = min(scores)
    best_ind = population[scores.index(best_score)]
    print(f"Initial Best L/D: {-best_score:.4f}")

    for gen in range(generations):
        print(f"Generation {gen+1}/{generations}")
        
        new_population = []
        new_scores = []
        
        for i in range(pop_size):
            # Mutation
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = random.sample(idxs, 3)
            
            mutant = []
            for j in range(len(BOUNDS)):
                # DE/rand/1/bin
                val = population[a][j] + mutation_factor * (population[b][j] - population[c][j])
                val = max(BOUNDS[j][0], min(BOUNDS[j][1], val)) # Clip
                mutant.append(val)
            
            # Crossover
            trial = []
            # Ensure at least one parameter is changed (not strictly required for DE but good practice)
            cross_points = [random.random() < crossover_prob for _ in range(len(BOUNDS))]
            if not any(cross_points):
                cross_points[random.randint(0, len(BOUNDS)-1)] = True
                
            for j in range(len(BOUNDS)):
                if cross_points[j]:
                    trial.append(mutant[j])
                else:
                    trial.append(population[i][j])
            
            # Selection
            score_trial = evaluate_design(trial)
            score_target = scores[i]
            
            if score_trial < score_target:
                new_population.append(trial)
                new_scores.append(score_trial)
                if score_trial < best_score:
                    best_score = score_trial
                    best_ind = trial
                    print(f"  New Best L/D: {-best_score:.4f} (L={trial[0]:.2f}, W={trial[1]:.2f}, B={trial[2]:.2f}, P={trial[3]:.2f})")
            else:
                new_population.append(population[i])
                new_scores.append(score_target)
        
        population = new_population
        scores = new_scores
        
    print("Optimization Complete.")
    print(f"Best L/D: {-best_score:.4f}")
    print(f"Best Parameters: {best_ind}")
    
    # Generate Final Model
    print(f"Generating optimized model to {BEST_STL_PATH}...")
    config = {
        'body_length': best_ind[0],
        'body_width': best_ind[1],
        'design_mach': DESIGN_MACH,
        'design_beta': best_ind[2],
        'planform_power': best_ind[3],
        'stl_path': BEST_STL_PATH,
        'grid_density': 80
    }
    generate_cone_waverider(config)
    
    # Save Config
    with open("hgv_config_optimized.json", "w") as f:
        json.dump(config, f, indent=4)
        
if __name__ == "__main__":
    optimize()

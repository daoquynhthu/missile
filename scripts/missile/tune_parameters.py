import subprocess
import csv
import math
import os
import sys
from pathlib import Path

# Configuration (paths relative to repository root)
_SCRIPT_DIR = Path(__file__).resolve().parent
EXE_PATH = str(_SCRIPT_DIR.parent.parent / "build" / "bin" / "Release" / "AeroSim.exe")
CSV_PATH = "simulation_data.csv"
T_END = 3000.0 # Increase simulation time to ensure it reaches target

# Optimization Targets
TARGET_RANGE = 4000000.0 # 4000 km
TARGET_APOGEE_MAX = 150000.0 # 150 km (Strict depressed trajectory for skip-glide)
LAUNCH_LAT = 40.960556
LAUNCH_LON = 100.298333
TARGET_LAT = 20.0
TARGET_LON = 135.0

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def run_simulation(start, rate, min_pitch, glide_aoa, pull_up_alt, pull_up_aoa):
    """Runs the simulation with given parameters."""
    if os.path.exists(CSV_PATH):
        try:
            os.remove(CSV_PATH)
        except:
            pass
        
    # Cmd: exe start rate min_pitch t_end glide_aoa pull_up_alt pull_up_aoa
    cmd = [EXE_PATH, str(start), str(rate), str(min_pitch), str(T_END), str(glide_aoa), str(pull_up_alt), str(pull_up_aoa)]
    # Suppress output to keep console clean
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None

    if not os.path.exists(CSV_PATH):
        return None

    return analyze_results()

def analyze_results():
    """Reads CSV and returns metrics."""
    max_alt = -1.0
    impact = False
    
    try:
        with open(CSV_PATH, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            rows = list(reader)
            if not rows:
                return None
                
            for row in rows:
                try:
                    alt = float(row[1])
                    if alt > max_alt:
                        max_alt = alt
                except:
                    continue
            
            # Final state
            last_row = rows[-1]
            last_alt = float(last_row[1])
            last_lat = float(last_row[10])
            last_lon = float(last_row[11])
            
            # Calculate Distance to Target
            dist_to_target = haversine(last_lat, last_lon, TARGET_LAT, TARGET_LON)
            
            if last_alt <= 100.0: # Impact
                impact = True
                
            return {
                "apogee": max_alt,
                "dist_to_target": dist_to_target,
                "impact": impact,
                "final_lat": last_lat,
                "final_lon": last_lon
            }
            
    except Exception as e:
        print(f"Error analyzing CSV: {e}")
        return None

def cost_function(metrics):
    if not metrics:
        return 1e9
        
    # 1. Range Cost (Primary)
    dist = metrics['dist_to_target']
    range_cost = dist # Minimize distance to target
    
    # 2. Apogee Penalty (Soft Constraint)
    apogee_penalty = 0.0
    if metrics['apogee'] > TARGET_APOGEE_MAX:
        apogee_penalty = (metrics['apogee'] - TARGET_APOGEE_MAX) * 10.0
        
    # 3. Glide Bonus (Encourage time in glide phase)
    # Qian Xuesen trajectory: High speed, long glide.
    # We can check if 'Phase' stayed in Glide (2) for a long time?
    # Or just trust Range.
    
    return range_cost + apogee_penalty

import random

def differential_evolution():
    # Parameters to optimize:
    # 0: boost_pitch_start [4.0, 15.0]
    # 1: boost_pitch_rate [0.5, 4.0]
    # 2: min_pitch [5.0, 45.0]
    # 3: glide_aoa_bias [0.0, 15.0]
    # 4: pull_up_alt [20000.0, 50000.0]
    # 5: pull_up_aoa [10.0, 25.0]
    
    bounds = [
        (10.0, 30.0),     # boost_pitch_start (Later for vertical climb)
        (0.5, 1.5),       # boost_pitch_rate (Slower turn to avoid high Alpha)
        (20.0, 50.0),     # min_pitch (Keep nose up at burnout)
        (5.0, 25.0),      # glide_aoa_bias
        (30000.0, 70000.0), # pull_up_alt
        (15.0, 30.0)      # pull_up_aoa
    ]
    
    pop_size = 20  # Increased population
    mutation_factor = 0.8
    crossover_prob = 0.7
    generations = 10 # Increased generations
    
    # Initialize Population
    population = []
    for _ in range(pop_size):
        ind = []
        for b in bounds:
            ind.append(random.uniform(b[0], b[1]))
        population.append(ind)
        
    # Evaluate Initial Population
    scores = []
    best_score = 1e9
    best_ind = None
    
    print("Initializing population...")
    for i, ind in enumerate(population):
        # Unpack: start, rate, min_pitch, glide_aoa, pull_up_alt, pull_up_aoa
        metrics = run_simulation(*ind)
        score = cost_function(metrics)
        scores.append(score)
        
        if score < best_score:
            best_score = score
            best_ind = list(ind)
        
        if metrics:
            print(f"  Ind {i}: Cost {score:.1f} (Dist: {metrics['dist_to_target']/1000.0:.1f}km)")
        else:
            print(f"  Ind {i}: Failed")

    # Evolution Loop
    for gen in range(generations):
        print(f"\nGeneration {gen+1}/{generations} - Best Cost: {best_score:.1f}")
        
        for i in range(pop_size):
            # 1. Mutation
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = random.sample(idxs, 3)
            
            mutant = []
            for j in range(len(bounds)):
                val = population[a][j] + mutation_factor * (population[b][j] - population[c][j])
                val = max(bounds[j][0], min(bounds[j][1], val)) # Clamp
                mutant.append(val)
                
            # 2. Crossover
            trial = []
            for j in range(len(bounds)):
                if random.random() < crossover_prob:
                    trial.append(mutant[j])
                else:
                    trial.append(population[i][j])
                    
            # 3. Selection
            metrics = run_simulation(*trial)
            score = cost_function(metrics)
            
            if score < scores[i]:
                population[i] = trial
                scores[i] = score
                if score < best_score:
                    best_score = score
                    best_ind = list(trial)
                    print(f"  New Best! Cost {best_score:.1f} (Dist: {metrics['dist_to_target']/1000.0:.1f}km)")
            
            print(f"  Gen {gen+1} Ind {i}: Cost {score:.1f}")

    print("\nOptimization Complete.")
    print(f"Best Parameters: {best_ind}")
    run_simulation(*best_ind) # Run one last time to save CSV

if __name__ == "__main__":
    differential_evolution()
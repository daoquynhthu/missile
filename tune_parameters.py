import subprocess
import csv
import math
import os
import sys

# Configuration
EXE_PATH = r"e:\missile\build\bin\Release\AeroSim.exe"
CSV_PATH = "simulation_data.csv"
T_END = 1200.0

# Optimization Targets
TARGET_APOGEE_MIN = 80000.0  # 80 km
TARGET_APOGEE_MAX = 100000.0 # 100 km (User Constraint)
TARGET_RANGE = 2200000.0     # 2200 km

def run_simulation(start, rate, min_pitch):
    """Runs the simulation with given parameters."""
    if os.path.exists(CSV_PATH):
        try:
            os.remove(CSV_PATH)
        except:
            pass
        
    cmd = [EXE_PATH, str(start), str(rate), str(min_pitch), str(T_END)]
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
    final_range = 0.0
    impact = False
    
    try:
        with open(CSV_PATH, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            # Time, Alt, Vel, Mach, Phase, Pitch, Thrust, ...
            # Alt is index 1.
            # Lat is index 10, Lon is index 11.
            
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
            
            # Calculate Range (Great Circle from 0,0)
            # R = EarthRadius * acos(cos(lat)*cos(lon))
            # Assuming launch at 0,0
            RE = 6378137.0
            lat_rad = math.radians(last_lat)
            lon_rad = math.radians(last_lon)
            
            # Central angle
            # Launch at 0,0. Target (lat, lon).
            # cos(c) = sin(0)sin(lat) + cos(0)cos(lat)cos(lon-0)
            #        = cos(lat)cos(lon)
            cos_c = math.cos(lat_rad) * math.cos(lon_rad)
            if cos_c > 1.0: cos_c = 1.0
            if cos_c < -1.0: cos_c = -1.0
            
            final_range = RE * math.acos(cos_c)
            
            if last_alt <= 100.0: # Impact or near ground
                impact = True
                
            return {
                "apogee": max_alt,
                "range": final_range,
                "impact": impact
            }
            
    except Exception as e:
        print(f"Error analyzing CSV: {e}")
        return None

def cost_function(metrics):
    if metrics is None:
        return float('inf')
        
    apogee = metrics['apogee']
    rng = metrics['range']
    impact = metrics['impact']
    
    # Penalties
    cost = 0.0
    
    # Apogee Constraint
    if apogee < TARGET_APOGEE_MIN:
        cost += (TARGET_APOGEE_MIN - apogee) * 0.1
    elif apogee > TARGET_APOGEE_MAX:
        cost += (apogee - TARGET_APOGEE_MAX) * 0.1
        
    # Range Objective (Maximize Range -> Minimize -Range)
    # Scale: 2000km = 2,000,000m. 
    # We want cost to be comparable.
    cost += (TARGET_RANGE - rng) * 0.001
    
    # Impact Constraint
    if not impact:
        cost += 10000.0 # Large penalty for not landing
        
    return cost

def nelder_mead():
    print("Starting Optimization using Nelder-Mead Algorithm...")
    
    # Initial Guess (Best known point)
    x0 = [6.4375, 2.675] # Start, Rate
    step = [0.5, 0.1]
    
    # Simplex initialization
    simplex = [
        x0,
        [x0[0] + step[0], x0[1]],
        [x0[0], x0[1] + step[1]]
    ]
    
    # Evaluate initial simplex
    scores = []
    min_pitch = 5.0
    
    print("Evaluating Initial Simplex:")
    for params in simplex:
        print(f"  Params: {params}...", end='', flush=True)
        metrics = run_simulation(params[0], params[1], min_pitch)
        if metrics:
            score = cost_function(metrics)
            scores.append((score, params, metrics))
            print(f" Cost={score:.2f} (Apogee={metrics['apogee']/1000:.1f}km, Range={metrics['range']/1000:.1f}km)")
        else:
            scores.append((float('inf'), params, None))
            print(" Failed.")
            
    # Algorithm Parameters
    alpha = 1.0  # Reflection
    gamma = 2.0  # Expansion
    rho = 0.5    # Contraction
    sigma = 0.5  # Shrink
    
    max_iter = 20
    tol = 100.0 # Cost tolerance
    
    for i in range(max_iter):
        # 1. Order
        scores.sort(key=lambda x: x[0])
        best = scores[0]
        worst = scores[-1]
        second_worst = scores[-2]
        
        print(f"\nIteration {i+1}: Best Cost={best[0]:.2f} (Start={best[1][0]:.2f}, Rate={best[1][1]:.2f})")
        
        if abs(best[0] - worst[0]) < tol:
            print("Converged.")
            break
            
        # Centroid of the best n points (here n=2, so centroid of best 2 points? No, centroid of all except worst)
        # Dimension D=2. Simplex has D+1=3 points.
        # Centroid of best 2.
        c = [(scores[0][1][0] + scores[1][1][0])/2.0, (scores[0][1][1] + scores[1][1][1])/2.0]
        
        # Reflection
        xr = [c[0] + alpha * (c[0] - worst[1][0]), c[1] + alpha * (c[1] - worst[1][1])]
        # Clamp parameters
        xr[0] = max(0.1, xr[0])
        xr[1] = max(0.1, xr[1])
        
        print(f"  Reflection: {xr}...", end='', flush=True)
        mr = run_simulation(xr[0], xr[1], min_pitch)
        sr = cost_function(mr)
        print(f" Cost={sr:.2f}")
        
        if scores[0][0] <= sr < second_worst[0]:
            scores[-1] = (sr, xr, mr)
            continue
            
        # Expansion
        if sr < scores[0][0]:
            xe = [c[0] + gamma * (xr[0] - c[0]), c[1] + gamma * (xr[1] - c[1])]
            xe[0] = max(0.1, xe[0])
            xe[1] = max(0.1, xe[1])
            
            print(f"  Expansion: {xe}...", end='', flush=True)
            me = run_simulation(xe[0], xe[1], min_pitch)
            se = cost_function(me)
            print(f" Cost={se:.2f}")
            
            if se < sr:
                scores[-1] = (se, xe, me)
            else:
                scores[-1] = (sr, xr, mr)
            continue
            
        # Contraction
        xc = [c[0] + rho * (worst[1][0] - c[0]), c[1] + rho * (worst[1][1] - c[1])]
        # Or Outside Contraction?
        # Standard NM:
        # If sr < worst, outside contraction using xr.
        # If sr >= worst, inside contraction using worst.
        
        use_xr_for_contraction = (sr < worst[0])
        base_point = xr if use_xr_for_contraction else worst[1]
        
        xc = [c[0] + rho * (base_point[0] - c[0]), c[1] + rho * (base_point[1] - c[1])]
        xc[0] = max(0.1, xc[0])
        xc[1] = max(0.1, xc[1])

        print(f"  Contraction: {xc}...", end='', flush=True)
        mc = run_simulation(xc[0], xc[1], min_pitch)
        sc = cost_function(mc)
        print(f" Cost={sc:.2f}")
        
        if sc < min(sr, worst[0]): # Improvement over worst (and reflection if used)
            scores[-1] = (sc, xc, mc)
        else:
            # Shrink
            print("  Shrink")
            for j in range(1, len(scores)):
                pt = scores[j][1]
                new_pt = [scores[0][1][0] + sigma * (pt[0] - scores[0][1][0]), 
                          scores[0][1][1] + sigma * (pt[1] - scores[0][1][1])]
                new_pt[0] = max(0.1, new_pt[0])
                new_pt[1] = max(0.1, new_pt[1])
                
                m = run_simulation(new_pt[0], new_pt[1], min_pitch)
                s = cost_function(m)
                scores[j] = (s, new_pt, m)

    print("\nOptimization Complete.")
    best = scores[0]
    print(f"Best Parameters: Start={best[1][0]}, Rate={best[1][1]}")
    if best[2]:
        print(f"Metrics: Apogee={best[2]['apogee']/1000:.1f}km, Range={best[2]['range']/1000:.1f}km")

if __name__ == "__main__":
    nelder_mead()

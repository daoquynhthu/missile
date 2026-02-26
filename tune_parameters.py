import subprocess
import csv
import math
import os
import sys

# Configuration
EXE_PATH = r"e:\missile\build\bin\Release\AeroSim.exe"
CSV_PATH = "simulation_data.csv"
T_END = 2000.0 # Increase simulation time to ensure it reaches target

# Optimization Targets
TARGET_APOGEE_MAX = 100000.0 # 100 km (User Constraint)
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
    if metrics is None:
        return 1e9
        
    # Weights
    w_dist = 1.0     # Minimize distance to target
    w_apogee = 100.0 # Penalize apogee violation heavily
    
    cost = metrics["dist_to_target"] / 1000.0 # Convert to km for reasonable scale
    
    # Apogee Penalty
    if metrics["apogee"] > TARGET_APOGEE_MAX:
        over = (metrics["apogee"] - TARGET_APOGEE_MAX) / 1000.0
        cost += w_apogee * (over ** 2)
        
    return cost

def nelder_mead():
    # Initial Guess [Start, Rate]
    x0 = [6.4375, 2.775] # Current best
    step = [0.5, 0.2]
    
    # Simplex initialization
    simplex = [x0]
    for i in range(len(x0)):
        point = list(x0)
        point[i] += step[i]
        simplex.append(point)
        
    # Evaluation
    scores = []
    for point in simplex:
        print(f"Evaluating: {point}")
        metrics = run_simulation(point[0], point[1], 5.0)
        if metrics:
            score = cost_function(metrics)
            print(f"  Apogee: {metrics['apogee']/1000:.1f}km, Dist: {metrics['dist_to_target']/1000:.1f}km -> Cost: {score:.1f}")
            scores.append((score, point, metrics))
        else:
            scores.append((1e9, point, None))
            
    # Iterations
    alpha = 1.0  # Reflection
    gamma = 2.0  # Expansion
    rho = 0.5    # Contraction
    sigma = 0.5  # Shrink
    
    for i in range(15): # Max iterations
        scores.sort(key=lambda x: x[0])
        best = scores[0]
        worst = scores[-1]
        second_worst = scores[-2]
        
        print(f"Iter {i+1}: Best Cost {best[0]:.1f} (Start={best[1][0]:.2f}, Rate={best[1][1]:.2f}) - Dist: {best[2]['dist_to_target']/1000:.1f}km")
        
        # Centroid
        centroid = [0.0] * len(x0)
        for j in range(len(x0)):
            for k in range(len(scores)-1):
                centroid[j] += scores[k][1][j]
            centroid[j] /= (len(scores)-1)
            
        # Reflection
        xr = [0.0] * len(x0)
        for j in range(len(x0)):
            xr[j] = centroid[j] + alpha * (centroid[j] - worst[1][j])
            
        metrics_r = run_simulation(xr[0], xr[1], 5.0)
        score_r = cost_function(metrics_r)
        
        if scores[0][0] <= score_r < second_worst[0]:
            scores[-1] = (score_r, xr, metrics_r)
            continue
            
        # Expansion
        if score_r < scores[0][0]:
            xe = [0.0] * len(x0)
            for j in range(len(x0)):
                xe[j] = centroid[j] + gamma * (xr[j] - centroid[j])
            metrics_e = run_simulation(xe[0], xe[1], 5.0)
            score_e = cost_function(metrics_e)
            if score_e < score_r:
                scores[-1] = (score_e, xe, metrics_e)
            else:
                scores[-1] = (score_r, xr, metrics_r)
            continue
            
        # Contraction
        xc = [0.0] * len(x0)
        for j in range(len(x0)):
            xc[j] = centroid[j] + rho * (worst[1][j] - centroid[j])
        metrics_c = run_simulation(xc[0], xc[1], 5.0)
        score_c = cost_function(metrics_c)
        if score_c < worst[0]:
            scores[-1] = (score_c, xc, metrics_c)
            continue
            
        # Shrink
        for k in range(1, len(scores)):
            for j in range(len(x0)):
                scores[k][1][j] = scores[0][1][j] + sigma * (scores[k][1][j] - scores[0][1][j])
            metrics = run_simulation(scores[k][1][0], scores[k][1][1], 5.0)
            scores[k] = (cost_function(metrics), scores[k][1], metrics)

    print("\nOptimization Complete")
    print(f"Best Parameters: Start={best[1][0]:.4f}, Rate={best[1][1]:.4f}")
    print(f"Result: Apogee={best[2]['apogee']/1000:.1f}km, Distance={best[2]['dist_to_target']/1000:.1f}km")

if __name__ == "__main__":
    nelder_mead()
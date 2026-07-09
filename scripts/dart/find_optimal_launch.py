import subprocess
import os
import pandas as pd
import numpy as np

SIM_EXE = r"build\Release\DartSim.exe"
TARGET_DIST = 25.233
TARGET_YAW_DEG = 7.3
TARGET_Z = 1.5

def run_sim(v0, pitch):
    traj_path = "output/logs/temp_traj.csv"
    if os.path.exists(traj_path):
        os.remove(traj_path)
    os.makedirs("output/logs", exist_ok=True)
    
    cmd = [SIM_EXE, str(v0), str(pitch), str(TARGET_YAW_DEG), traj_path, "0"]
    subprocess.run(cmd, capture_output=True)
    if os.path.exists(traj_path):
        try:
            df = pd.read_csv(traj_path)
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna()
            if df.empty:
                return None
            target_x = TARGET_DIST * np.cos(TARGET_YAW_DEG * np.pi / 180.0)
            target_y = TARGET_DIST * np.sin(TARGET_YAW_DEG * np.pi / 180.0)
            df["miss_3d"] = np.sqrt((df["X"] - target_x) ** 2 + (df["Y"] - target_y) ** 2 + (df["Z"] - TARGET_Z) ** 2)
            idx = df["miss_3d"].idxmin()
            best = df.iloc[idx]
            return {
                "v0": v0,
                "pitch": pitch,
                "miss_3d": float(best["miss_3d"]),
                "x": float(best["X"]),
                "y": float(best["Y"]),
                "z": float(best["Z"]),
                "time": float(best["Time"]),
            }
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return None
    return None

def optimize_pitch_for_speed(v0, low=0.0, high=15.0, iterations=10):
    best = None
    for _ in range(iterations):
        p1 = low + (high - low) / 3.0
        p2 = high - (high - low) / 3.0
        r1 = run_sim(round(v0, 4), round(p1, 4))
        r2 = run_sim(round(v0, 4), round(p2, 4))
        if r1 is None or r2 is None:
            break
        print(f"v0={v0:5.2f} m/s p1={p1:6.3f} miss1={r1['miss_3d']:.4f} | p2={p2:6.3f} miss2={r2['miss_3d']:.4f}")
        if best is None or r1["miss_3d"] < best["miss_3d"]:
            best = r1
        if r2["miss_3d"] < best["miss_3d"]:
            best = r2
        if r1["miss_3d"] <= r2["miss_3d"]:
            high = p2
        else:
            low = p1

    final_pitches = np.linspace(low, high, 7)
    for pitch in final_pitches:
        result = run_sim(round(v0, 4), round(pitch, 4))
        if result is not None and (best is None or result["miss_3d"] < best["miss_3d"]):
            best = result
    return best

def search_design():
    speed_grid = np.linspace(25.0, 45.0, 9)
    best = None

    for v0 in speed_grid:
        candidate = optimize_pitch_for_speed(v0)
        if candidate is None:
            continue
        print(f"speed={candidate['v0']:.3f} m/s -> best pitch={candidate['pitch']:.4f} deg miss={candidate['miss_3d']:.4f} m")
        if best is None or candidate["miss_3d"] < best["miss_3d"]:
            best = candidate

    refine_speeds = np.linspace(max(20.0, best["v0"] - 2.0), best["v0"] + 2.0, 9)
    for v0 in refine_speeds:
        candidate = optimize_pitch_for_speed(v0, max(0.0, best["pitch"] - 2.0), best["pitch"] + 2.0, iterations=8)
        if candidate is not None and candidate["miss_3d"] < best["miss_3d"]:
            best = candidate

    return best

if __name__ == "__main__":
    # First regenerate table
    subprocess.run(["python", "scripts/dart/generate_dart_aero_table.py"])
    # Rebuild
    subprocess.run(["cmake", "--build", "build", "--config", "Release", "--target", "DartSim"])
    
    best = search_design()
    print(f"\nBest Design: v0={best['v0']:.4f} m/s, pitch={best['pitch']:.4f} deg, miss={best['miss_3d']:.4f} m")

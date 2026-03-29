import subprocess
import os
import csv
import math

EXE_PATH = r"e:\missile\build\bin\Release\AeroSim.exe"
CSV_PATH = "simulation_data.csv"

def run_debug():
    # ... (same as before)
    # reasonable params
    start = 10.0
    rate = 1.0
    min_pitch = 30.0
    t_end = 1000.0
    glide_aoa = 10.0
    pull_up_alt = 40000.0
    pull_up_aoa = 20.0

    print("Running debug simulation...")
    cmd = [EXE_PATH, str(start), str(rate), str(min_pitch), str(t_end), str(glide_aoa), str(pull_up_alt), str(pull_up_aoa)]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("Simulation Output:")
    print(result.stdout)
    if result.stderr:
        print("Simulation Error:")
        print(result.stderr)

    if not os.path.exists(CSV_PATH):
        print("No CSV generated.")
        return

    print("Analyzing CSV...")
    with open(CSV_PATH, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
        
        # Header: t, alt, vel, mach, aoa, pitch, gamma, q_alpha, ax, ay, az, lat, lon, mass
        
        max_alt = 0
        max_vel = 0
        
        for row in rows:
            t = float(row[0])
            alt = float(row[1])
            vel = float(row[2])
            thrust = float(row[6])
            
            if alt > max_alt: max_alt = alt
            if vel > max_vel: max_vel = vel
            
            if abs(t - 10.0) < 0.5 or abs(t - 30.0) < 0.5 or abs(t - 50.0) < 0.5:
                print(f"T={t:.1f}, Alt={alt:.1f}, Vel={vel:.1f}, Pitch={row[5]}, Thrust={thrust:.1f}, Mass={row[12]}")

        print(f"Max Alt: {max_alt/1000.0:.2f} km")
        print(f"Max Vel: {max_vel:.1f} m/s")

if __name__ == "__main__":
    run_debug()

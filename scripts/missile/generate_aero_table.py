import subprocess
import json
import csv
import os
import sys
import concurrent.futures
import time

# Configuration
AERO_CALC_EXE = r"e:\missile\build\bin\Release\AeroCalc.exe"
STL_PATH = "e:/missile/hgv_model_optimized.stl"
OUTPUT_CSV = "e:/missile/aerodynamics_table.csv"
CONFIG_PATH = "e:/missile/hgv_config_optimized.json"

# High-Fidelity Grids
MACH_GRID = [0.5, 0.8, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0]
ALPHA_GRID = [-10.0, -5.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0]
BETA = 0.0

# Global ref values (Matched to missile_config.cpp for IRBM)
REF_AREA = 1.1310
REF_LENGTH = 12.0
REF_SPAN = 3.0

def compute_point(args):
    mach, alpha = args
    cmd = [
        AERO_CALC_EXE,
        STL_PATH,
        str(mach),
        str(alpha),
        str(BETA),
        str(REF_AREA),
        str(REF_LENGTH),
        str(REF_SPAN)
    ]
    
    try:
        # Run AeroCalc
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = proc.stdout
        
        # Parse JSON output
        start = output.find('{')
        end = output.rfind('}') + 1
        if start != -1 and end != -1:
            json_str = output[start:end]
            data = json.loads(json_str)
            return {
                'Mach': mach,
                'Alpha': alpha,
                'Beta': BETA,
                'CX': data.get('CX', 0.0),
                'CY': data.get('CY', 0.0),
                'CZ': data.get('CZ', 0.0),
                'CL': data.get('CL', 0.0),
                'CD': data.get('CD', 0.0),
                'L_D': data.get('L_D', 0.0),
                'Cl': data.get('Cl', 0.0),
                'Cm': data.get('Cm', 0.0),
                'Cn': data.get('Cn', 0.0)
            }
        else:
            print(f"Error parsing output for Mach={mach}, Alpha={alpha}")
            return None
    except Exception as e:
        print(f"Error computing Mach={mach}, Alpha={alpha}: {e}")
        return None

def main():
    global REF_AREA, REF_LENGTH, REF_SPAN
    
    # --- 1. Load Config & Calculate Reference Values ---
    print(f"Using IRBM Reference Values: RefArea={REF_AREA:.4f}, RefLen={REF_LENGTH:.4f}, RefSpan={REF_SPAN:.4f}")

    if not os.path.exists(STL_PATH):
        print(f"Error: Optimized model {STL_PATH} not found. Run optimize_shape.py first.")
        return

    # Skip config loading as we use hardcoded IRBM values for consistency with missile_config.cpp
    
    # --- 2. Generate Grid ---
    tasks = []
    for mach in MACH_GRID:
        for alpha in ALPHA_GRID:
            tasks.append((mach, alpha))
            
    print(f"Generating table for {len(tasks)} points...")
    print(f"Mach Grid: {MACH_GRID}")
    print(f"Alpha Grid: {ALPHA_GRID}")
    
    results = []
    start_time = time.time()
    
    # Parallel Execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_point = {executor.submit(compute_point, task): task for task in tasks}
        
        completed = 0
        for future in concurrent.futures.as_completed(future_to_point):
            task = future_to_point[future]
            try:
                res = future.result()
                if res:
                    results.append(res)
            except Exception as exc:
                print(f"Task {task} generated an exception: {exc}")
            
            completed += 1
            if completed % 10 == 0:
                print(f"Progress: {completed}/{len(tasks)} ({completed/len(tasks)*100:.1f}%)")

    end_time = time.time()
    print(f"Computation finished in {end_time - start_time:.2f}s")
    
    # --- 3. Save to CSV ---
    if not results:
        print("Error: No results generated.")
        return
        
    # Sort results
    results.sort(key=lambda x: (x['Mach'], x['Alpha']))
    
    # Write Full Format
    try:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header matching AerodynamicsModel::load_csv_table expectation
            # mach,alpha,beta,CX,CY,CZ,CL,CD,L_D,Cl,Cm,Cn
            writer.writerow(['Mach', 'Alpha', 'Beta', 'CX', 'CY', 'CZ', 'CL', 'CD', 'L_D', 'Cl', 'Cm', 'Cn'])
            
            for r in results:
                writer.writerow([
                    f"{r['Mach']:.2f}",
                    f"{r['Alpha']:.2f}",
                    f"{r['Beta']:.2f}",
                    f"{r['CX']:.6f}",
                    f"{r['CY']:.6f}",
                    f"{r['CZ']:.6f}",
                    f"{r['CL']:.6f}",
                    f"{r['CD']:.6f}",
                    f"{r['L_D']:.6f}",
                    f"{r['Cl']:.6f}",
                    f"{r['Cm']:.6f}",
                    f"{r['Cn']:.6f}"
                ])
                
        print(f"Successfully saved {len(results)} points to {OUTPUT_CSV}")
        
    except Exception as e:
        print(f"Error saving CSV: {e}")

if __name__ == "__main__":
    main()

import subprocess
import json
import csv
import os
import sys
import concurrent.futures
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

AERO_CALC_EXE = os.path.join(PROJECT_ROOT, "build", "bin", "Release", "AeroCalc.exe")
STL_PATH = os.path.join(PROJECT_ROOT, "data", "missile", "hgv_model_optimized.stl")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "missile", "aerodynamics_table.csv")

MACH_GRID = [0.5, 0.8, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0]
ALPHA_GRID = [-10.0, -5.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0]
BETA_GRID = [0.0]
BETA = 0.0

REF_AREA = 1.1310
REF_LENGTH = 12.0
REF_SPAN = 3.0
WET_AREA = 40.0
PLANFORM_AREA = 3.0
BASE_AREA = 0.1
NOSE_FINENESS = 3.0
COM_X = 6.0

def compute_point(args):
    mach, alpha, beta = args
    cmd = [
        AERO_CALC_EXE,
        "--stl", STL_PATH,
        "--mach", str(mach),
        "--alpha", str(alpha),
        "--beta", str(beta),
        "--ref-area", str(REF_AREA),
        "--ref-length", str(REF_LENGTH),
        "--ref-span", str(REF_SPAN),
        "--wet-area", str(WET_AREA),
        "--planform-area", str(PLANFORM_AREA),
        "--base-area", str(BASE_AREA),
        "--nose-fineness", str(NOSE_FINENESS),
        "--com-x", str(COM_X),
        "--mode", "auto",
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120)
        output = proc.stdout
        start = output.find('{')
        end = output.rfind('}') + 1
        if start != -1 and end != -1:
            data = json.loads(output[start:end])
            return {
                'Mach': mach,
                'Alpha': alpha,
                'Beta': beta,
                'CX': data.get('CX', 0.0),
                'CY': data.get('CY', 0.0),
                'CZ': data.get('CZ', 0.0),
                'CL': data.get('CL', 0.0),
                'CD': data.get('CD', 0.0),
                'L_D': data.get('L_D', 0.0),
                'Cl': data.get('Cl', 0.0),
                'Cm': data.get('Cm', 0.0),
                'Cn': data.get('Cn', 0.0),
            }
        else:
            print(f"Parse error for M={mach} a={alpha}: {output[:200]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"Timeout for M={mach} a={alpha}")
        return None
    except Exception as e:
        print(f"Error M={mach} a={alpha}: {e}")
        return None

def main():
    if not os.path.exists(AERO_CALC_EXE):
        print(f"Error: AeroCalc not found at {AERO_CALC_EXE}. Build first.")
        return

    tasks = []
    for mach in MACH_GRID:
        for alpha in ALPHA_GRID:
            for beta in BETA_GRID:
                tasks.append((mach, alpha, beta))

    print(f"Generating {len(tasks)} points...")
    print(f"Mach: {MACH_GRID}")
    print(f"Alpha: {ALPHA_GRID}")

    results = []
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {executor.submit(compute_point, t): t for t in tasks}
        completed = 0
        for future in concurrent.futures.as_completed(future_map):
            task = future_map[future]
            try:
                res = future.result()
                if res:
                    results.append(res)
            except Exception as e:
                print(f"Task {task} failed: {e}")
            completed += 1
            if completed % 10 == 0:
                print(f"Progress: {completed}/{len(tasks)} ({completed/len(tasks)*100:.1f}%)")

    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.1f}s, got {len(results)}/{len(tasks)} results")

    if not results:
        print("No results. Aborting.")
        return

    results.sort(key=lambda x: (x['Mach'], x['Alpha'], x['Beta']))

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Mach', 'Alpha', 'Beta', 'CX', 'CY', 'CZ', 'CL', 'CD', 'L_D', 'Cl', 'Cm', 'Cn'])
        for r in results:
            writer.writerow([
                f"{r['Mach']:.2f}", f"{r['Alpha']:.2f}", f"{r['Beta']:.2f}",
                f"{r['CX']:.6f}", f"{r['CY']:.6f}", f"{r['CZ']:.6f}",
                f"{r['CL']:.6f}", f"{r['CD']:.6f}", f"{r['L_D']:.6f}",
                f"{r['Cl']:.6f}", f"{r['Cm']:.6f}", f"{r['Cn']:.6f}",
            ])

    print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

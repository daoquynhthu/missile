import subprocess
from pathlib import Path

import pandas as pd


MC_EXE = Path("build/Release/DartMC.exe")
OUTPUT_CSV = Path("output/logs/dart_mc_closed_loop_structure_sweep.csv")
SEED = 123456789
NUM_SIMS = 2000
V0 = 25.0
PITCH = 6.2311
YAW = 7.3
V0_SIGMA = 0.3
PITCH_SIGMA = 0.2
YAW_SIGMA = 0.2


def run_case(args):
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    return proc.stdout


def main():
    if not MC_EXE.exists():
        raise FileNotFoundError(f"Missing executable: {MC_EXE}")

    terminal_range_values = "10,12,14"
    max_accel_values = "5,6,7"
    rate_damp_values = "0.35,0.55,0.75"
    pitch_moment_values = "8,10,12"
    yaw_moment_values = "2,3,4"

    cmd = [
        str(MC_EXE),
        "--closed-loop-structure-sweep",
        str(NUM_SIMS),
        f"{V0:.4f}",
        f"{PITCH:.4f}",
        f"{YAW:.4f}",
        f"{V0_SIGMA:.4f}",
        f"{PITCH_SIGMA:.4f}",
        f"{YAW_SIGMA:.4f}",
        terminal_range_values,
        max_accel_values,
        rate_damp_values,
        pitch_moment_values,
        yaw_moment_values,
        str(SEED),
    ]
    print(run_case(cmd))

    df = pd.read_csv(OUTPUT_CSV)
    best_policies = df.sort_values(["HitRate", "MeanMissMM"], ascending=[False, True]).head(20)
    best = best_policies.iloc[0]

    print("\nBest closed-loop structure policies:")
    print(best_policies.to_string(index=False))

    guided_cmd = [
        str(MC_EXE),
        str(NUM_SIMS),
        f"{V0:.4f}",
        f"{PITCH:.4f}",
        f"{YAW:.4f}",
        "2.0",
        "2.5",
        f"{V0_SIGMA:.4f}",
        f"{PITCH_SIGMA:.4f}",
        f"{YAW_SIGMA:.4f}",
        str(SEED),
    ]
    ballistic_cmd = [
        str(MC_EXE),
        str(NUM_SIMS),
        f"{V0:.4f}",
        f"{PITCH:.4f}",
        f"{YAW:.4f}",
        "0",
        "0",
        f"{V0_SIGMA:.4f}",
        f"{PITCH_SIGMA:.4f}",
        f"{YAW_SIGMA:.4f}",
        str(SEED),
    ]

    print("\nCurrent closed-loop baseline:")
    print(run_case(guided_cmd))
    print("\nBallistic baseline:")
    print(run_case(ballistic_cmd))
    best_cmd = [
        str(MC_EXE),
        "--closed-loop-structure-sweep",
        str(NUM_SIMS),
        f"{V0:.4f}",
        f"{PITCH:.4f}",
        f"{YAW:.4f}",
        f"{V0_SIGMA:.4f}",
        f"{PITCH_SIGMA:.4f}",
        f"{YAW_SIGMA:.4f}",
        f"{best['TerminalRange']:.6f}",
        f"{best['MaxAccel']:.6f}",
        f"{best['RateDamp']:.6f}",
        f"{best['PitchMomentCoeff']:.6f}",
        f"{best['YawMomentCoeff']:.6f}",
        str(SEED),
    ]
    print("\nBest closed-loop structure policy:")
    print(run_case(best_cmd))
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

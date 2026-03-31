import subprocess
from pathlib import Path

import pandas as pd


MC_EXE = Path("build/Release/RMDartMC.exe")
OUTPUT_DIR = Path("output/logs")
SEED = 123456789

def main():
    if not MC_EXE.exists():
        raise FileNotFoundError(f"Missing executable: {MC_EXE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pitch_values = "6.1311,6.1811,6.2311,6.2811,6.3311,6.3811"
    dp_values = "-0.20,-0.19,-0.18,-0.17,-0.16,-0.15,-0.14,-0.13,-0.12,-0.11,-0.10,-0.09,-0.08,-0.07,-0.06,-0.05,-0.04,0.00"
    cmd = [
        str(MC_EXE),
        "--forced-dp-sweep",
        "25.0",
        pitch_values,
        "7.3",
        dp_values,
        str(SEED),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    print(proc.stdout)

    csv_path = OUTPUT_DIR / "rm_dart_mc_forced_dp_sweep.csv"
    df = pd.read_csv(csv_path)

    best_per_pitch = (
        df.sort_values(["PitchDeg", "MeanMissMM"])
        .groupby("PitchDeg", as_index=False)
        .first()
        .sort_values("PitchDeg")
    )
    global_best = df.sort_values("MeanMissMM").head(12)

    print("\nBest per pitch:")
    print(best_per_pitch.to_string(index=False))
    print("\nGlobal best:")
    print(global_best.to_string(index=False))
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()

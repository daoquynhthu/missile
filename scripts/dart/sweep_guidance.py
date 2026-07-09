import subprocess
import pandas as pd
import os

MC_EXE = r"build\Release\DartMC.exe"
SWEEP_OUT = "output/logs/dart_mc_sweep.csv"

def sweep_params(pitch=9.6664):
    nav_ratios = [3.0, 4.0, 5.0]
    ctrl_gains = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    cmd = [
        MC_EXE,
        "--sweep",
        "2000",
        "25.0",
        str(pitch),
        "7.3",
        ",".join(str(v) for v in nav_ratios),
        ",".join(str(v) for v in ctrl_gains),
    ]
    subprocess.run(cmd, check=True)

    if not os.path.exists(SWEEP_OUT):
        raise FileNotFoundError(SWEEP_OUT)

    df = pd.read_csv(SWEEP_OUT)
    print(f"{'NavRatio':>10} | {'CtrlGain':>10} | {'HitRate':>10} | {'MeanMissMM':>12}")
    print("-" * 52)

    best_idx = df["HitRate"].astype(float).idxmax()
    for _, row in df.iterrows():
        print(f"{row['NavRatio']:10.1f} | {row['CtrlGain']:10.1f} | {row['HitRate']:10.2f}% | {row['MeanMissMM']:12.2f}")

    best = df.loc[best_idx]
    print(f"\nBest: {best['HitRate']:.2f}% (NR={best['NavRatio']}, CG={best['CtrlGain']})")

if __name__ == "__main__":
    sweep_params()

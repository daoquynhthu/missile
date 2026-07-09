import math
import subprocess
from pathlib import Path

import pandas as pd


GPU_SCAN_CSV = Path("output/logs/dart_mc_forced_dp_sweep.csv")
DEBUG_EXE = Path("build/bin/Release/DartDebug.exe")
TARGET_DIST = 25.233
TARGET_YAW_DEG = 7.3
TARGET_Z = 1.5
V0 = 25.0
YAW = 7.3


def first_guidance_state(pitch_deg: float):
    cmd = [
        str(DEBUG_EXE),
        f"{V0:.4f}",
        f"{pitch_deg:.4f}",
        f"{YAW:.4f}",
        "2",
        "2.5",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    df = pd.read_csv("dart_debug_log.csv")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    active = df[df["dp"].abs() > 1e-6]
    if active.empty:
        raise RuntimeError(f"No guidance activity for pitch={pitch_deg}")
    row = active.iloc[0]
    tx = TARGET_DIST * math.cos(math.radians(TARGET_YAW_DEG))
    x_rem = tx - float(row["X"])
    tgo = x_rem / max(float(row["Vx"]), 1.0)
    z_pred = float(row["Z"]) + float(row["Vz"]) * tgo + 0.5 * 9.788 * tgo * tgo
    z_err = TARGET_Z - z_pred
    return {
        "pitch_deg": pitch_deg,
        "x_rem": x_rem,
        "z_err": z_err,
        "dp_first": float(row["dp"]),
    }


def main():
    if not GPU_SCAN_CSV.exists():
        raise FileNotFoundError(f"Missing scan csv: {GPU_SCAN_CSV}")
    if not DEBUG_EXE.exists():
        raise FileNotFoundError(f"Missing debug exe: {DEBUG_EXE}")

    df = pd.read_csv(GPU_SCAN_CSV)
    best = (
        df.sort_values(["PitchDeg", "MeanMissMM"])
        .groupby("PitchDeg", as_index=False)
        .first()
        .sort_values("PitchDeg")
    )

    rows = []
    for pitch_deg in best["PitchDeg"]:
        state = first_guidance_state(float(pitch_deg))
        best_row = best[best["PitchDeg"] == pitch_deg].iloc[0]
        rows.append(
            {
                "pitch_deg": float(pitch_deg),
                "best_dp": float(best_row["ForcedDP"]),
                "best_miss_mm": float(best_row["MeanMissMM"]),
                "z_err_m": float(state["z_err"]),
                "x_rem_m": float(state["x_rem"]),
            }
        )

    learned = pd.DataFrame(rows).sort_values("z_err_m").reset_index(drop=True)
    thresholds = []
    for idx in range(len(learned) - 1):
        thresholds.append(0.5 * (learned.loc[idx, "z_err_m"] + learned.loc[idx + 1, "z_err_m"]))

    print("Learned points:")
    print(learned.to_string(index=False))
    print("\nThresholds:")
    for idx, value in enumerate(thresholds, start=1):
        print(f"guid_lookup_err_{idx} = {abs(value):.6f}")
    print("\nDP values:")
    for idx, value in enumerate(learned["best_dp"].abs(), start=1):
        print(f"guid_lookup_dp_{idx} = {value:.6f}")


if __name__ == "__main__":
    main()

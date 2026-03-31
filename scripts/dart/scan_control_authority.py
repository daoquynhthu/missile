import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


DEBUG_EXE = Path("build/bin/Release/RMDartDebug.exe")
OUTPUT_DIR = Path("output/logs")
TARGET_DIST = 25.233
TARGET_YAW_DEG = 7.3
TARGET_Z = 1.5
DEFAULT_V0 = 25.0
DEFAULT_YAW = 7.3


def target_xyz():
    yaw_rad = math.radians(TARGET_YAW_DEG)
    return (
        TARGET_DIST * math.cos(yaw_rad),
        TARGET_DIST * math.sin(yaw_rad),
        TARGET_Z,
    )


def run_debug(v0: float, pitch_deg: float, yaw_deg: float, forced_dp: float):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(DEBUG_EXE),
        f"{v0:.4f}",
        f"{pitch_deg:.4f}",
        f"{yaw_deg:.4f}",
        "0",
        "0",
        f"{forced_dp:.4f}",
        "0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    df = pd.read_csv("dart_debug_log.csv")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    tx, ty, tz = target_xyz()
    df["miss_3d"] = np.sqrt((df["X"] - tx) ** 2 + (df["Y"] - ty) ** 2 + (df["Z"] - tz) ** 2)
    best = df.loc[df["miss_3d"].idxmin()]
    return {
        "v0": v0,
        "pitch_deg": pitch_deg,
        "forced_dp": forced_dp,
        "miss_mm": float(best["miss_3d"] * 1000.0),
        "z_err_mm": float((best["Z"] - tz) * 1000.0),
        "x": float(best["X"]),
        "y": float(best["Y"]),
        "z": float(best["Z"]),
        "time": float(best["Time"]),
        "dp": float(best["dp"]),
        "mom_y": float(best["MomY"]),
    }


def run_scan(
    pitch_values: np.ndarray,
    dp_values: np.ndarray,
    v0: float = DEFAULT_V0,
    yaw_deg: float = DEFAULT_YAW,
):
    rows = []
    total = len(pitch_values) * len(dp_values)
    idx = 0
    for pitch_deg in pitch_values:
        for forced_dp in dp_values:
            idx += 1
            result = run_debug(v0, float(pitch_deg), yaw_deg, float(forced_dp))
            rows.append(result)
            print(
                f"[{idx:03d}/{total:03d}] pitch={pitch_deg:.4f} dp={forced_dp:.4f} "
                f"miss={result['miss_mm']:.3f}mm zerr={result['z_err_mm']:.3f}mm"
            )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame):
    best_per_pitch = (
        df.sort_values(["pitch_deg", "miss_mm"])
        .groupby("pitch_deg", as_index=False)
        .first()
        .sort_values("pitch_deg")
    )
    global_best = df.sort_values("miss_mm").head(12)
    return best_per_pitch, global_best


def main():
    if not DEBUG_EXE.exists():
        raise FileNotFoundError(f"Missing debug executable: {DEBUG_EXE}")

    pitch_values = np.array([6.1311, 6.1811, 6.2311, 6.2811, 6.3311, 6.3811], dtype=float)
    dp_values = np.round(np.arange(-0.20, 0.001, 0.01), 4)

    df = run_scan(pitch_values, dp_values)
    csv_path = OUTPUT_DIR / "authority_scan_results.csv"
    df.to_csv(csv_path, index=False)

    best_per_pitch, global_best = summarize(df)
    print("\nBest per pitch:")
    print(best_per_pitch.to_string(index=False))
    print("\nGlobal best:")
    print(global_best.to_string(index=False))
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()

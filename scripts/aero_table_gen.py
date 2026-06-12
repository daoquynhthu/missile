"""Python ctypes wrapper for aero_solver_c.dll

Usage:
    from aero_table_gen import generate_aero_table

    ok = generate_aero_table(
        stl_path="data/missile/hgv_model_optimized.stl",
        csv_path="data/missile/aerodynamics_table.csv",
        mach_grid=[0.5, 2.0, 10.0],
        alpha_grid=[-5, 0, 5, 10],
        beta_grid=[0.0],
        ref_area=1.131, ref_length=12.0, ref_span=3.0, com_x=6.0
    )
"""

import ctypes
import os
from pathlib import Path

# Locate the DLL
_dll_dir = Path(__file__).resolve().parent.parent / "build" / "bin" / "Release"
_dll_path = _dll_dir / "aero_solver_c.dll"

if not _dll_path.exists():
    raise RuntimeError(f"Cannot find aero_solver_c.dll at {_dll_path}")

_lib = ctypes.CDLL(str(_dll_path))

# --- generate_aero_table_c ---
_lib.generate_aero_table_c.argtypes = [
    ctypes.c_char_p,  # stl_path
    ctypes.c_char_p,  # csv_path
    ctypes.POINTER(ctypes.c_double),  # mach_grid
    ctypes.c_int,     # n_mach
    ctypes.POINTER(ctypes.c_double),  # alpha_grid
    ctypes.c_int,     # n_alpha
    ctypes.POINTER(ctypes.c_double),  # beta_grid
    ctypes.c_int,     # n_beta
    ctypes.c_float,   # ref_area
    ctypes.c_float,   # ref_length
    ctypes.c_float,   # ref_span
    ctypes.c_float,   # com_x
]
_lib.generate_aero_table_c.restype = ctypes.c_int

# --- aero_solver_last_error ---
_lib.aero_solver_last_error.argtypes = []
_lib.aero_solver_last_error.restype = ctypes.c_char_p


def generate_aero_table(
    stl_path: str,
    csv_path: str,
    mach_grid: list[float],
    alpha_grid: list[float],
    beta_grid: list[float] = None,
    ref_area: float = 1.131,
    ref_length: float = 12.0,
    ref_span: float = 3.0,
    com_x: float = 6.0,
) -> bool:
    """Generate a complete aerodynamics table using the GPU solver.

    Args:
        stl_path: Path to STL geometry file (relative to CWD).
        csv_path: Output CSV file path.
        mach_grid: List of Mach numbers.
        alpha_grid: List of angles of attack (degrees).
        beta_grid: List of sideslip angles (degrees). Defaults to [0.0].
        ref_area: Reference area (m^2).
        ref_length: Reference length (m).
        ref_span: Reference span (m).
        com_x: Center of mass X position (m).

    Returns:
        True on success, False on error.
    """
    if beta_grid is None:
        beta_grid = [0.0]

    # Convert Python lists to C arrays
    mach_arr = (ctypes.c_double * len(mach_grid))(*mach_grid)
    alpha_arr = (ctypes.c_double * len(alpha_grid))(*alpha_grid)
    beta_arr = (ctypes.c_double * len(beta_grid))(*beta_grid)

    ret = _lib.generate_aero_table_c(
        stl_path.encode("utf-8"),
        csv_path.encode("utf-8"),
        mach_arr, len(mach_grid),
        alpha_arr, len(alpha_grid),
        beta_arr, len(beta_grid),
        ctypes.c_float(ref_area),
        ctypes.c_float(ref_length),
        ctypes.c_float(ref_span),
        ctypes.c_float(com_x),
    )

    if ret != 0:
        err = _lib.aero_solver_last_error()
        print(f"Error: {err.decode('utf-8') if err else 'unknown'}")
        return False
    return True


if __name__ == "__main__":
    # Test: generate a small 5x4 grid (20 points)
    ok = generate_aero_table(
        stl_path="data/missile/hgv_model_optimized.stl",
        csv_path="data/missile/aerodynamics_table.csv",
        mach_grid=[0.5, 2.0, 5.0, 10.0, 20.0],
        alpha_grid=[-5, 0, 5, 10],
        beta_grid=[0.0],
    )
    print(f"{'PASS' if ok else 'FAIL'}")

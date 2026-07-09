import numpy as np
import csv
import os

TOTAL_LENGTH = 0.265
NOSE_WIDTH = 0.030
NOSE_HEIGHT = 0.020
BODY_RADIUS = 0.0125
NOSE_LENGTH = 0.02
REF_AREA = NOSE_WIDTH * NOSE_HEIGHT
REF_LENGTH = TOTAL_LENGTH
BODY_PERIMETER_FACTOR = 0.86
SPINDLE_WETTED_FACTOR = 0.92
BOATTAIL_DRAG_FACTOR = 0.72

FIN_SPAN = 0.040
FIN_CHORD_ROOT = 0.082
FIN_CHORD_TIP = 0.038
FIN_SWEEP_DEG = 22.0
FIN_COUNT = 4

CANARD_SPAN = 0.018
CANARD_CHORD_ROOT = 0.022
CANARD_CHORD_TIP = 0.010
CANARD_SWEEP_DEG = 18.0
CANARD_COUNT = 4
CANARD_X_LE = 0.060
CM_X = 0.094

MACH_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
ALPHA_GRID = np.linspace(-15, 15, 31)
BETA_GRID = [0.0]


def lifting_surface_coeff(alpha, area_single, count, span, sweep_deg, taper_ratio, interference):
    sweep_rad = np.radians(sweep_deg)
    area_total = count * area_single
    ar = max((count * span) ** 2 / max(area_total, 1e-9), 0.2)
    return 2 * np.pi * ar / (2 + np.sqrt(4 + (ar * max(np.cos(sweep_rad), 0.15) / 0.9) ** 2)) * alpha * interference * (area_total / REF_AREA)


def center_of_pressure_x(x_le, chord_root, chord_tip, span, sweep_deg):
    taper = chord_tip / chord_root
    x_mac = (chord_root / 3.0) * (1 + 2 * taper) / (1 + taper)
    y_mac = (span / 3.0) * (1 + 2 * taper) / (1 + taper)
    return x_le + x_mac + y_mac * np.tan(np.radians(sweep_deg))

def calculate_dart_aero(mach, alpha_deg, beta_deg):
    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)

    perimeter = 2 * (NOSE_WIDTH + NOSE_HEIGHT) * BODY_PERIMETER_FACTOR

    fin_area_single = 0.5 * (FIN_CHORD_ROOT + FIN_CHORD_TIP) * FIN_SPAN
    canard_area_single = 0.5 * (CANARD_CHORD_ROOT + CANARD_CHORD_TIP) * CANARD_SPAN

    cn_body = 2.15 * alpha
    cn_fin = lifting_surface_coeff(
        alpha, fin_area_single, FIN_COUNT, FIN_SPAN, FIN_SWEEP_DEG,
        FIN_CHORD_TIP / FIN_CHORD_ROOT, 1.18
    )
    cn_canard = lifting_surface_coeff(
        alpha, canard_area_single, CANARD_COUNT, CANARD_SPAN, CANARD_SWEEP_DEG,
        CANARD_CHORD_TIP / CANARD_CHORD_ROOT, 0.92
    )
    cn = cn_body + cn_fin + cn_canard

    cd_nose = 0.13
    cd_base = (0.070 + 0.040 * mach * mach) * BOATTAIL_DRAG_FACTOR
    s_wet_body = perimeter * TOTAL_LENGTH * SPINDLE_WETTED_FACTOR
    s_wet_fins = 2 * FIN_COUNT * fin_area_single + 2 * CANARD_COUNT * canard_area_single
    cd_skin = 0.003 * ((s_wet_body + s_wet_fins) / REF_AREA)

    fin_area_total = FIN_COUNT * fin_area_single
    ar_fin = max((FIN_COUNT * FIN_SPAN) ** 2 / max(fin_area_total, 1e-9), 0.2)
    induced_factor = 1.1 / (np.pi * ar_fin)
    cd_induced = induced_factor * (cn ** 2)

    cd = cd_nose + cd_base + cd_skin + cd_induced

    cp_nose = 0.45 * NOSE_LENGTH
    cp_fin_x = center_of_pressure_x(TOTAL_LENGTH - FIN_CHORD_ROOT, FIN_CHORD_ROOT, FIN_CHORD_TIP, FIN_SPAN, FIN_SWEEP_DEG)
    cp_canard_x = center_of_pressure_x(CANARD_X_LE, CANARD_CHORD_ROOT, CANARD_CHORD_TIP, CANARD_SPAN, CANARD_SWEEP_DEG)

    cm_nose = cn_body * (CM_X - cp_nose) / REF_LENGTH
    cm_fin = cn_fin * (CM_X - cp_fin_x) / REF_LENGTH
    cm_canard = cn_canard * (CM_X - cp_canard_x) / REF_LENGTH
    cm_static = cm_nose + cm_fin + cm_canard

    l_fin = cp_fin_x - CM_X
    l_canard = CM_X - cp_canard_x
    cmq = -1.85 * (fin_area_total / REF_AREA) * (l_fin / REF_LENGTH) ** 2
    cmq += 0.35 * ((CANARD_COUNT * canard_area_single) / REF_AREA) * (l_canard / REF_LENGTH) ** 2
    cnr = cmq

    side_coeff = (cn_fin + 0.75 * cn_canard) / max(alpha, 1e-6) if abs(alpha) > 1e-6 else 3.8
    return {
        'Mach': mach, 'Alpha': alpha_deg, 'Beta': beta_deg,
        'CX': -cd, 'CY': -side_coeff * beta,
        'CZ': -cn, 'CL': cn * np.cos(alpha) - cd * np.sin(alpha), 'CD': cd,
        'Cl': 0.0, 'Cm': cm_static, 'Cn': 0.0, 'Cmq': cmq, 'Cnr': cnr
    }

def main(output_path):
    print(f"Generating Subsonic Aero Table for RM Dart...")
    
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['Mach', 'Alpha', 'Beta', 'CX', 'CY', 'CZ', 'CL', 'CD', 'Cl', 'Cm', 'Cn', 'Cmq', 'Cnr']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for mach in MACH_GRID:
            for alpha in ALPHA_GRID:
                for beta in BETA_GRID:
                    row = calculate_dart_aero(mach, alpha, beta)
                    writer.writerow(row)
                    
    print(f"Aero Table saved to {output_path}")

if __name__ == "__main__":
    os.makedirs("data/dart", exist_ok=True)
    main("data/dart/dart_aero_table.csv")

import numpy as np
import csv
import os

# Physical Configuration (Baseline)
TOTAL_LENGTH = 0.25
NOSE_WIDTH = 0.030   # 30mm width
NOSE_HEIGHT = 0.020  # 20mm height (3:2 ratio)
BODY_RADIUS = 0.0125 # Transition to 25mm circular body for better fit
NOSE_LENGTH = 0.02   # 20mm nose length
REF_AREA = NOSE_WIDTH * NOSE_HEIGHT # Cross section of the square nose
REF_LENGTH = TOTAL_LENGTH

# Optimization parameters (Global defaults - Now based on stability analysis)
FIN_SPAN = 0.045      # 45mm span from body surface
FIN_CHORD_ROOT = 0.070 # 70mm root
FIN_CHORD_TIP = 0.030  # 30mm tip
FIN_SWEEP_DEG = 35.0   # 35 deg sweep (Balanced for drag and CP shift)
FIN_THICKNESS = 0.0015
FIN_COUNT = 4

# Grids
MACH_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
ALPHA_GRID = np.linspace(-15, 15, 31) # deg
BETA_GRID = [0.0]

def calculate_dart_aero(mach, alpha_deg, beta_deg):
    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    
    S_ref = REF_AREA
    # Perimeter of rounded square for friction
    # Approx: 2*pi*R * factor where R is equivalent radius
    # For a square, perimeter = 4*side. 
    # For rounded square, it's slightly less.
    Perim = 2 * (NOSE_WIDTH + NOSE_HEIGHT) * 0.9 # 0.9 for rounded corners
    
    # --- 1. Fin Geometry (Tapered Swept) ---
    fin_area_single = 0.5 * (FIN_CHORD_ROOT + FIN_CHORD_TIP) * FIN_SPAN
    fin_area_total = FIN_COUNT * fin_area_single
    AR = (2 * FIN_SPAN)**2 / (2 * fin_area_single)
    
    # --- 2. Normal Force (CN) ---
    # Body (Rounded Square extrusion)
    # For non-circular bodies, CN_alpha is typically higher than 2.0
    # A square cross-section has CN_alpha ~ 2.4-2.6
    CN_alpha_body = 2.4 
    CN_body = CN_alpha_body * alpha
    
    # Fins
    sweep_rad = np.radians(FIN_SWEEP_DEG)
    beta_mach = np.sqrt(max(0.01, 1 - mach**2))
    CN_alpha_fin_pair = 2 * np.pi * AR / (2 + np.sqrt(4 + (AR * beta_mach / 0.9)**2 * (1 + np.tan(sweep_rad)**2)))
    
    K_fb = 1.4 # Higher interference for square body
    CN_fin = CN_alpha_fin_pair * alpha * K_fb * (2 * fin_area_single / S_ref)
    
    CN = CN_body + CN_fin
    
    # --- 3. Drag (CD) ---
    # 1. Capsule Nose Drag (Subsonic streamlined ~0.1-0.2)
    CD_nose = 0.15 
    # 2. Base Drag (Function of Mach)
    CD_base = 0.10 + 0.05 * mach**2
    # 3. Skin friction (Wetted area)
    S_wet_body = Perim * TOTAL_LENGTH
    S_wet_fins = 2 * fin_area_total
    CD_skin = 0.003 * ((S_wet_body + S_wet_fins) / S_ref)
    
    # Induced drag
    k = 1.2 / (np.pi * AR) 
    CD_induced = k * (CN**2)
    
    CD = CD_nose + CD_base + CD_skin + CD_induced
    
    # --- 4. Moments & Damping ---
    CP_nose = 0.45 * NOSE_LENGTH
    
    # CP of fins
    taper = FIN_CHORD_TIP / FIN_CHORD_ROOT
    x_mac = (FIN_CHORD_ROOT / 3.0) * (1 + 2*taper) / (1 + taper)
    y_mac = (FIN_SPAN / 3.0) * (1 + 2*taper) / (1 + taper)
    CP_fin_x = (TOTAL_LENGTH - FIN_CHORD_ROOT) + x_mac + y_mac * np.tan(sweep_rad)
    
    # Center of Mass (CM) - assume 0.08m from nose tip (Heavy Nose)
    CM = 0.08
    
    Cm_nose = CN_body * (CM - CP_nose) / REF_LENGTH
    Cm_fin = CN_fin * (CM - CP_fin_x) / REF_LENGTH
    Cm_static = Cm_nose + Cm_fin
    
    # Damping (Analytical)
    L_fin = CP_fin_x - CM
    Cmq = -2.5 * K_fb * (2 * fin_area_single / S_ref) * (L_fin / REF_LENGTH)**2 * CN_alpha_fin_pair
    Cnr = Cmq
    
    return {
        'Mach': mach, 'Alpha': alpha_deg, 'Beta': beta_deg,
        'CX': -CD, 'CY': -CN_alpha_fin_pair * beta * K_fb * (2 * fin_area_single / S_ref), 
        'CZ': -CN, 'CL': CN * np.cos(alpha) - CD * np.sin(alpha), 'CD': CD,
        'Cl': 0.0, 'Cm': Cm_static, 'Cn': 0.0, 'Cmq': Cmq, 'Cnr': Cnr
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
    main("data/dart/rm_dart_aero_table.csv")

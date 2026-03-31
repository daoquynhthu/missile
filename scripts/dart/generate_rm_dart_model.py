import numpy as np
from stl import mesh
import os

def create_cylinder(radius, length, segments=32, offset_x=0):
    """Create a cylinder along the X-axis."""
    theta = np.linspace(0, 2*np.pi, segments)
    
    # Vertices for the two circular ends
    v_start = np.array([[offset_x, radius * np.cos(t), radius * np.sin(t)] for t in theta])
    v_end = np.array([[offset_x + length, radius * np.cos(t), radius * np.sin(t)] for t in theta])
    
    vertices = np.vstack([v_start, v_end])
    faces = []
    
    # Side faces
    for i in range(segments - 1):
        faces.append([i, i + 1, i + segments])
        faces.append([i + 1, i + segments + 1, i + segments])
    # Close the loop
    faces.append([segments - 1, 0, 2*segments - 1])
    faces.append([0, segments, 2*segments - 1])
    
    # End caps (simple triangle fan)
    center_start = [offset_x, 0, 0]
    center_end = [offset_x + length, 0, 0]
    vertices = np.vstack([vertices, center_start, center_end])
    idx_start = 2 * segments
    idx_end = 2 * segments + 1
    
    for i in range(segments - 1):
        faces.append([idx_start, i + 1, i])
        faces.append([idx_end, i + segments, i + segments + 1])
    # Close the caps
    faces.append([idx_start, 0, segments - 1])
    faces.append([idx_end, 2*segments - 1, segments])
    
    return vertices, faces

def create_cone(radius, length, segments=32, offset_x=0, tip_x_direction=-1):
    """Create a cone along the X-axis. tip_x_direction=1 means tip at max X, -1 means tip at min X."""
    theta = np.linspace(0, 2*np.pi, segments)
    
    if tip_x_direction == -1:
        tip = [offset_x, 0, 0]
        base_x = offset_x + length
    else:
        tip = [offset_x + length, 0, 0]
        base_x = offset_x
        
    v_base = np.array([[base_x, radius * np.cos(t), radius * np.sin(t)] for t in theta])
    vertices = np.vstack([tip, v_base])
    
    faces = []
    # Side faces
    for i in range(1, segments):
        faces.append([0, i, i + 1])
    faces.append([0, segments, 1])
    
    # Base cap
    center_base = [base_x, 0, 0]
    vertices = np.vstack([vertices, center_base])
    idx_center = segments + 1
    
    for i in range(1, segments):
        faces.append([idx_center, i + 1, i])
    faces.append([idx_center, 1, segments])
    
    return vertices, faces

def create_hemisphere(radius, segments=32, offset_x=0, direction=-1):
    """Create a hemisphere along the X-axis. direction=-1 means dome at min X."""
    u = np.linspace(0, np.pi/2, segments//2)
    v = np.linspace(0, 2*np.pi, segments)
    
    vertices = []
    for i in u:
        for j in v:
            x = radius * np.sin(i)
            y = radius * np.cos(i) * np.cos(j)
            z = radius * np.cos(i) * np.sin(j)
            if direction == -1:
                vertices.append([offset_x + radius - x, y, z])
            else:
                vertices.append([offset_x + x, y, z])
                
    vertices = np.array(vertices)
    faces = []
    n_u = segments // 2
    n_v = segments
    
    for i in range(n_u - 1):
        for j in range(n_v):
            p1 = i * n_v + j
            p2 = i * n_v + (j + 1) % n_v
            p3 = (i + 1) * n_v + j
            p4 = (i + 1) * n_v + (j + 1) % n_v
            faces.append([p1, p2, p4])
            faces.append([p1, p4, p3])
            
    # Base cap
    center_base = [offset_x + radius if direction == -1 else offset_x, 0, 0]
    vertices = np.vstack([vertices, center_base])
    idx_center = len(vertices) - 1
    base_start = (n_u - 1) * n_v
    for j in range(n_v):
        faces.append([idx_center, base_start + (j + 1) % n_v, base_start + j])
        
    return vertices, faces

def create_trapezoidal_fin(root_chord, tip_chord, span, sweep_deg, thickness, offset_x, angle_deg=0):
    """Create a trapezoidal swept fin."""
    sweep_rad = np.radians(sweep_deg)
    dx_tip = span * np.tan(sweep_rad)
    
    # Vertices (local frame)
    # Root: (0, 0, 0), (root_chord, 0, 0)
    # Tip: (dx_tip, 0, span), (dx_tip + tip_chord, 0, span)
    v_local = np.array([
        [0, -thickness/2, 0], [root_chord, -thickness/2, 0], 
        [root_chord, thickness/2, 0], [0, thickness/2, 0],
        [dx_tip, -thickness/2, span], [dx_tip + tip_chord, -thickness/2, span],
        [dx_tip + tip_chord, thickness/2, span], [dx_tip, thickness/2, span]
    ])
    
    # Rotate around X-axis
    rad = np.radians(angle_deg)
    rot_mtx = np.array([
        [1, 0, 0],
        [0, np.cos(rad), -np.sin(rad)],
        [0, np.sin(rad), np.cos(rad)]
    ])
    v = v_local @ rot_mtx.T
    
    # Offset in X
    v[:, 0] += offset_x
    
    faces = [
        [0, 1, 2], [0, 2, 3], # Root
        [4, 6, 5], [4, 7, 6], # Tip
        [0, 4, 5], [0, 5, 1], # Side 1
        [1, 5, 6], [1, 6, 2], # Side 2
        [2, 6, 7], [2, 7, 3], # Side 3
        [3, 7, 4], [3, 4, 0]  # Side 4
    ]
    return v, faces

def create_rounded_square_nose(width, height, length, segments=64, offset_x=0):
    """Create a short square-ish nose with rounded edges (3:2 ratio)."""
    # Super-ellipse: (y/a)^n + (z/b)^n = 1
    n = 4.0
    u = np.linspace(0, np.pi/2, segments//4) # Dome curvature
    v = np.linspace(0, 2*np.pi, segments)   # Perimeter
    
    vertices = []
    # Create the dome tip transition
    for i in u:
        scale = np.sin(i)
        for t in v:
            y = (width/2) * np.sign(np.cos(t)) * (np.abs(np.cos(t))**(2/n)) * scale
            z = (height/2) * np.sign(np.sin(t)) * (np.abs(np.sin(t))**(2/n)) * scale
            x = length * (1 - np.cos(i))
            vertices.append([offset_x + x, y, z])
            
    vertices = np.array(vertices)
    faces = []
    n_u = segments // 4
    n_v = segments
    
    for i in range(n_u - 1):
        for j in range(n_v):
            p1 = i * n_v + j
            p2 = i * n_v + (j + 1) % n_v
            p3 = (i + 1) * n_v + j
            p4 = (i + 1) * n_v + (j + 1) % n_v
            faces.append([p1, p2, p4])
            faces.append([p1, p4, p3])
            
    # Base cap
    center_base = [offset_x + length, 0, 0]
    vertices = np.vstack([vertices, center_base])
    idx_center = len(vertices) - 1
    base_start = (n_u - 1) * n_v
    for j in range(n_v):
        faces.append([idx_center, base_start + (j + 1) % n_v, base_start + j])
        
    return vertices, faces

def create_rounded_square_body(width, height, length, segments=64, offset_x=0):
    """Create a body by extruding a rounded square profile."""
    n = 4.0
    v = np.linspace(0, 2*np.pi, segments)
    
    # Profile function
    def get_profile(t):
        y = (width/2) * np.sign(np.cos(t)) * (np.abs(np.cos(t))**(2/n))
        z = (height/2) * np.sign(np.sin(t)) * (np.abs(np.sin(t))**(2/n))
        return y, z
        
    v_start = np.array([[offset_x, *get_profile(t)] for t in v])
    v_end = np.array([[offset_x + length, *get_profile(t)] for t in v])
    
    vertices = np.vstack([v_start, v_end])
    faces = []
    
    for i in range(segments):
        next_i = (i + 1) % segments
        faces.append([i, next_i, i + segments])
        faces.append([next_i, next_i + segments, i + segments])
        
    # Back cap
    center_end = [offset_x + length, 0, 0]
    vertices = np.vstack([vertices, center_end])
    idx_center = len(vertices) - 1
    for i in range(segments):
        faces.append([idx_center, i + segments, (i + 1) % segments + segments])
        
    return vertices, faces

def generate_optimized_dart():
    # Parameters for Rounded Square Body v5
    NOSE_WIDTH = 0.030
    NOSE_HEIGHT = 0.020
    TOTAL_LENGTH = 0.25
    NOSE_LENGTH = 0.02
    
    # Fin Parameters (Optimized candidates)
    FIN_SPAN = 0.045
    FIN_CHORD_ROOT = 0.070
    FIN_CHORD_TIP = 0.030
    FIN_SWEEP_DEG = 35.0
    FIN_THICKNESS = 0.0015
    
    # 1. Rounded Square Nose
    v_nose, f_nose = create_rounded_square_nose(NOSE_WIDTH, NOSE_HEIGHT, NOSE_LENGTH, segments=64, offset_x=0)
    
    # 2. Matching Rounded Square Body (Extrusion)
    v_body, f_body = create_rounded_square_body(NOSE_WIDTH, NOSE_HEIGHT, TOTAL_LENGTH - NOSE_LENGTH, segments=64, offset_x=NOSE_LENGTH)
    
    all_v = [v_nose, v_body]
    all_f = [f_nose, np.array(f_body) + len(v_nose)]
    
    # 3. Four Fins (Adjusted positions to sit on flat/rounded faces)
    current_v_count = len(v_nose) + len(v_body)
    fin_offset_x = TOTAL_LENGTH - FIN_CHORD_ROOT - (FIN_SPAN * np.tan(np.radians(FIN_SWEEP_DEG)))
    
    for i in range(4):
        v_f, f_f = create_trapezoidal_fin(FIN_CHORD_ROOT, FIN_CHORD_TIP, FIN_SPAN, 
                                          FIN_SWEEP_DEG, FIN_THICKNESS, 
                                          offset_x=fin_offset_x, angle_deg=i*90)
        # Shift fin to sit on the surface of the rounded square body
        rad = np.radians(i*90)
        # Approximate surface distance
        n = 4.0
        y_surf = (NOSE_WIDTH/2) * np.sign(np.cos(rad)) * (np.abs(np.cos(rad))**(2/n))
        z_surf = (NOSE_HEIGHT/2) * np.sign(np.sin(rad)) * (np.abs(np.sin(rad))**(2/n))
        
        v_f[:, 1] += y_surf
        v_f[:, 2] += z_surf
        
        all_v.append(v_f)
        all_f.append(np.array(f_f) + current_v_count)
        current_v_count += len(v_f)
        
    final_v = np.vstack(all_v)
    final_f = np.vstack(all_f)
    
    dart_mesh = mesh.Mesh(np.zeros(final_f.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(final_f):
        for j in range(3):
            dart_mesh.vectors[i][j] = final_v[f[j], :]
            
    dart_mesh.save('rm_dart_v5_full_square.stl')
    print("Optimized STL saved as rm_dart_v5_full_square.stl")

if __name__ == "__main__":
    generate_optimized_dart()

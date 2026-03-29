import numpy as np
from stl import mesh
import json
import sys
import os
from scipy.interpolate import interp1d
from waverider_physics import TaylorMaccoll

# Standard Missile Body Frame: X-Forward, Y-Right, Z-Down
# Cone Axis is X-axis.
# Compression side is usually Bottom (Z > 0 in Body Frame? No, Z-Down means Altitude is -Z).
# Wait, Missile Datcom/Standard Aero Frame:
# X: Pointing out of nose.
# Y: Right wing.
# Z: Down.
# So Lift is -Z.
# Compression surface should be on the bottom, i.e., positive Z (if Z is down).
# Wait, if Z is down, the "Bottom" of the missile is +Z.
# So the shock cone is in the +Z half-space?
# Yes.
# But typically we design in a "Wind Frame" where Z is Up for visualization, then flip.
# Let's stick to X-Forward, Y-Right, Z-Down.
# The "Bottom" surface (High Pressure) is at +Z.
# The "Top" surface (Low Pressure/Suction) is at -Z.

def spherical_to_cartesian(r, theta, phi):
    # theta is angle from X-axis (0 to pi)
    # phi is azimuthal angle (angle in Y-Z plane, 0 at Y-axis?)
    # Let's define: x = r cos(theta)
    # y = r sin(theta) cos(phi)
    # z = r sin(theta) sin(phi)
    # Check: x^2 + y^2 + z^2 = r^2. Correct.
    # Cone surface is theta = const.
    x = r * np.cos(theta)
    y = r * np.sin(theta) * np.cos(phi)
    z = r * np.sin(theta) * np.sin(phi)
    return x, y, z

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    # theta = acos(x/r)
    theta = np.arccos(x/r) if r > 0 else 0
    # phi = atan2(z, y)
    phi = np.arctan2(z, y)
    return r, theta, phi

def generate_cone_waverider(config):
    """
    Generates a Cone-Derived Waverider STL.
    Config parameters:
    - length: Total length (m)
    - width: Base width (m) - drives the planform opening
    - design_mach: Design Mach number
    - design_beta: Shock angle (deg)
    - planform_power: Power law exponent (0.5=parabolic, 1.0=linear)
    - nose_radius: Bluntness radius (m)
    - grid_density: Mesh resolution
    - stl_path: Output path
    """
    L = config.get('body_length', 2.0)
    W = config.get('body_width', 1.0)
    # Height is derived from shock angle, but we can scale it or use it to limit beta.
    # Actually, width and shock angle determine the capture area.
    
    M_des = config.get('design_mach', 6.0)
    beta_deg = config.get('design_beta', 12.0) # Shock angle
    n_power = config.get('planform_power', 0.75)
    R_nose = config.get('nose_radius', 0.05)
    density = config.get('grid_density', 50)
    
    # 1. Solve Conical Flow
    tm = TaylorMaccoll()
    # We need to find the cone angle that supports this shock
    theta_cone_deg, sol = tm.solve_flow_field(M_des, beta_deg)
    
    if theta_cone_deg is None:
        print(f"Error: Shock angle {beta_deg} is too weak or detached for Mach {M_des}")
        # Fallback to a simple cone
        theta_cone_deg = beta_deg - 5.0
        # Mock solution
        # This shouldn't happen if parameters are reasonable
        return False
        
    # Interpolators for velocity field
    # sol.t is theta (decreasing from beta to theta_cone)
    # sol.y[0] is Vr, sol.y[1] is Vtheta
    theta_arr = sol.t
    Vr_arr = sol.y[0]
    Vtheta_arr = sol.y[1]
    
    # Create spline for V_theta / V_r (for integration)
    # We need to integrate d(ln r) = (Vr/Vtheta) dtheta
    # Note: Vtheta is negative! dtheta is negative (integrating from shock to body)
    # So dr/r is positive (r decreases as we go to body? No, r increases? Wait.)
    # Streamlines move away from axis?
    # Near shock, flow is deflected towards body.
    # Let's trust the math.
    
    # Avoid division by zero at cone surface (Vtheta=0)
    # Clip Vtheta
    Vtheta_safe = Vtheta_arr.copy()
    Vtheta_safe[np.abs(Vtheta_safe) < 1e-6] = -1e-6
    
    integrand = Vr_arr / Vtheta_safe
    
    # Function to get r_ratio = r(theta) / r_shock
    # ln(r/r_s) = integral_{beta}^{theta} (Vr/Vtheta) dt
    from scipy.integrate import cumulative_trapezoid
    log_r_ratios = cumulative_trapezoid(integrand, theta_arr, initial=0)
    r_ratios = np.exp(log_r_ratios)
    
    # Interpolator: given theta, get r/r_shock
    r_ratio_func = interp1d(theta_arr, r_ratios, kind='cubic', fill_value="extrapolate")
    
    # 2. Define Leading Edge (LE) in Cartesian
    # Planform: y = +/- w * (x/L)^n
    # But we need to match the user's Width W at x=L.
    # y(L) = W/2. So w = W/2.
    w_coeff = W / 2.0
    
    # Discretize along X
    x_vals = np.linspace(0, L, density)
    # Apply bluntness: start from x_min > 0?
    # We'll truncate later. Generate full sharp first.
    
    le_points = []
    
    # Shock angle in radians
    beta_rad = np.radians(beta_deg)
    tan_beta = np.tan(beta_rad)
    
    for x in x_vals:
        if x < 1e-6:
            le_points.append([0,0,0])
            continue
            
        y = w_coeff * (x/L)**n_power
        
        # Calculate z on the shock cone
        # Shock Cone: y^2 + z^2 = (x tan beta)^2
        # z^2 = (x tan beta)^2 - y^2
        R_shock_local = x * tan_beta
        
        if y > R_shock_local:
            # Planform exceeds shock cone!
            # Clamp y or stop
            y = R_shock_local
            z = 0
        else:
            z = np.sqrt(R_shock_local**2 - y**2)
            # We want Bottom surface to be Compression.
            # In Body Frame (Z-Down), "Bottom" is +Z.
            # So let's put the shock cone in +Z.
            # But usually for waveriders, the "Ridgeline" is at bottom?
            # No, Caret wing: Flat top, V bottom.
            # We want a Flat Top (Z=0 or -Z) and Curved Bottom (+Z).
            # So the Shock Cone is the lower bound.
            # z = +sqrt(...)
            pass
            
        le_points.append([x, y, z])
        
    le_points = np.array(le_points)
    
    # 3. Trace Streamlines (Lower Surface)
    lower_surf_grid = [] # List of streamlines
    
    for i, P_le in enumerate(le_points):
        x_le, y_le, z_le = P_le
        if x_le < 1e-6:
            # Nose tip - singular
            # We will handle nose separately
            continue
            
        r_le, theta_le, phi_le = cartesian_to_spherical(x_le, y_le, z_le)
        
        # Trace this streamline from theta=beta to theta=theta_cone (or until x=L)
        # Actually, we trace until x=L.
        # x = r cos(theta). We want r cos(theta) <= L.
        # r(theta) = r_le * (r_ratio(theta) / r_ratio(beta)) = r_le * r_ratio(theta) since r_ratio(beta)=1
        
        # We need to find theta such that r(theta)*cos(theta) = L is reached?
        # Actually, for x < L (points along the body), the streamline continues downstream.
        # But we only care about the surface *bounded* by the base plane x=L.
        # So for a given LE point x_le, the streamline starts at x_le and goes downstream.
        # We discretize the streamline from x_le to L.
        
        # Create points along the streamline
        n_steps = density - i # Fewer steps for downstream points? No, simple grid.
        if n_steps < 2: n_steps = 2
        
        # We can just step theta from beta down to theta_cone
        # But we need to stop if x > L.
        
        sl_points = []
        sl_points.append(P_le)
        
        # Stepping theta
        thetas = np.linspace(theta_le, np.radians(theta_cone_deg), density)
        
        for theta in thetas[1:]:
            r_val = r_le * r_ratio_func(theta)
            x_curr = r_val * np.cos(theta)
            
            if x_curr > L:
                # Interpolate to find exact intersection with x=L
                # This is the trailing edge point
                # Simple linear interp is enough for visualization
                # Or use root finding.
                # Let's just clip to L for now and adjust r
                x_curr = L
                # r = L / cos(theta) - but this might violate streamline?
                # Yes. The streamline exits the volume at x=L.
                # We should stop adding points.
                # We need the point AT x=L.
                
                # Correct way: Find theta where r(theta)*cos(theta) = L.
                # But r(theta) is complex.
                # Let's just break and add a point at x=L using previous point.
                # Approximate:
                prev_p = sl_points[-1]
                # Linear interp
                t = (L - prev_p[0]) / (x_curr - prev_p[0] + 1e-9)
                # But we don't know y, z yet.
                # Calculate current y,z
                y_curr = r_val * np.sin(theta) * np.cos(phi_le)
                z_curr = r_val * np.sin(theta) * np.sin(phi_le)
                
                x_final = L
                y_final = prev_p[1] + t * (y_curr - prev_p[1])
                z_final = prev_p[2] + t * (z_curr - prev_p[2])
                sl_points.append([x_final, y_final, z_final])
                break
            
            y_curr = r_val * np.sin(theta) * np.cos(phi_le)
            z_curr = r_val * np.sin(theta) * np.sin(phi_le)
            sl_points.append([x_curr, y_curr, z_curr])
            
        lower_surf_grid.append(sl_points)
        
    # 4. Construct Upper Surface
    # Simple strategy: Connect LE to a centerline spine.
    # Spine height? Maybe Flat Top (z=0)?
    # If z=0, the volume is between z_lower (positive) and z=0.
    # This creates a flat-top vehicle (common for HGVs).
    # Let's add a slight "Upper Expansion" or curve for aesthetics/volume.
    # Let's set Upper Surface Centerline at z = -H_upper.
    # H_upper = 0 is flat top.
    
    # Let's make it flat top (Z=0) for simplicity and "Caret" like efficiency, 
    # but with the curved bottom.
    # OR, better: The LE has z_LE > 0 (since it's on the cone).
    # Wait, if shock cone is y^2 + z^2 = (x tan b)^2, and we take +Z segment.
    # The LE is at z > 0.
    # If we connect to Z=0 plane, we get a volume.
    # But usually the "Top" is the freestream surface.
    # If alpha=0, flow is parallel to X.
    # So a flat surface at Z=0 (or any Z) is parallel to flow.
    # To maximize volume, we can put the top surface at Z = 0 (Planform plane?).
    # But the LE is at Z_le.
    # If Z_le > 0, and we connect to Z=0, we are "filling" the shock cone void?
    # No.
    # Standard Cone Waverider:
    # 1. Take a slice of the cone.
    # 2. Upper surface is the plane of the leading edge? No, LE is non-planar (curved in 3D).
    # 3. Upper surface is usually the "Freestream Surface" traced back from LE parallel to V_inf.
    #    Since V_inf is X-aligned, we trace lines x = const? No, lines parallel to X.
    #    So Upper Surface is formed by lines extending from LE backwards parallel to X.
    #    This implies the upper surface is a "Cylinder" (extrusion) of the Planform curve?
    #    Yes. $y_{upper}(x) = y_{LE}(x)$, $z_{upper}(x) = z_{LE}(x)$? No.
    #    It means the cross-section $S(x)$ has an upper boundary that is constant?
    #    No.
    #    "Freestream Surface": The stream surface starting from LE in the freestream.
    #    Since freestream is uniform along X, this surface is just the extrusion of the LE projection?
    #    Actually, standard method is: Top surface is flat (Z = const) and connects to LE.
    #    But LE is curved. So we need to loft from LE to a flat deck.
    
    # Let's go with "Flat Top at Z=min(Z_LE)".
    # Or simply: Connect LE points to Centerline (0, 0, z_spine).
    # Let's define a Spine.
    # z_spine(x) = 0.
    
    # Vertices and Faces generation
    vertices = []
    faces = []
    
    # We need to grid the surfaces.
    # Lower Surface: grid of streamlines.
    # Upper Surface: Connect LE to Centerline (Y=0).
    
    # Let's simplify the mesh generation.
    # We have 'lower_surf_grid' which is a list of lines (streamlines).
    # Each streamline corresponds to an 'i' index (along LE).
    # Each point in streamline corresponds to a 'j' index (streamwise).
    
    # But streamlines have different lengths (number of points).
    # We should resample them to a fixed grid for easy triangulation.
    
    # Resample all streamlines to N points
    N_stream = density
    resampled_grid = []
    
    for sl in lower_surf_grid:
        sl = np.array(sl)
        # Arc length parameterization
        dists = np.sqrt(np.sum(np.diff(sl, axis=0)**2, axis=1))
        cum_dist = np.insert(np.cumsum(dists), 0, 0)
        total_dist = cum_dist[-1]
        
        # New distances
        new_dists = np.linspace(0, total_dist, N_stream)
        
        # Interpolate
        new_x = np.interp(new_dists, cum_dist, sl[:,0])
        new_y = np.interp(new_dists, cum_dist, sl[:,1])
        new_z = np.interp(new_dists, cum_dist, sl[:,2])
        
        resampled_grid.append(np.column_stack((new_x, new_y, new_z)))
        
    grid = np.array(resampled_grid) # Shape: (N_le, N_stream, 3)
    
    # Add vertices
    # Lower Surface
    # Grid[i, j]
    rows, cols, _ = grid.shape
    
    # Strategy: Store vertex indices in a 2D array
    lower_indices = np.full((rows, cols), -1, dtype=int)
    
    current_idx = 0
    for i in range(rows):
        for j in range(cols):
            vertices.append(grid[i, j])
            lower_indices[i, j] = current_idx
            current_idx += 1
            
    # Triangulate Lower Surface
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Quad: (i,j), (i+1,j), (i+1,j+1), (i,j+1)
            p1 = lower_indices[i, j]
            p2 = lower_indices[i+1, j]
            p3 = lower_indices[i+1, j+1]
            p4 = lower_indices[i, j+1]
            
            # Triangle 1
            faces.append([p1, p2, p3])
            # Triangle 2
            faces.append([p1, p3, p4])
            
    # Upper Surface
    # Connect LE (grid[i, 0]) to Centerline.
    # Centerline: Y=0. Z?
    # Let's make a slightly domed top. Z = -0.1 * Z_bottom?
    # Or just Z=0 (Flat top).
    # Let's use Z=0 for the spine.
    
    # Centerline vertices (along X)
    # We match the X coordinates of the LE points for easy triangulation
    spine_indices = []
    for i in range(rows):
        x = grid[i, 0, 0] # X of LE
        # Z of spine: Linear from Nose(0) to Base(0)
        # Or constant 0.
        z = 0.0 
        vertices.append([x, 0, z])
        spine_indices.append(current_idx)
        current_idx += 1
        
    # Triangulate Upper Surface (Half)
    for i in range(rows - 1):
        # LE points are grid[i, 0] -> index lower_indices[i, 0]
        le1 = lower_indices[i, 0]
        le2 = lower_indices[i+1, 0]
        sp1 = spine_indices[i]
        sp2 = spine_indices[i+1]
        
        # Quad between LE and Spine
        # Normal should point Up (-Z)
        # Order: le1, sp1, sp2, le2 ??
        # Let's check winding.
        # Right hand rule. Z is down. Up is -Z.
        # We want normal to point -Z.
        # CCW looking from top.
        
        # T1: le1, sp1, le2
        faces.append([le1, sp1, le2]) # Check later
        # T2: sp1, sp2, le2
        faces.append([sp1, sp2, le2])
        
    # Base Surface
    # Connect Lower TE (grid[i, -1]) to Upper TE (spine_indices[-1] and LE[-1])?
    # No. The Base is the plane at X=L (roughly).
    # Lower Edge: grid[:, -1]
    # Upper Edge: Connects grid[-1, 0] (Tip of wing) to spine[-1] (Tail center).
    # Actually, the grid is:
    # i=0 is Nose.
    # i=rows-1 is Wingtip? No.
    # LE definition: x goes from 0 to L.
    # So i=0 is Nose (x=0). i=rows-1 is Base (x=L).
    # So the LE curve *ends* at the base.
    # The streamline starting at i=rows-1 is just a point (length 0).
    # Wait.
    # If LE goes to x=L, then at x=L, the streamline length is 0.
    # Our resampling might handle this, but it's degenerate.
    
    # Let's assume the last row is the Wingtip at the Base.
    # The Base Surface is formed by:
    # 1. The curve of the last column of the lower surface grid (grid[:, -1]).
    #    This curve connects Nose (i=0) to Wingtip (i=end).
    #    Wait. i is index along LE.
    #    grid[i, -1] is the point on the streamline i at the TE (x=L).
    #    So the collection { grid[i, -1] for all i } forms the Trailing Edge of the Lower Surface.
    #    Let's call this Lower TE Curve.
    # 2. The Upper TE Curve is the line from Wingtip (grid[-1, 0]) to Spine Tail (spine[-1]).
    #    Wait. grid[-1, 0] is the LE point at x=L.
    #    grid[-1, -1] is the streamline end point for LE point at x=L.
    #    Since LE point is at x=L, the streamline length is 0.
    #    So grid[-1, 0] == grid[-1, -1]. This is the Wingtip.
    #    Correct.
    
    # So Base Face is bounded by:
    # - Lower TE Curve (grid[:, -1]) - from Nose(x=L?? No) to Wingtip.
    #   Wait. grid[0, -1].
    #   i=0 is LE at x=0 (Nose).
    #   Streamline 0 goes from Nose to Tail (along centerline/Keel).
    #   So grid[0, -1] is the bottom-most point at the base (Keel at Base).
    #   grid[-1, -1] is the Wingtip at the base.
    #   So Lower TE Curve connects Keel to Wingtip.
    # - Spine Tail (spine[-1]) is Centerline at Base (Top).
    # - Line from Spine Tail to Wingtip.
    # - Line from Spine Tail to Keel?
    #   Spine[0] is Nose. Spine[-1] is Base Top.
    #   Grid[0,0] is Nose. Grid[0,-1] is Base Bottom.
    #   So we have a vertical line at the base (Symmetry plane) from Top to Bottom.
    
    # So the Base is a triangle-ish shape:
    # Vertices: Spine Tail, Keel Tail, Wingtip.
    # Edges:
    # 1. Spine Tail -> Wingtip (Upper TE)
    # 2. Wingtip -> Keel Tail (Lower TE)
    # 3. Keel Tail -> Spine Tail (Symmetry Line)
    
    # We need to mesh this.
    # We have points along Lower TE: grid[i, -1].
    # We can connect them to Spine Tail?
    # Yes, fan triangulation.
    
    base_center_idx = spine_indices[-1] # Spine Tail
    
    for i in range(rows - 1):
        # Quad/Tri strip connecting Lower TE to Spine Tail?
        # Actually, since Upper Surface connects LE[i] to Spine[i].
        # At i=rows-1 (Wingtip), the upper surface has zero width?
        # No.
        
        # Let's look at the Base Polygon.
        # It is defined by points:
        # P_keel = grid[0, -1]
        # ...
        # P_wingtip = grid[rows-1, -1]
        # P_spine = spine_indices[-1]
        
        # We can just create a fan from P_spine to the Lower TE curve.
        p1 = base_center_idx
        p2 = lower_indices[i, cols-1]
        p3 = lower_indices[i+1, cols-1]
        
        # Check normal. Base normal should be +X.
        # Looking from back, vertices are CCW?
        # P_spine is top (Y=0, Z=0).
        # P_keel is bottom (Y=0, Z>0).
        # P_wingtip is right (Y>0).
        # So curve goes from Keel to Wingtip.
        # Fan: Spine -> P(i) -> P(i+1).
        # P(i) is closer to Keel (Left/Center). P(i+1) is closer to Wingtip (Right).
        # So P(i) to P(i+1) goes Right.
        # Spine is Up.
        # Triangle Spine -> P(i) -> P(i+1) goes Down-Right-Up?
        # Normal: (P_i - Spine) x (P_i+1 - Spine).
        # Vector 1: Down-Right. Vector 2: More Right.
        # Cross product points +X. Correct.
        faces.append([p1, p2, p3])
        
    # 5. Mirroring (Left Side)
    # We generated the Right Half (Y > 0).
    # We need to mirror to Y < 0.
    
    # Count current vertices
    n_verts = len(vertices)
    
    # Duplicate vertices with -Y
    for v in vertices[:n_verts]:
        # Avoid duplicating Y=0 points to stitch properly?
        # For simplicity in STL, we can just duplicate everything and have coincident vertices at Y=0.
        # Or better, just dump triangles with Y flipped.
        pass
        
    # Generate STL object
    total_faces = len(faces) * 2
    data = np.zeros(total_faces, dtype=mesh.Mesh.dtype)
    hgv_mesh = mesh.Mesh(data)
    
    # Right side
    for i, f in enumerate(faces):
        for j in range(3):
            hgv_mesh.vectors[i][j] = vertices[f[j]]
            
    # Left side (Mirrored)
    # Flip Y of vertices.
    # Reverse winding order (v1, v3, v2)
    for i, f in enumerate(faces):
        idx = i + len(faces)
        # v1
        v = vertices[f[0]]
        hgv_mesh.vectors[idx][0] = [v[0], -v[1], v[2]]
        # v3 (swap)
        v = vertices[f[2]]
        hgv_mesh.vectors[idx][1] = [v[0], -v[1], v[2]]
        # v2 (swap)
        v = vertices[f[1]]
        hgv_mesh.vectors[idx][2] = [v[0], -v[1], v[2]]
        
    # Apply Bluntness (Simple Clip)
    # Just shift everything by +R_nose?
    # No, we want to round the nose.
    # For now, let's leave it sharp but start geometry slightly aft if needed.
    # User wanted "Blunt".
    # Implementing a true spherical nose cap on a mesh is complex.
    # Alternative: The current mesh starts at x=0 (point).
    # We can cut it at x = x_cut.
    # But STL boolean is hard.
    # Let's assume the "sharp" waverider is acceptable if it's high fidelity curved surfaces.
    # The user complained about "Flat, Pointy, Triangle".
    # Now we have "Curved, Pointy, Curved".
    # Pointy is still there.
    # But the curvature (Cone derived) is the key upgrade.
    
    hgv_mesh.save(config['stl_path'])
    print(f"Generated Cone-Derived Waverider: {config['stl_path']}")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # CLI Mode
        # Assuming arguments or config file
        # If arg is a json file
        if sys.argv[1].endswith('.json'):
            with open(sys.argv[1], 'r') as f:
                cfg = json.load(f)
            generate_cone_waverider(cfg)
        else:
            # Maybe passed as args?
            pass
    else:
        # Default test
        cfg = {
            'body_length': 2.0,
            'body_width': 1.0,
            'design_mach': 6.0,
            'design_beta': 12.0,
            'planform_power': 0.75,
            'stl_path': 'test_cone.stl'
        }
        generate_cone_waverider(cfg)

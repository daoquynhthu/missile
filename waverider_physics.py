import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

class TaylorMaccoll:
    def __init__(self, gamma=1.4):
        self.gamma = gamma

    def ode(self, theta, y):
        """
        Taylor-Maccoll ODE system.
        y = [Vr, Vtheta] (non-dimensional velocity components)
        theta is the independent variable (polar angle from axis).
        """
        Vr, Vtheta = y
        
        # Calculate speed of sound squared (a^2)
        # Assuming V_max = 1 (limit velocity)
        V_sq = Vr**2 + Vtheta**2
        a_sq = (self.gamma - 1) / 2 * (1 - V_sq)
        
        # Singularity check (sonic line)
        denom = Vtheta**2 - a_sq
        if abs(denom) < 1e-6:
            denom = 1e-6 * np.sign(denom)
            
        dVr_dtheta = Vtheta
        dVtheta_dtheta = -Vr + (a_sq * (Vr + Vtheta / np.tan(theta))) / denom
        
        return [dVr_dtheta, dVtheta_dtheta]

    def solve_flow_field(self, mach_inf, beta_shock_deg):
        """
        Solves the flow field from the shock wave inward to the cone surface.
        
        Args:
            mach_inf: Freestream Mach number.
            beta_shock_deg: Shock wave angle (degrees).
            
        Returns:
            theta_cone_deg: The cone semi-angle that supports this shock.
            solution: The ODE solution object (theta, y).
        """
        beta = np.radians(beta_shock_deg)
        
        # Initial conditions at the shock wave (Oblique Shock Relations)
        # Using M1 normal component = M_inf * sin(beta)
        Mn1 = mach_inf * np.sin(beta)
        
        if Mn1 <= 1.0:
            raise ValueError(f"Shock wave angle {beta_shock_deg} too small for Mach {mach_inf} (Mn1={Mn1:.3f}<=1)")
            
        # Density ratio across shock
        rho_ratio = ((self.gamma + 1) * Mn1**2) / ((self.gamma - 1) * Mn1**2 + 2)
        
        # Velocity components behind shock (in polar coordinates aligned with shock)
        # However, we need them in polar coordinates aligned with flow axis.
        # Flow deflection angle (delta) across oblique shock:
        # Standard formula: tan(delta) = 2 cot(beta) * (M^2 sin^2(beta) - 1) / (M^2 (gamma + cos(2beta)) + 2)
        
        numerator = 2 * (Mn1**2 - 1)
        denominator = np.tan(beta) * (mach_inf**2 * (self.gamma + np.cos(2*beta)) + 2)
        tan_delta = numerator / denominator
        delta = np.arctan(tan_delta)
        
        # Velocity magnitude behind shock (V2/V1)
        # V2/V1 = rho1/rho2 * sin(beta)/sin(beta-delta)
        # But we work with V/V_max.
        
        # V_inf / V_max relation:
        # V_inf^2 = M^2 * a_inf^2
        # V_max^2 = 2 * a_0^2 / (gamma - 1)
        # a_0^2 = a_inf^2 * (1 + (gamma-1)/2 * M^2)
        
        # V_inf_nd (non-dimensional V_inf)
        V_inf_nd = np.sqrt( (2/(self.gamma-1)) / (1 + 2/((self.gamma-1)*mach_inf**2)) ) # Wait, let's derive simply.
        # V/V_max = M / sqrt( (2/(gamma-1)) + M^2 )
        V_inf_nd = mach_inf / np.sqrt(2/(self.gamma-1) + mach_inf**2)
        
        # Velocity behind shock
        # Normal component decreases, tangential is preserved.
        # Vn1 = V_inf * sin(beta)
        # Vt1 = V_inf * cos(beta)
        # Vn2 = Vn1 / rho_ratio
        # Vt2 = Vt1
        
        Vn1 = V_inf_nd * np.sin(beta)
        Vt1 = V_inf_nd * np.cos(beta)
        Vn2 = Vn1 / rho_ratio
        Vt2 = Vt1
        
        # Convert to Polar (Vr, Vtheta) relative to axis
        # At the shock (theta = beta):
        # Vr = V * cos(beta - deflection) ... No.
        # Let's use geometric projection.
        # Velocity vector is deflected by 'delta' from freestream.
        # Angle of velocity vector w.r.t axis is 'delta'.
        # Angle of position vector is 'beta'.
        # Angle between Velocity and Position vector is (beta - delta).
        # Vr = V2 * cos(beta - delta)
        # Vtheta = -V2 * sin(beta - delta)  (negative because theta increases away from axis, Vtheta points towards axis usually?)
        # Standard convention: Vtheta is negative in compression region (flow turns towards body).
        
        V2 = np.sqrt(Vn2**2 + Vt2**2)
        Vr_shock = V2 * np.cos(beta - delta)
        Vtheta_shock = -V2 * np.sin(beta - delta)
        
        # Integrate inward (decreasing theta) from beta to 0 (or until Vtheta=0)
        # We stop when Vtheta = 0 (Cone surface condition)
        
        sol = solve_ivp(
            self.ode,
            [beta, 0.01], # Integrate from shock to near axis
            [Vr_shock, Vtheta_shock],
            events=lambda t, y: y[1], # Stop when Vtheta = 0
            dense_output=True,
            rtol=1e-6,
            atol=1e-8
        )
        
        if not sol.t_events[0].size:
            # Did not hit the cone surface (maybe detached shock or weak solution issues)
            return None, None
            
        theta_cone = sol.t_events[0][0]
        return np.degrees(theta_cone), sol

    def find_shock_angle(self, mach_inf, target_cone_deg):
        """
        Finds the shock angle required for a specific cone half-angle.
        """
        def residual(beta_deg):
            if beta_deg <= target_cone_deg: return -1.0
            # Mach wave angle is asin(1/M)
            mu = np.degrees(np.arcsin(1.0/mach_inf))
            if beta_deg <= mu: return -1.0 # Physically impossible shock
            
            try:
                theta_c, _ = self.solve_flow_field(mach_inf, beta_deg)
                if theta_c is None: return -100.0 # Failed
                return theta_c - target_cone_deg
            except ValueError:
                return -100.0

        # Search range: From cone angle to 90 deg (or detachment)
        # Mach wave is lower bound.
        lower_bound = np.degrees(np.arcsin(1.0/mach_inf)) * 1.01
        if lower_bound < target_cone_deg: lower_bound = target_cone_deg + 0.1
        
        try:
            res = root_scalar(residual, bracket=[lower_bound, 85.0], method='brentq')
            return res.root
        except:
            return None
            

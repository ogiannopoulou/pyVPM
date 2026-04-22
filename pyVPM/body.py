import numpy as np
import os

class Body:
    def __init__(self, filename=None):
        self.points = None
        self.normals = None
        self.tangents = None
        self.lengths = None
        self.midpoints = None
        
        if filename is not None:
            self.load_from_dat(filename)
            self.compute_geometry()
            
    def load_from_dat(self, filename):
        """
        Parses the specific Fortran .dat body file format (e.g. Dom1.dat).
        Assumes standard format where point count preceeds coordinates.
        """
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return
            
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        points = []
        in_points = False
        n_points = 0
        read_points = 0
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('!'):
                continue
                
            if 'nCpoints_of_Chunk_1' in line:
                n_points = int(line.split()[0])
                in_points = True
                continue
                
            # Stop condition could be reaching next boundary or end
            if in_points:
                parts = line.split()
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    points.append([x, y])
                    read_points += 1
                except ValueError:
                    # Not a coordinate line
                    pass
                    
                if read_points == n_points:
                    break
                    
        self.points = np.array(points).T # Shape (2, N)
        print(f"Loaded {self.points.shape[1]} points from {filename}")
        
    def compute_geometry(self):
        """
        Computes tangents, normals, panel lengths, and midpoints.
        Currently using centered finite differences for boundaries.
        Matches basic boundary element method (BEM) formulation.
        """
        if self.points is None or self.points.shape[1] < 2:
            return
            
        n = self.points.shape[1]
        x = self.points[0, :]
        y = self.points[1, :]
        
        self.midpoints = np.zeros((2, n))
        self.normals = np.zeros((2, n))
        self.tangents = np.zeros((2, n))
        self.lengths = np.zeros(n)
        
        # Computing closed loop properties
        for i in range(n):
            j = (i + 1) % n # next point wrapped
            
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            length = np.sqrt(dx**2 + dy**2)
            if length < 1e-14:
                continue
            if length < 1e-14:
                continue
            
            # Midpoint bounds the panel for BEM evaluation
            mx = x[i] + 0.5 * dx
            my = y[i] + 0.5 * dy
            
            # Tangent vector
            tx = dx / length
            ty = dy / length
            
            # Normal vector (pointing outward right-hand rule)
            nx = dy / length
            ny = -dx / length
            
            self.lengths[i] = length
            self.midpoints[0, i] = mx
            self.midpoints[1, i] = my
            self.tangents[0, i] = tx
            self.tangents[1, i] = ty
            self.normals[0, i] = nx
            self.normals[1, i] = ny
            
    def compute_self_interaction_matrix(self):
        """
        Matches body_self_interaction in potential calculus.
        Caches BEM influence coefficients for boundary condition resolution,
        using strictly real numbers (2D vectors) instead of complex math.
        """
        n = self.points.shape[1]
        
        # We need U and V influence coefficients, so the matrix is size (2*n, n)
        # or we return two strictly real matrices
        self.bem_matrix_u = np.zeros((n, n), dtype=np.float64)
        self.bem_matrix_v = np.zeros((n, n), dtype=np.float64)
        
        for i in range(n):
            xi = self.midpoints[0, i]
            yi = self.midpoints[1, i]
            for j in range(n):
                if i == j:
                    # Principal value for self-induction
                    self.bem_matrix_u[i, j] = 0.5 * self.tangents[0, i]
                    self.bem_matrix_v[i, j] = 0.5 * self.tangents[1, i]
                else:
                    xj = self.midpoints[0, j]
                    yj = self.midpoints[1, j]
                    
                    dx = xj - xi
                    dy = yj - yi
                    d2 = dx**2 + dy**2
                    
                    # Real component of - i / (2 * pi * (z_j - z_i))
                    factor = self.lengths[j] / (2.0 * np.pi * d2)
                    
                    # u = -y / r^2, v = x / r^2
                    self.bem_matrix_u[i, j] = -factor * dy
                    self.bem_matrix_v[i, j] =  factor * dx

    def generate_shed_particles(self, p_x, p_y, p_gamma, u_inf, v_inf, blob_radius):
        """
        Enforces the no-slip condition along the body panels by computing
        the local slip velocity and shedding compensating vortex particles.
        
        Returns arrays of (new_x, new_y, new_gamma) for the newly emitted particles.
        """
        if self.points is None:
            return np.array([]), np.array([]), np.array([])
            
        from .fmm import direct_velocity_kernel
        
        # 1. Compute induced velocity from all existing particles on the panel midpoints
        if len(p_x) > 0:
            u_ind, v_ind = direct_velocity_kernel(
                self.midpoints[0], self.midpoints[1], 
                p_x, p_y, p_gamma, blob_radius
            )
        else:
            u_ind = np.zeros_like(self.midpoints[0])
            v_ind = np.zeros_like(self.midpoints[1])
            
        # 2. Total velocity at the boundary before shedding (Freestream + Induced)
        u_total = u_inf + u_ind
        v_total = v_inf + v_ind
        
        # 3. Calculate slip velocity tangentially along the panels
        # Tangent vector is (tx, ty). Slip velocity = V_total dot Tangent
        slip_velocity = u_total * self.tangents[0] + v_total * self.tangents[1]
        self.slip_velocity = slip_velocity
        
        # 4. Circulation needed to cancel the slip: dGamma = V_slip * ds
        # To enforce no-slip, shed vorticity must equal the slip flow traversing the panel length
        gamma_shed = slip_velocity * self.lengths
        
        # 5. Emit new particles slightly off the boundary along the normal direction
        # Emit distance typically relates to the core radius or boundary layer thickness
        emit_dist = blob_radius * 1.1 
        
        new_x = self.midpoints[0] + self.normals[0] * emit_dist
        new_y = self.midpoints[1] + self.normals[1] * emit_dist
        
        return new_x, new_y, gamma_shed

    def compute_forces(self, dt, rho, nu, accel_x=0.0, accel_y=0.0):
        """
        Calculates aerodynamics forces (drag and lift) matching mod_Interactions.f90.
        Uses the shed vorticity (slip velocity * ds) and pressure components.
        Returns: Drag_vort, Lift_vort, Drag_pressure, Lift_pressure
        """
        if self.points is None:
            return 0.0, 0.0, 0.0, 0.0
            
        n = self.lengths.shape[0]
        drag_vort = 0.0
        lift_vort = 0.0
        drag_pressure = 0.0
        lift_pressure = 0.0
        
        # We need the previous timestep velocity on the boundary to calculate acceleration/pressure
        # For an accelerating freestream, t1 corresponds to acceleration dot tau !
        # The unsteady term t4 relates to the temporal derivative of velocity potential on boundary
        # For pure slip shedding matching mod_Interactions:
        # Drag_vort = Sum(tau_x * vort * ds) 
        # Lift_vort = Sum(tau_y * vort * ds)
        # vorticity at boundary is represented by the slip velocity
        
        vorticity = getattr(self, 'slip_velocity', np.zeros(n))
        
        # In Fortran: vis_cin_rho = nu * rho
        vis_cin_rho = nu * rho
        
        pressure = 0.0
        
        for i in range(1, n):
            tau_x = self.tangents[0, i]
            tau_y = self.tangents[1, i]
            norm_x = self.normals[0, i]
            norm_y = self.normals[1, i]
            ds = self.lengths[i]
            
            # Vorticity components
            drag_vort += tau_x * vorticity[i] * ds
            lift_vort += tau_y * vorticity[i] * ds
            
            # Pressure component (Unsteady Bernoulli integration)
            t1 = accel_x * tau_x + accel_y * tau_y
            # t4 represents the temporal derivative of velocity potential on boundary
            t4 = -0.5 * (vorticity[i] + vorticity[i-1]) / dt
            
            pressure = pressure + rho * ds * (t1 + t4)
            
            drag_pressure -= pressure * norm_x * ds
            lift_pressure -= pressure * norm_y * ds
            
        # Don't forget index 0 vorticity (Fortran likely loops over all for vort but 2..n for pressure)
        # Actually, let's just make sure we capture index 0 for the vorticity force:
        drag_vort += self.tangents[0, 0] * vorticity[0] * self.lengths[0]
        lift_vort += self.tangents[1, 0] * vorticity[0] * self.lengths[0]
        
        drag_vort *= vis_cin_rho
        lift_vort *= vis_cin_rho
        
        return drag_vort, lift_vort, drag_pressure, lift_pressure

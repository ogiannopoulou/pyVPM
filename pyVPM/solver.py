
import math
import numpy as np
import os
from .domain import Domain
from .diffusion import redistribute_and_remesh
from .velocity import compute_self_induced_velocity
from .fmm import fmm_velocity
from .body import Body

class VPMSolver:
    def __init__(self, dt=0.08, t_max=1.6, reynolds=200, dx=0.01, output_dir="results_py", resume=False):
        self.dt = dt
        self.t_max = t_max
        self.re = reynolds
        self.viscosity = 1.0 / reynolds
        self.rho = 1.0
        self.time = 0.0
        
        # Free-stream velocity
        self.u_inf = 1.0
        self.v_inf = 0.0
        
        # Domain setup
        self.domain = Domain(1, dx, self.viscosity, dt)
        self.domain.setup_mesh(-3.5, 12.0, -5.0, 5.0)
        
        self.step_counter = 0
        
        # Particles
        self.x = np.array([], dtype=np.float64)
        self.y = np.array([], dtype=np.float64)
        self.gamma = np.array([], dtype=np.float64)
        self.blob_radius = 1.5 * self.domain.dx 
        
        # Bodies in the domain (will shed vorticity)
        self.bodies = []
        
        # Output
        self.output_dir = output_dir
        if not resume:
            os.makedirs(self.output_dir, exist_ok=True)

        # Forces output
        self.forces_file = os.path.join(self.output_dir, "Forces.dat")
        if not resume:
            with open(self.forces_file, 'w') as f:
                f.write("# Time  Drag_pressure  Lift_pressure  Drag_vort  Lift_vort  Drag_total  Lift_total\n")

        
    def add_body(self, geometry_file):
        """Loads a body from .dat geometry limits and adds it to boundary logic"""
        b = Body(geometry_file)
        self.bodies.append(b)
        
    def add_particles(self, x, y, gamma):
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)
        self.gamma = np.append(self.gamma, gamma)
        
    def step(self):
        # Time ramp to match Fortran 'time_ramp = 1.0'
        time_ramp = 1.0
        if self.time <= time_ramp:
            import math
            arg = self.time * math.pi / time_ramp
            u_scale = max(0.5 * (-math.cos(arg) + 1.0), 1e-5)
            self.accel_x = self.u_inf * max((0.5 * math.pi * math.sin(arg)) / time_ramp, 1e-5)
            self.accel_y = self.v_inf * max((0.5 * math.pi * math.sin(arg)) / time_ramp, 1e-5)
        else:
            u_scale = 1.0
            self.accel_x = 0.0
            self.accel_y = 0.0
            
        current_u_inf = self.u_inf * u_scale
        current_v_inf = self.v_inf * u_scale

        # 0. Enforce Geometry Boundaries (Shedding new particles)
        for body in self.bodies:
            sx, sy, sgamma = body.generate_shed_particles(
                self.x, self.y, self.gamma, 
                current_u_inf, current_v_inf, 
                self.blob_radius
            )
            if len(sx) > 0:
                self.add_particles(sx, sy, sgamma)

        # 1. Convection (Velocity Evaluation)
        # We need to compute velocity at particle locations
        # Using the QuadTree FMM panel method for fast velocity summation
        if len(self.x) > 100:
            u, v = fmm_velocity(self.x, self.y, self.gamma, self.blob_radius, max_particles=50, max_level=5)
        else:
            u, v = compute_self_induced_velocity(self.x, self.y, self.gamma, self.blob_radius)
            
        # Add freestream convection
        u += current_u_inf
        v += current_v_inf
            
        # Euler integration for convection
        self.x += u * self.dt
        self.y += v * self.dt
        
        # Penalize particles that enter solid bodies (kill them inside)
        import matplotlib.path as mpath
        for body in self.bodies:
            if body.points is not None:
                outline = np.vstack([body.points[0, :], body.points[1, :]]).T
                path = mpath.Path(outline)
                pts = np.column_stack([self.x, self.y])
                if len(pts) > 0:
                    outside_mask = ~path.contains_points(pts)
                    self.x = self.x[outside_mask]
                    self.y = self.y[outside_mask]
                    self.gamma = self.gamma[outside_mask]
        
        self.step_counter += 1
        
        # 2. Diffusion (Redistribution on Mesh)
        # Execute only every cdt steps
        if self.step_counter % self.domain.cdt == 0:
            print(f"Diffusing at step {self.step_counter} (dt_diff={self.domain.dt_diff:.3f})")
            self.x, self.y, self.gamma = redistribute_and_remesh(
                self.domain, self.x, self.y, self.gamma
            )
        
        
        # 3. Calculate and save aerodynamic forces
        for body in self.bodies:
            drag_v, lift_v, drag_p, lift_p = body.compute_forces(
                dt=self.domain.dt_diff, 
                rho=self.rho, 
                nu=self.viscosity,
                accel_x=getattr(self, 'accel_x', 0.0),
                accel_y=getattr(self, 'accel_y', 0.0)
            )
            drag_tot = drag_p + drag_v
            lift_tot = lift_p + lift_v
            
            with open(self.forces_file, 'a') as f:
                f.write(f"{self.time:16.6e} {drag_p:16.6e} {lift_p:16.6e} {drag_v:16.6e} {lift_v:16.6e} {drag_tot:16.6e} {lift_tot:16.6e}\n")

        self.time += self.dt
        
    def run(self):
        n_steps = int(self.t_max / self.dt)
        print(f"Starting VPM simulation: Re={self.re}, Steps={n_steps}, Mesh dx={self.domain.dx}")
        print(f"Mesh size: {self.domain.mesh.nx}x{self.domain.mesh.ny}")
        
        self.save_vtk(0)
        
        for i in range(1, n_steps + 1):
            self.step()
            
            if i % 10 == 0:
                print(f"Step {i}/{n_steps}, Time={self.time:.3f}, Particles={len(self.x)}")
                self.save_vtk(i)
                
    def save_vtk(self, step):
        filename = os.path.join(self.output_dir, f"step_{step:04d}.vtk")
        n_points = len(self.x)
        if n_points == 0:
            return
            
        with open(filename, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("PyVPM Output\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")
            f.write(f"POINTS {n_points} float\n")
            for i in range(n_points):
                f.write(f"{self.x[i]} {self.y[i]} 0.0\n")
            
            f.write(f"CELLS {n_points} {n_points*2}\n")
            for i in range(n_points):
                f.write(f"1 {i}\n")
            
            f.write(f"CELL_TYPES {n_points}\n")
            # Vertex
            for i in range(n_points):
                f.write("1\n") 
                
            f.write(f"POINT_DATA {n_points}\n")
            f.write("SCALARS Circulation float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for val in self.gamma:
                f.write(f"{val}\n")

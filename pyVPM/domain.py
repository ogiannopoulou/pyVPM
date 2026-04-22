
import numpy as np

class Mesh:
    def __init__(self, dx, x_min, x_max, y_min, y_max):
        self.dx = dx
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        
        self.nx = int((x_max - x_min) / dx) + 1
        self.ny = int((y_max - y_min) / dx) + 1
        
        self.x_grid = np.linspace(x_min, x_min + (self.nx-1)*dx, self.nx)
        self.y_grid = np.linspace(y_min, y_min + (self.ny-1)*dx, self.ny)
        
        self.circulation = np.zeros((self.nx, self.ny))
        self.ready_to_diffuse = np.zeros((self.nx, self.ny), dtype=bool)

    def reset(self):
        self.circulation.fill(0.0)
        self.ready_to_diffuse.fill(False)

class Domain:
    def __init__(self, idx, dx, vis_cin, dt_conv, L_rif=1.0):
        self.index = idx
        self.dx = dx
        self.vis_cin = vis_cin
        self.L_rif = L_rif
        self.dt_conv = dt_conv
        
        self.dVol0 = dx * dx
        
        self.rdiff = 0.0
        self.rdiff_sq = 0.0
        self.diff_coeff = 0.0
        self.diff_cut = 0.0
        self.dt_diff = 0.0
        self.cdt = 1 # Steps per diffusion
        
        self.mesh = None
        self.calculate_parameters()

    def calculate_parameters(self):
        # Parameters matching Fortran
        Nnode_support = 51.0
        pi_inv = 1.0 / np.pi
        error_exp = 5.0
        
        # rdiff
        self.rdiff = np.sqrt(Nnode_support * pi_inv * self.dVol0)
        self.rdiff_sq = self.rdiff * self.rdiff
        
        # Target dt_diff
        # Formula: dt_diff = (Nnode * pi_inv * dVol0 * inv_vis * 0.25) / (error_exp * ln(10))
        inv_vis = 1.0 / self.vis_cin
        target_dt_diff = (Nnode_support * pi_inv * self.dVol0 * inv_vis * 0.25) / (error_exp * np.log(10.0))
        
        # Adjust to multiple of dt_conv
        self.cdt = max(1, int(round(target_dt_diff / self.dt_conv)))
        self.dt_diff = self.cdt * self.dt_conv
        
        print(f"Domain Setup: dx={self.dx}, nu={self.vis_cin}")
        print(f"  rdiff={self.rdiff:.4f}")
        print(f"  target_dt_diff={target_dt_diff:.4f}, actual_dt_diff={self.dt_diff:.4f} (cdt={self.cdt})")
        
        # Diffusion coefficients
        self.diff_coeff = 0.25 * inv_vis / self.dt_diff
        
        val_at_cutoff = np.exp(-self.rdiff_sq * self.diff_coeff)
        prefactor = 0.25 * pi_inv * inv_vis / self.dt_diff
        
        # Ensure L_rif is not zero
        lr = self.L_rif if self.L_rif > 1e-9 else 1.0
        self.diff_cut = prefactor * val_at_cutoff * (self.dVol0**2) / lr
        
        print(f"  diff_coeff={self.diff_coeff:.2f}, diff_cut={self.diff_cut:.2e}")

    def setup_mesh(self, x_min, x_max, y_min, y_max):
        # Extend mesh by rdiff to ensure coverage
        extension = self.rdiff * 1.5 
        self.mesh = Mesh(self.dx, x_min - extension, x_max + extension, y_min - extension, y_max + extension)

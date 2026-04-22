
import numpy as np
from numba import jit

@jit(nopython=True, fastmath=True, parallel=False)
def distribute_particles_to_mesh_kernel(
    p_x, p_y, p_circ, 
    mesh_circ, 
    x_min, y_min, dx, nx, ny, 
    rdiff, diff_coeff
):
    n_particles = len(p_x)
    rdiff_sq = rdiff * rdiff
    
    # Calculate search radius in index space
    nodes_r = int(np.ceil(rdiff / dx)) + 1
    
    for k in range(n_particles):
        px, py, gamma = p_x[k], p_y[k], p_circ[k]
        
        # Grid index of the particle
        i_center = int(round((px - x_min) / dx))
        j_center = int(round((py - y_min) / dx))
        
        i_start = max(0, i_center - nodes_r)
        i_end = min(nx, i_center + nodes_r + 1)
        j_start = max(0, j_center - nodes_r)
        j_end = min(ny, j_center + nodes_r + 1)
        
        # Pass 1: Total Weight
        total_weight = 0.0
        
        for i in range(i_start, i_end):
            node_x = x_min + i * dx
            dx_sq = (node_x - px)**2
            if dx_sq > rdiff_sq: continue
            
            for j in range(j_start, j_end):
                node_y = y_min + j * dx
                dy_sq = (node_y - py)**2
                dist2 = dx_sq + dy_sq
                
                if dist2 > rdiff_sq:
                    continue
                
                weight = np.exp(-dist2 * diff_coeff)
                total_weight += weight
                
        if total_weight < 1e-20:
            # Map strictly to nearest node if weights are too small
            # This handles the degenerate case, though strictly incorrect for diffusion
            if 0 <= i_center < nx and 0 <= j_center < ny:
                 mesh_circ[i_center, j_center] += gamma
            continue
            
        inv_total = 1.0 / total_weight
        
        # Pass 2: Distribute
        for i in range(i_start, i_end):
            node_x = x_min + i * dx
            dx_sq = (node_x - px)**2
            if dx_sq > rdiff_sq: continue

            for j in range(j_start, j_end):
                node_y = y_min + j * dx
                dy_sq = (node_y - py)**2
                dist2 = dx_sq + dy_sq
                
                if dist2 > rdiff_sq:
                    continue
                
                weight = np.exp(-dist2 * diff_coeff)
                mesh_circ[i, j] += weight * inv_total * gamma

def redistribute_and_remesh(domain, particles_x, particles_y, particles_gamma):
    if domain.vis_cin <= 1e-12:
        return particles_x, particles_y, particles_gamma

    # 1. Reset Mesh
    domain.mesh.reset()
    
    # 2. Redistribute
    distribute_particles_to_mesh_kernel(
        particles_x, particles_y, particles_gamma,
        domain.mesh.circulation,
        domain.mesh.x_min, domain.mesh.y_min, domain.mesh.dx,
        domain.mesh.nx, domain.mesh.ny,
        domain.rdiff, domain.diff_coeff
    )
    
    # 3. Generate New Particles
    active_mask = np.abs(domain.mesh.circulation) > domain.diff_cut
    indices_i, indices_j = np.where(active_mask)
    
    nx = len(indices_i)
    # Reallocate arrays
    new_x = np.empty(nx, dtype=np.float64)
    new_y = np.empty(nx, dtype=np.float64)
    new_gamma = np.empty(nx, dtype=np.float64)
    
    for k in range(nx):
        i, j = indices_i[k], indices_j[k]
        new_x[k] = domain.mesh.x_min + i * domain.mesh.dx
        new_y[k] = domain.mesh.y_min + j * domain.mesh.dx
        new_gamma[k] = domain.mesh.circulation[i, j]
        
    return new_x, new_y, new_gamma

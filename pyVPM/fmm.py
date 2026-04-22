import numpy as np
from numba import jit, set_num_threads
from .panels import QuadTree

# Using Numba for the computation heavy parts of the Fast Multipole evaluation
@jit(nopython=True, fastmath=True)
def direct_velocity_kernel(target_x, target_y, source_x, source_y, gamma, blob_radius):
    """ Direct Biot-Savart computation between target and source arrays """
    nt = len(target_x)
    ns = len(source_x)
    
    u = np.zeros(nt)
    v = np.zeros(nt)
    r_sq = blob_radius * blob_radius
    
    for i in range(nt):
        tx = target_x[i]
        ty = target_y[i]
        u_acc = 0.0
        v_acc = 0.0
        
        for j in range(ns):
            dx = tx - source_x[j]
            dy = ty - source_y[j]
            d2 = dx*dx + dy*dy + r_sq
            
            # Simple cutoff
            if d2 > 1e-14:
                val = gamma[j] / (2.0 * np.pi * d2)
                u_acc -= val * dy
                v_acc += val * dx
                
        u[i] = u_acc
        v[i] = v_acc
        
    return u, v

def fmm_velocity(x, y, gamma, blob_radius, max_particles=50, max_level=4, n_terms=10, theta=0.5):
    """
    Evaluates velocities using a Fast Multipole Method (Treecode approach).
    Uses purely real math (no complex numbers).
    """
    qt = QuadTree(max_level=max_level)
    pos = np.vstack((x, y))
    
    # In self-induced velocity, arrival == departure
    sorted_idx, _ = qt.build_tree(pos, pos, max_part_per_panel=max_particles)
    
    # Sort particles to match contiguous memory layouts for fast slice evaluation
    x_s = x[sorted_idx]
    y_s = y[sorted_idx]
    gamma_s = gamma[sorted_idx]
    
    # Output arrays
    u_s = np.zeros_like(x_s)
    v_s = np.zeros_like(y_s)
    
    # Pre-compute multipole moments for all panels
    for panel in qt.panels:
        if panel.first_arrival != -1:
            panel.ak, panel.bk = compute_panel_multipoles(
                panel.center[0], panel.center[1], 
                x_s, y_s, gamma_s, 
                panel.first_arrival, panel.last_arrival + 1, 
                n_terms
            )
            
    # Evaluate velocities combining Near-Field (Direct) and Far-Field (Multipole)
    # MAC (Multipole Acceptance Criterion) parameter theta
    for panel in qt.panels:
        # We only evaluate targets at the leaf nodes (nodes containing particles)
        if panel.first_arrival == -1 or len(panel.children) > 0:
            continue
            
        start = panel.first_arrival
        end = panel.last_arrival + 1
        t_x = x_s[start:end]
        t_y = y_s[start:end]
        
        # Traverse the tree for these target particles
        def _traverse(node):
            if node.first_arrival == -1:
                return
                
            # Distance from target panel center to source node center
            dx = panel.center[0] - node.center[0]
            dy = panel.center[1] - node.center[1]
            dist = np.sqrt(dx**2 + dy**2)
            
            # Multipole Acceptance Criterion (MAC)
            # If the node is far enough, use its multipole expansion
            if node.level > 0 and (node.side_length / dist < theta if dist > 0 else False):
                # Far-field: Evaluate multipole expansion
                for i in range(len(t_x)):
                    du, dv = evaluate_multipole_velocity(
                        t_x[i], t_y[i], node.center[0], node.center[1], 
                        node.ak, node.bk, n_terms
                    )
                    u_s[start + i] += du
                    v_s[start + i] += dv
            elif len(node.children) == 0:
                # Near-field / Leaf node: Direct summation
                s_start = node.first_arrival
                s_end = node.last_arrival + 1
                u_part, v_part = direct_velocity_kernel(
                    t_x, t_y, x_s[s_start:s_end], y_s[s_start:s_end], 
                    gamma_s[s_start:s_end], blob_radius
                )
                u_s[start:end] += u_part
                v_s[start:end] += v_part
            else:
                # Node is too close but not a leaf: open it and check children
                for child in node.children:
                    _traverse(child)

        # Start traversal from the root node (panel 0)
        _traverse(qt.panels[0])
            
    # Reverse sort to return to original particle order
    u = np.zeros_like(u_s)
    v = np.zeros_like(v_s)
    u[sorted_idx] = u_s
    v[sorted_idx] = v_s
    
    return u, v

@jit(nopython=True, fastmath=True)
def compute_panel_multipoles(panel_center_x, panel_center_y, p_x, p_y, p_gamma, p_start, p_end, n_terms=10):
    """
    Computes purely real multi-pole coefficients (ak, bk) for a panel.
    Equivalent to a complex expansion around panel_center, evaluated using ONLY real numbers.
    For z_c = panel_center, Z_j = p - z_c (offset of particle from center).
    ak + i*bk = sum( (-i * Gamma_j / 2*pi) * (Z_j)^k )
    Returns arrays ak, bk of length n_terms.
    """
    ak = np.zeros(n_terms)
    bk = np.zeros(n_terms)
    
    for j in range(p_start, p_end):
        gamma_val = p_gamma[j] / (2.0 * np.pi)
        dx = p_x[j] - panel_center_x
        dy = p_y[j] - panel_center_y
        
        # k = 0 term: (-i * gamma) * (dx + i*dy)^0 = -i * gamma
        # ak_0 = 0, bk_0 = -gamma
        ak[0] += 0.0
        bk[0] -= gamma_val
        
        # Keep track of (dx + i*dy)^k using real numbers
        z_real = 1.0
        z_imag = 0.0
        
        for k in range(1, n_terms):
            # multiply by (dx + i*dy)
            next_real = z_real * dx - z_imag * dy
            next_imag = z_real * dy + z_imag * dx
            z_real = next_real
            z_imag = next_imag
            
            # term is (-i * gamma) * (z_real + i * z_imag)
            # = gamma * z_imag - i * gamma * z_real
            ak[k] += gamma_val * z_imag
            bk[k] -= gamma_val * z_real
            
    return ak, bk

@jit(nopython=True, fastmath=True)
def evaluate_multipole_velocity(target_x, target_y, panel_center_x, panel_center_y, ak, bk, n_terms=10):
    """
    Evaluates velocity at target (x,y) from panel's multipole expansion.
    Calculates sum_{k=0}^{n_terms-1} (ak + i*bk) / (Z - z_c)^{k+1} using ONLY real numbers.
    Returns u, v components.
    """
    dx = target_x - panel_center_x
    dy = target_y - panel_center_y
    r2 = dx*dx + dy*dy
    
    if r2 < 1e-14:
        return 0.0, 0.0
        
    u_acc = 0.0
    v_acc = 0.0
    
    # We need 1 / (Z - z_c)^{k+1}. 
    # For k=0: 1 / (dx + i*dy) = (dx - i*dy) / r^2
    # Define inv_Z = (dx - i*dy) / r^2
    inv_z_real = dx / r2
    inv_z_imag = -dy / r2
    
    # Keep track of 1 / (Z - z_c)^{k+1}
    # Initial value for k=0 is inv_Z
    term_real = inv_z_real
    term_imag = inv_z_imag
    
    for k in range(n_terms):
        # We add (ak + i*bk) * (term_real + i*term_imag)
        vel_real = ak[k] * term_real - bk[k] * term_imag
        vel_imag = ak[k] * term_imag + bk[k] * term_real
        
        # velocity is u - i*v, so u = vel_real, v = -vel_imag
        u_acc += vel_real
        v_acc -= vel_imag
        
        # Update term for next iteration by multiplying by inv_Z
        next_real = term_real * inv_z_real - term_imag * inv_z_imag
        next_imag = term_real * inv_z_imag + term_imag * inv_z_real
        term_real = next_real
        term_imag = next_imag
        
    return u_acc, v_acc

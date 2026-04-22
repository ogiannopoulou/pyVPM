
import numpy as np
from numba import jit, prange

@jit(nopython=True, fastmath=True)
def biot_savart_kernel_single(target_x, target_y, source_x, source_y, gamma, blob_radius):
    dx = target_x - source_x
    dy = target_y - source_y
    r2 = dx*dx + dy*dy
    
    # Regularization
    denom = r2 + blob_radius*blob_radius
    if denom < 1e-20:
        return 0.0, 0.0
        
    val = gamma / (2.0 * np.pi * denom)
    
    u = -val * dy
    v =  val * dx
    
    return u, v

@jit(nopython=True, parallel=True, fastmath=True)
def compute_self_induced_velocity(x, y, gamma, blob_radius):
    n = len(x)
    u = np.zeros(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.float64)
    
    for i in prange(n):
        u_acc = 0.0
        v_acc = 0.0
        for j in range(n):
            if i == j:
                continue
            du, dv = biot_savart_kernel_single(x[i], y[i], x[j], y[j], gamma[j], blob_radius)
            u_acc += du
            v_acc += dv
        u[i] = u_acc
        v[i] = v_acc
            
    return u, v

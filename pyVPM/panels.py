import numpy as np

class Panel:
    """
    Representation of a single panel (node) in the quad-tree for Fast Multipole Method (FMM).
    Translates 'single_panel' from DVM_panels.f90
    """
    def __init__(self, label=0, level=0, center=None, side_length=0.0):
        self.label = label
        self.level = level
        
        # Pointers to particle data (indices in a sorted array)
        self.first_arrival = -1
        self.last_arrival = -1
        self.first_departure = -1
        self.last_departure = -1
        
        # Position in a binary/integer lattice for the given level
        self.binary_pos = np.zeros(2, dtype=int)
        
        # Geometric properties
        self.center = np.zeros(2, dtype=float) if center is None else np.array(center, dtype=float)
        self.side_length = float(side_length)
        
        # FMM multipole expansion coefficients
        self.ak = None
        self.bk = None
        
        # Children panels in the quad-tree
        self.children = []

class QuadTree:
    """
    Handles Panel initialization and domain decomposition, 
    mirroring the DVM_panels module routines.
    """
    def __init__(self, max_level=5, kemax=10, ext_dom=0.0):
        self.max_level = max_level
        self.kemax = kemax
        self.ext_dom = ext_dom
        self.size_panel = np.zeros(self.max_level + 1)
        self.panels = []
        
        # Constants for identifying sub-panels and neighbors
        self.binary_identification = np.array([
            [0, 0],  # Top-Left
            [0, 1],  # Bottom-Left
            [1, 0],  # Top-Right
            [1, 1]   # Bottom-Right
        ], dtype=int)
        
        self.panel_index_shift = np.array([
            [-1.0,  1.0],
            [-1.0, -1.0],
            [ 1.0,  1.0],
            [ 1.0, -1.0]
        ], dtype=float)

    def decomposition_init(self, arrival_pos, departure_pos):
        """
        Initializes domain decomposition given particle positions.
        Replaces 'decomposition_init' from DVM_panels.f90.
        """
        if departure_pos.shape[1] > 0:
            min_pos_dept = np.min(departure_pos, axis=1)
            max_pos_dept = np.max(departure_pos, axis=1)
        else:
            min_pos_dept = np.array([0.0, 0.0])
            max_pos_dept = np.array([0.0, 0.0])

        if arrival_pos.shape[1] > 0:
            min_pos_arr = np.min(arrival_pos, axis=1)
            max_pos_arr = np.max(arrival_pos, axis=1)
        else:
            min_pos_arr = min_pos_dept
            max_pos_arr = max_pos_dept

        min_pos = np.minimum(min_pos_dept, min_pos_arr)
        max_pos = np.maximum(max_pos_dept, max_pos_arr)

        length_dom_x = max_pos[0] - min_pos[0]
        length_dom_y = max_pos[1] - min_pos[1]

        # Make the domain square
        if length_dom_x > length_dom_y:
            min_pos[1] -= (length_dom_x - length_dom_y) * 0.5
        elif length_dom_x < length_dom_y:
            min_pos[0] -= (length_dom_y - length_dom_x) * 0.5

        length_dom = max(length_dom_x, length_dom_y) * (1.0 + self.ext_dom)
        min_pos -= self.ext_dom * 0.5 * length_dom

        # Compute panel sizes for each tree level
        for i in range(1, self.max_level + 1):
            self.size_panel[i] = length_dom / (2**i)

        # Initialize the global root panel (Panel 0)
        root_panel = Panel()
        root_panel.label = 0
        root_panel.level = 0
        root_panel.binary_pos = np.array([0, 0], dtype=int)
        root_panel.side_length = length_dom
        root_panel.center = min_pos + length_dom * 0.5
        
        self.panels = [root_panel]
        return root_panel

    def subdivide_panel(self, parent_panel, arrival_pos, departure_pos, 
                        arrival_indices, departure_indices):
        """
        Subdivides the domain of a panel into four equal quadrants and assigns
        particles into those quadrants.
        Translates 'four_parts' from DVM_velocity_evaluation.f90.
        
        Returns:
            children: List of the 4 new Panel objects.
            arr_indices_out: List of 4 numpy arrays containing arrival indices per child.
            dep_indices_out: List of 4 numpy arrays containing departure indices per child.
        """
        size_panel_div4 = parent_panel.side_length / 4.0
        
        # Child centers:
        # 0 (Top-Left):   (-size/4, +size/4)
        # 1 (Bottom-Left):(-size/4, -size/4)
        # 2 (Top-Right):  (+size/4, +size/4)
        # 3 (Bottom-Right):(+size/4, -size/4)
        child_centers = [
            np.array([parent_panel.center[0] - size_panel_div4, parent_panel.center[1] + size_panel_div4]),
            np.array([parent_panel.center[0] - size_panel_div4, parent_panel.center[1] - size_panel_div4]),
            np.array([parent_panel.center[0] + size_panel_div4, parent_panel.center[1] + size_panel_div4]),
            np.array([parent_panel.center[0] + size_panel_div4, parent_panel.center[1] - size_panel_div4])
        ]
        
        children = []
        for i in range(4):
            child = Panel()
            # In Fortran label counting usually follows creation order
            child.label = len(self.panels) + len(children)
            child.level = parent_panel.level + 1
            child.side_length = parent_panel.side_length / 2.0
            child.center = child_centers[i]
            child.binary_pos = parent_panel.binary_pos * 2 + self.binary_identification[i]
            children.append(child)
            
        parent_panel.children = children
        
        arr_child_indices = [np.array([], dtype=int) for _ in range(4)]
        dep_child_indices = [np.array([], dtype=int) for _ in range(4)]
        
        # Partition arrival particles
        if len(arrival_indices) > 0:
            arr_pts = arrival_pos[:, arrival_indices]
            # Replicate Fortran matching logic: Use sign() with y effectively flipped
            dx = arr_pts[0, :] - parent_panel.center[0]
            dy = -arr_pts[1, :] + parent_panel.center[1]
            
            # Use where condition avoiding perfectly 0 giving sign=0.
            # Fortran INT(1 + SIGN(1.0, d_vor))/2 means even exactly 0 becomes 1
            sign_dx = np.where(dx >= 0, 1.0, -1.0)
            sign_dy = np.where(dy >= 0, 1.0, -1.0)
            
            bin_x = ((1.0 + sign_dx) / 2).astype(int)
            bin_y = ((1.0 + sign_dy) / 2).astype(int)
            
            child_idx = bin_x * 2 + bin_y
            for i in range(4):
                arr_child_indices[i] = arrival_indices[child_idx == i]
                
        # Partition departure particles
        if len(departure_indices) > 0:
            dep_pts = departure_pos[:, departure_indices]
            dx = dep_pts[0, :] - parent_panel.center[0]
            dy = -dep_pts[1, :] + parent_panel.center[1]
            
            sign_dx = np.where(dx >= 0, 1.0, -1.0)
            sign_dy = np.where(dy >= 0, 1.0, -1.0)
            
            bin_x = ((1.0 + sign_dx) / 2).astype(int)
            bin_y = ((1.0 + sign_dy) / 2).astype(int)
            
            child_idx = bin_x * 2 + bin_y
            for i in range(4):
                dep_child_indices[i] = departure_indices[child_idx == i]
                
        return children, arr_child_indices, dep_child_indices

    def build_tree(self, arrival_pos, departure_pos, max_part_per_panel=100):
        """
        Recursively builds the FMM quad-tree decomposition down to max_level
        or until the number of departure particles in a panel is <= max_part_per_panel.
        Updates self.panels with the flattened tree.
        """
        root_panel = self.decomposition_init(arrival_pos, departure_pos)
        
        # Start indices
        n_arr = arrival_pos.shape[1] if arrival_pos.ndim == 2 else 0
        n_dep = departure_pos.shape[1] if departure_pos.ndim == 2 else 0
        arr_indices = np.arange(n_arr)
        dep_indices = np.arange(n_dep)
        
        root_panel.first_arrival = 0
        root_panel.last_arrival = n_arr - 1
        root_panel.first_departure = 0
        root_panel.last_departure = n_dep - 1
        
        # To maintain contiguous slices (like Fortran lists), we rebuild sorting arrays
        new_arrival_indices = []
        new_departure_indices = []
        
        def _recursive_build(panel, arr_inds, dep_inds):
            # Check stopping criteria
            if (panel.level >= self.max_level) or (len(dep_inds) <= max_part_per_panel):
                # Leaf node: store data indices for final contiguous arrays
                if len(arr_inds) > 0:
                    panel.first_arrival = len(new_arrival_indices)
                    new_arrival_indices.extend(arr_inds)
                    panel.last_arrival = len(new_arrival_indices) - 1
                else:
                    panel.first_arrival = -1
                    panel.last_arrival = -1
                    
                if len(dep_inds) > 0:
                    panel.first_departure = len(new_departure_indices)
                    new_departure_indices.extend(dep_inds)
                    panel.last_departure = len(new_departure_indices) - 1
                else:
                    panel.first_departure = -1
                    panel.last_departure = -1
                return
            
            # Subdivide
            children, arr_sub_inds, dep_sub_inds = self.subdivide_panel(
                panel, arrival_pos, departure_pos, arr_inds, dep_inds
            )
            
            self.panels.extend(children)
            
            # Calculate contiguous pointers recursively
            for i, child in enumerate(children):
                _recursive_build(child, arr_sub_inds[i], dep_sub_inds[i])
                
            # Aggregate parent pointers from children bounds
            valid_arr = [c for c in children if c.first_arrival != -1]
            if valid_arr:
                panel.first_arrival = valid_arr[0].first_arrival
                panel.last_arrival = valid_arr[-1].last_arrival
            else:
                panel.first_arrival = -1
                panel.last_arrival = -1
                
            valid_dep = [c for c in children if c.first_departure != -1]
            if valid_dep:
                panel.first_departure = valid_dep[0].first_departure
                panel.last_departure = valid_dep[-1].last_departure
            else:
                panel.first_departure = -1
                panel.last_departure = -1

        _recursive_build(root_panel, arr_indices, dep_indices)
        
        # Return the permutation required to sort the original particle arrays
        return np.array(new_arrival_indices, dtype=int), np.array(new_departure_indices, dtype=int)


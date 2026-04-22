
import os
import json
import re

def create_pvd_and_series(directory="results_py", output_name="simulation_results"):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    # List all VTK files
    files = [f for f in os.listdir(directory) if f.startswith("step_") and f.endswith(".vtk")]
    files.sort()  # crucial for correct time ordering

    if not files:
        print("No VTK files found.")
        return

    # Create PVD file
    pvd_filename = os.path.join(directory, f"{output_name}.pvd")
    
    # Create JSON series file
    json_series_filename = os.path.join(directory, f"{output_name}.vtk.series")
    json_data = {
        "file-series-version": "1.0",
        "files": []
    }

    with open(pvd_filename, 'w') as pvd:
        pvd.write('<?xml version="1.0"?>\n')
        pvd.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        pvd.write('<Collection>\n')
        
        for vtk_file in files:
            # Extract step number to estimate time or use index
            # Assuming step_0000.vtk, step_0100.vtk etc.
            # We don't have the exact time unless we parse the filename or assume dt
            # Let's try to parse index from filename
            match = re.search(r'step_(\d+)', vtk_file)
            if match:
                step_num = int(match.group(1))
                # Assuming dt=0.01 from the main script defaults
                time = step_num * 0.01 
            else:
                time = 0.0 # Fallback
            
            pvd.write(f'<DataSet timestep="{time}" file="{vtk_file}"/>\n')
            
            # Add to JSON series
            json_data["files"].append({"name": vtk_file, "time": time})
            
        pvd.write('</Collection>\n')
        pvd.write('</VTKFile>\n')

    # Write JSON series file
    with open(json_series_filename, 'w') as f:
        json.dump(json_data, f, indent=4)

    print(f"Created {pvd_filename} and {json_series_filename}")

if __name__ == "__main__":
    create_pvd_and_series()

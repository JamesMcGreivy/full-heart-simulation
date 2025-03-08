# Utility functions for openCARP meshes
# Author - James McGreivy

import os
import random
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from tqdm import tqdm
import pyvista as pv
from pyvista import themes
import vtk
from vtk.util import numpy_support

# ~~ For Loading the openCARP meshes into python ~~ #
class OpenCARPMeshReader:
    """
    Class to read openCARP mesh files (.pts, .elem, .surf) and UVC data (.csv)
    and convert them to NumPy arrays and pandas DataFrames.
    """
    
    def __init__(self, base_dir, prefix):
        """
        Initialize the reader with the directory and file prefix.
        """
        self.base_dir = base_dir
        self.prefix = prefix
        
        # Set file paths
        self.pts_file = os.path.join(base_dir, f"{prefix}.pts")
        self.elem_file = os.path.join(base_dir, f"{prefix}.elem")
        self.surf_file = os.path.join(base_dir, f"{prefix}.surf")
        self.uvc_file = os.path.join(base_dir, f"{prefix}_UVC.csv")
        
        # Storage for mesh components
        self.points = None
        self.tetrahedra = None
        self.triangles = None
        self.triangle_regions = None
        self.tetrahedra_regions = None
        self.uvc_data = None
    
    def read_points(self):
        """
        Read the .pts file and return an array of points with dimension (n_points x 3).
        """
        print(f"Reading points from {self.pts_file}")
        
        with open(self.pts_file, 'r') as f:
            # Read the number of points from the first line
            n_points = int(f.readline().strip())
            
            # Initialize the points array
            points = np.zeros((n_points, 3))
            
            # Read each point
            for i in range(n_points):
                line = f.readline().strip()
                # Parse x, y, z coordinates
                x, y, z = map(float, line.split())
                points[i] = [x, y, z]
        
        self.points = points
        print(f"Loaded {n_points} points with shape {points.shape}")
        return points
    
    def read_tetrahedra(self):
        """
        Read the .elem file and return an array of tetrahedra with dimension (n_tetrahedra x 4).
        Each tetrahedron contains indices to the points array.
        """
        print(f"Reading tetrahedra from {self.elem_file}")
        
        tetrahedra = []
        tetrahedra_regions = []
        
        with open(self.elem_file, 'r') as f:
            # Read the number of elements from the first line
            n_elements = int(f.readline().strip())
            
            # Read each tetrahedron
            for i in range(n_elements):
                line = f.readline().strip()
                
                # Parse the line for a tetrahedron (Tt format)
                if line.startswith('Tt'):
                    parts = line.split()
                    
                    # Extract the 4 vertex indices
                    indices = [int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])]
                    
                    # Extract region ID (if available)
                    region_id = int(parts[5]) if len(parts) > 5 else 0
                    
                    tetrahedra.append(indices)
                    tetrahedra_regions.append(region_id)
        
        # Convert to NumPy arrays
        tetrahedra = np.array(tetrahedra)
        tetrahedra_regions = np.array(tetrahedra_regions)
        
        self.tetrahedra = tetrahedra
        self.tetrahedra_regions = tetrahedra_regions
        
        print(f"Loaded {len(tetrahedra)} tetrahedra with shape {tetrahedra.shape}")
        return tetrahedra, tetrahedra_regions
    
    def read_triangles(self):
        """
        Read the .surf file and return an array of triangles with dimension (n_triangles x 3).
        Each triangle contains indices to the points array.
        """
        print(f"Reading triangles from {self.surf_file}")
        
        triangles = []
        triangle_regions = []
        
        with open(self.surf_file, 'r') as f:
            lines = f.readlines()
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                i += 1
                
                # Check if this line defines a region
                region_match = re.match(r"(\d+)(?:\s+Reg\s+(\d+))?", line)
                if region_match:
                    n_triangles_in_region = int(region_match.group(1))
                    
                    if region_match.group(2):
                        region_id = int(region_match.group(2))
                    else:
                        region_id = -1
                    
                    # Read triangles for this region
                    for j in range(n_triangles_in_region):
                        if i < len(lines):
                            tri_line = lines[i].strip()
                            i += 1
                            
                            # Parse triangle (Tr format)
                            if tri_line.startswith('Tr'):
                                parts = tri_line.split()
                                # Extract the 3 vertex indices
                                indices = [int(parts[1]), int(parts[2]), int(parts[3])]
                                
                                triangles.append(indices)
                                triangle_regions.append(region_id)
        
        # Convert to NumPy arrays
        triangles = np.array(triangles)
        triangle_regions = np.array(triangle_regions)
        
        self.triangles = triangles
        self.triangle_regions = triangle_regions
        
        print(f"Loaded {len(triangles)} triangles with shape {triangles.shape}")
        return triangles, triangle_regions
    
    def read_uvc_data(self):
        """
        Read the UVC data from the CSV file into a pandas DataFrame.
        """
        if os.path.exists(self.uvc_file):
            print(f"Reading UVC data from {self.uvc_file}")
            
            # Read the CSV file into a pandas DataFrame
            uvc_data = pd.read_csv(self.uvc_file)
            
            self.uvc_data = uvc_data
            
            print(f"Loaded UVC data with shape {uvc_data.shape}")
            print(f"UVC columns: {uvc_data.columns.tolist()}")
            
            return uvc_data
        else:
            print(f"UVC file not found: {self.uvc_file}")
            return None
    
    def read_all(self):
        """
        Read all mesh files and UVC data.
        """
        points = self.read_points()
        tetrahedra, tetrahedra_regions = self.read_tetrahedra()
        triangles, triangle_regions = self.read_triangles()
        uvc_data = self.read_uvc_data()
        
        return (points, tetrahedra, tetrahedra_regions, triangles, triangle_regions, uvc_data)
    



# ~~ For using the UVCs to compute fiber and sheet directions ~~ #



def compute_normed_gradients(points, phi_transmural, phi_longitudinal, phi_circumferential):
    """Compute normalized gradient of phi."""
    from scipy.spatial import KDTree
    
    tree = KDTree(points)
    grad_phi_transmural = np.zeros_like(points)
    grad_phi_longitudinal = np.zeros_like(points)
    grad_phi_circumferential = np.zeros_like(points)
    
    for i, p in enumerate(points):
        _, neighbors = tree.query(p, k=20)  # Find nearby points
        
        neighbor_points = points[neighbors]
        
        neighbor_phi_transmural = phi_transmural[neighbors]
        grad = np.linalg.lstsq(neighbor_points - p, neighbor_phi_transmural - phi_transmural[i], rcond=None)[0]
        grad_phi_transmural[i] = grad / (1e-9 + np.linalg.norm(grad))  # Normalize

        neighbor_phi_longitudinal = phi_longitudinal[neighbors]
        grad = np.linalg.lstsq(neighbor_points - p, neighbor_phi_longitudinal - phi_longitudinal[i], rcond=None)[0]
        grad_phi_longitudinal[i] = grad / (1e-9 + np.linalg.norm(grad))  # Normalize

        neighbor_phi_circumferential = phi_circumferential[neighbors]
        grad = np.linalg.lstsq(neighbor_points - p, neighbor_phi_circumferential - phi_circumferential[i], rcond=None)[0]
        grad_phi_circumferential[i] = grad / (1e-9 + np.linalg.norm(grad))  # Normalize
    
    return grad_phi_transmural, grad_phi_longitudinal, grad_phi_circumferential

def compute_fiber_sheet_directions(TransmuralField, LongitudinalField, CircumferentialField, phi_transmural, endo_fiber_angle=60, epi_fiber_angle=-60, endo_sheet_angle=-65, epi_sheet_angle=25):
    """
    Compute rule-based cardiac fiber and sheet directions.
    """
    # Number of points
    n_points = len(TransmuralField)
    
    # Initialize output arrays
    fiber_directions = np.zeros_like(TransmuralField)
    sheet_directions = np.zeros_like(TransmuralField)
    sheet_normal_directions = np.zeros_like(TransmuralField)
    
    # Fiber angle parameters (in radians)
    endo_fiber_angle = np.radians(endo_fiber_angle)
    epi_fiber_angle = np.radians(epi_fiber_angle)
    
    # Sheet angle parameters (in radians)
    endo_sheet_angle = np.radians(endo_sheet_angle)
    epi_sheet_angle = np.radians(epi_sheet_angle)
    
    for i in range(n_points):
        # Normalize local coordinate system vectors
        t_vec = TransmuralField[i] / np.linalg.norm(TransmuralField[i])
        l_vec = LongitudinalField[i] / np.linalg.norm(LongitudinalField[i])
        c_vec = CircumferentialField[i] / np.linalg.norm(CircumferentialField[i])
        
        # Interpolate fiber angle based on transmural position (linear interpolation)
        phi = phi_transmural[i]
        fiber_angle = endo_fiber_angle * (1 - phi) + epi_fiber_angle * phi
        
        # Interpolate sheet angle based on transmural position (linear interpolation)
        sheet_angle = endo_sheet_angle * (1 - phi) + epi_sheet_angle * phi
        
        # Compute fiber direction by rotating the circumferential vector toward the longitudinal direction
        # Rotation around transmural axis
        fiber_dir = c_vec * np.cos(fiber_angle) + l_vec * np.sin(fiber_angle)
        fiber_dir = fiber_dir / np.linalg.norm(fiber_dir)
        
        # IMPROVED SHEET CALCULATION:
        # First, define sheet direction as a rotation of the transmural vector in the transmural-longitudinal plane
        # This rotation is around an axis perpendicular to both transmural and longitudinal vectors (i.e., circumferential)
        sheet_dir = t_vec * np.cos(sheet_angle) + l_vec * np.sin(sheet_angle)
        
        # Make sheet_dir orthogonal to fiber_dir using Gram-Schmidt process
        # Project out any component of sheet_dir that's parallel to fiber_dir
        fiber_component = np.dot(sheet_dir, fiber_dir) * fiber_dir
        sheet_dir = sheet_dir - fiber_component
        sheet_dir = sheet_dir / np.linalg.norm(sheet_dir)
        
        # Compute sheet normal as cross product to ensure perfect orthogonality
        sheet_normal = np.cross(fiber_dir, sheet_dir)
        sheet_normal = sheet_normal / np.linalg.norm(sheet_normal)
        
        # Store the results
        fiber_directions[i] = fiber_dir
        sheet_directions[i] = sheet_dir
        sheet_normal_directions[i] = sheet_normal
    
    return fiber_directions, sheet_directions, sheet_normal_directions

def average_unit_vector(vectors):
    """
    Calculate the average unit vector from a set of vectors.
    """
    average_vector = np.mean(vectors, axis=0)
    norm = np.linalg.norm(average_vector)
    
    # Avoid division by zero
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0])  # Default to x-axis if vectors cancel out
    
    return average_vector / norm

def precompute_element_directions(fiber_dirs, sheet_dirs, sheet_normal_dirs, tetrahedra):
    """
    Precompute the averaged fiber, sheet and sheet normal directions for all elements
    """
    n_elements = len(tetrahedra)
    element_fiber_dirs = np.zeros((n_elements, 3))
    element_sheet_dirs = np.zeros((n_elements, 3))
    element_normal_dirs = np.zeros((n_elements, 3))
    
    # Use vectorized operations where possible
    for i in tqdm(range(n_elements), desc="Precomputing element directions"):
        # Get the nodes for this element
        tet_nodes = tetrahedra[i]
        
        # Average the directions for this element
        element_fiber_dirs[i] = average_unit_vector(fiber_dirs[tet_nodes])
        element_sheet_dirs[i] = average_unit_vector(sheet_dirs[tet_nodes])
        element_normal_dirs[i] = average_unit_vector(sheet_normal_dirs[tet_nodes])
    
    return element_fiber_dirs, element_sheet_dirs, element_normal_dirs

def write_lon_file(filename, fiber_directions, sheet_directions, sheet_normal_directions, tetrahedra, batch_size=10000):
    """
    Write fiber, sheet, and sheet normal directions to a .lon file for openCARP.
    """
    n_elements = len(tetrahedra)
    
    print(f"Processing {n_elements} elements for .lon file...")
    
    # Precompute all element directions at once
    element_fiber_dirs, element_sheet_dirs, element_normal_dirs = precompute_element_directions(
        fiber_directions, sheet_directions, sheet_normal_directions, tetrahedra
    )
    
    # Verify orthogonality of precomputed directions
    print("Verifying orthogonality of element directions...")
    max_dot_f_s = np.max(np.abs(np.sum(element_fiber_dirs * element_sheet_dirs, axis=1)))
    max_dot_f_n = np.max(np.abs(np.sum(element_fiber_dirs * element_normal_dirs, axis=1)))
    max_dot_s_n = np.max(np.abs(np.sum(element_sheet_dirs * element_normal_dirs, axis=1)))
    
    print(f"Maximum dot products: fiber·sheet={max_dot_f_s:.6f}, fiber·normal={max_dot_f_n:.6f}, sheet·normal={max_dot_s_n:.6f}")
    
    print(f"Writing to {filename}...")
    
    # Write to file in batches
    with open(filename, 'w') as f:
        # Header: 3 for fiber, sheet, and sheet-normal directions
        f.write('2\n')
        
        # Write in batches to avoid memory issues
        for i in tqdm(range(0, n_elements, batch_size), desc="Writing .lon file"):
            batch_end = min(i + batch_size, n_elements)
            batch_elements = range(i, batch_end)
            
            # Create a single string for the batch
            lines = []
            for j in batch_elements:
                f_vec = element_fiber_dirs[j]
                s_vec = element_sheet_dirs[j]
                line = f"{f_vec[0]:.8f} {f_vec[1]:.8f} {f_vec[2]:.8f} " + \
                       f"{s_vec[0]:.8f} {s_vec[1]:.8f} {s_vec[2]:.8f} "
                lines.append(line)
            
            # Write the batch
            f.write('\n'.join(lines) + '\n')
    
    print(f"Successfully wrote fiber orientations to {filename}")

def read_fiber_directions(lon_file_path):
    """
    Read fiber directions from a .lon file
    """
    # Open the file and read all lines
    with open(lon_file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip the first line (header)
    data_lines = lines[1:]
    
    # Initialize an array to store fiber directions
    n_elements = len(data_lines)
    fiber_directions = np.zeros((n_elements, 3))
    
    # Parse each line to extract the fiber directions
    for i, line in enumerate(data_lines):
        # Split the line into values
        values = line.strip().split()
        
        # First 3 values are the fiber direction
        fiber_directions[i, 0] = float(values[0])
        fiber_directions[i, 1] = float(values[1])
        fiber_directions[i, 2] = float(values[2])
    
    return fiber_directions

def map_element_fibers_to_points(element_fiber_dirs, tetrahedra, n_points):
    """
    Map element fiber directions to mesh points using a vectorized approach
    """
    # Initialize arrays to store sum of fiber directions and count of elements per point
    fiber_sum = np.zeros((n_points, 3))
    count = np.zeros(n_points)
    
    # Create a flattened index of point-element pairs
    elem_indices = np.repeat(np.arange(len(tetrahedra)), 4)
    point_indices = tetrahedra.flatten()
    
    # Add the fiber direction of each element to its associated points
    for i, p_idx in enumerate(point_indices):
        elem_idx = elem_indices[i]
        fiber_sum[p_idx] += element_fiber_dirs[elem_idx]
        count[p_idx] += 1
    
    # Avoid division by zero
    count[count == 0] = 1
    
    # Compute average fiber direction for each point
    fiber_dirs = fiber_sum / count[:, np.newaxis]
    
    # Normalize the fiber directions
    norms = np.linalg.norm(fiber_dirs, axis=1)
    norms[norms == 0] = 1  # Avoid division by zero
    fiber_dirs = fiber_dirs / norms[:, np.newaxis]
    
    return fiber_dirs



# ~~ For tagging the fast conducting endocardium ~~ #



def tag_fast_conducting_endocardium(
    points, 
    tetrahedra, 
    triangles, 
    triangle_regions, 
    phi_longitudinal, 
    output_file,
    long_min=0.1,
    long_max=0.9,
):
    """
    Tag tetrahedra for fast conducting endocardium based on:
    1. Points with longitudinal coordinates between long_min and long_max
    2. Points that are part of triangles with triangle_regions == 3 or 4 (Endocardium)
    """
    # Step 1: Identify points that satisfy phi_longitudinal criterion
    print(f"Identifying points with phi_longitudinal between {long_min} and {long_max}...")
    longitudinal_mask = (phi_longitudinal >= long_min) & (phi_longitudinal <= long_max)
    valid_longitudinal_points = set(np.where(longitudinal_mask)[0])
    print(f"Found {len(valid_longitudinal_points)} points with valid longitudinal coordinate")
    
    # Step 2: Identify points that are part of endocardial triangles (regions 3 or 4)
    print("Identifying endocardial points...")
    endocardial_triangles = np.where((triangle_regions == 3) | (triangle_regions == 4))[0]
    endocardial_points = set()
    for tri_idx in endocardial_triangles:
        for point_idx in triangles[tri_idx]:
            endocardial_points.add(point_idx)
    print(f"Found {len(endocardial_points)} points on the endocardium")
    
    # Step 3: Find intersection of both sets
    valid_points = valid_longitudinal_points.intersection(endocardial_points)
    print(f"Found {len(valid_points)} points that satisfy both criteria")
    
    # Step 4: Mark tetrahedra containing valid points
    print("Tagging tetrahedra...")
    n_tetrahedra = tetrahedra.shape[0]
    tetrahedra_tags = np.zeros(n_tetrahedra, dtype=int)
    
    # Convert to set for faster lookups
    valid_points_set = set(valid_points)
    
    # Process in batches to avoid memory issues
    batch_size = 10000
    for batch_start in tqdm(range(0, n_tetrahedra, batch_size)):
        batch_end = min(batch_start + batch_size, n_tetrahedra)
        batch_tetrahedra = tetrahedra[batch_start:batch_end]
        
        # Check each tetrahedron in the batch
        for i, tet in enumerate(batch_tetrahedra):
            # If any point in the tetrahedron is valid, tag it as 1
            if any(point_idx in valid_points_set for point_idx in tet):
                tetrahedra_tags[batch_start + i] = 1
    
    num_tagged = np.sum(tetrahedra_tags == 1)
    print(f"Tagged {num_tagged} tetrahedra ({(num_tagged/n_tetrahedra)*100:.2f}%) as fast conducting endocardium")
    
    # Step 5: Write the modified .elem file
    print(f"Writing modified element file to {output_file}...")
    with open(output_file, 'w') as fout:
        # Write header (number of tetrahedra)
        fout.write(f"{n_tetrahedra}\n")
        
        # Process each tetrahedron and write to file
        for i in tqdm(range(n_tetrahedra)):
            tet = tetrahedra[i]
            # Format: Tt node1 node2 node3 node4 tag
            line = f"Tt {tet[0]} {tet[1]} {tet[2]} {tet[3]} {tetrahedra_tags[i]}"
            fout.write(f"{line}\n")
    
    print(f"Successfully wrote modified element file to {output_file}")

def save_fascicular_sites_to_vtx(is_fascicular_site, output_filename):
    """
    Save the indices of fascicular sites to a .vtx file.
    """
    # Get indices of fascicular sites
    site_indices = np.where(is_fascicular_site)[0]
    num_sites = len(site_indices)
    
    print(f"Saving {num_sites} fascicular site indices to {output_filename}")
    
    # Write to file
    with open(output_filename, 'w') as f:
        # Write header (number of sites)
        f.write(f"{num_sites}\n")
        f.write("extra\n")
        
        # Write each index on a new line
        for idx in site_indices:
            f.write(f"{idx}\n")
    
    print(f"Successfully saved fascicular sites to {os.path.abspath(output_filename)}")
    return os.path.abspath(output_filename)



# ~~ Visualization Functions ~~ #



def visualize_phi(points, phi, subsample_factor=10, point_size=5, cmap="viridis"):
    """
    Visualize a scalar field using just points (no mesh connectivity required).
    """
    # Set up theme
    pv.set_plot_theme("document")
    
    print(f"Original points: {len(points)}")
    
    # Subsample points to make visualization manageable
    n_points = len(points)
    n_sample = n_points // subsample_factor
    
    # Make sure we don't have too few or too many points
    n_sample = max(1000, min(n_sample, 50000))
    
    print(f"Subsampling points from {n_points} to {n_sample}")
    
    # Random sampling
    sample_indices = sorted(random.sample(range(n_points), n_sample))
    
    # Create subsampled arrays
    sampled_points = points[sample_indices]
    sampled_phi = phi[sample_indices]
    
    # Create point cloud
    cloud = pv.PolyData(sampled_points)
    cloud.point_data["transmural"] = sampled_phi
    
    # Create plotter
    plotter = pv.Plotter()
    plotter.add_text("Phi Point Cloud", font_size=14)
    
    # Add point cloud with scalar values
    plotter.add_mesh(cloud, render_points_as_spheres=True, point_size=point_size,
                    scalars="transmural", cmap=cmap, opacity=1.0,
                    show_scalar_bar=True, scalar_bar_args={"title": "Phi"})
    
    # Set view and show
    plotter.view_isometric()
    plotter.show()


def visualize_vector_fields(points, triangles, triangle_regions, 
                           transmural_field, longitudinal_field, circumferential_field, 
                           subsample_factor=20, glyph_scale=0.005):
    """
    Create side-by-side visualizations of the vector fields on epicardium and endocardium.
    Left plot: Epicardium (triangle_regions == 2)
    Right plot: Endocardium (triangle_regions == 3 or 4)
    """
    # Set up a nice theme for visualization
    my_theme = themes.DarkTheme()
    my_theme.lighting = True
    my_theme.show_edges = False
    my_theme.background = 'white'
    my_theme.window_size = [1600, 800]
    pv.set_plot_theme(my_theme)
    
    print(f"Original points: {len(points)}")
    print(f"Original triangles: {len(triangles)}")
    
    # Prepare triangle cells for PyVista format
    triangle_cells = np.column_stack(
        (np.full(len(triangles), 3), triangles)
    ).flatten()
    
    # Create full mesh
    full_mesh = pv.PolyData(points, triangle_cells)
    
    # Create a plotter with two side-by-side viewports
    plotter = pv.Plotter(shape=(1, 2))
    
    # Filter triangles by region
    epicardium_triangles = triangles[triangle_regions == 2]
    endocardium_triangles = triangles[np.logical_or(triangle_regions == 3, triangle_regions == 4)]
    
    print(f"Epicardium triangles: {len(epicardium_triangles)}")
    print(f"Endocardium triangles: {len(endocardium_triangles)}")
    
    # Create separate meshes for each surface
    epicardium_cells = np.column_stack(
        (np.full(len(epicardium_triangles), 3), epicardium_triangles)
    ).flatten()
    
    endocardium_cells = np.column_stack(
        (np.full(len(endocardium_triangles), 3), endocardium_triangles)
    ).flatten()
    
    epicardium_mesh = pv.PolyData(points, epicardium_cells)
    endocardium_mesh = pv.PolyData(points, endocardium_cells)
    
    # Function to sample points from a mesh
    def create_sampled_vectors(mesh, field_names=["transmural", "longitudinal", "circumferential"]):
        # Extract unique points from this mesh
        unique_point_ids = np.unique(mesh.faces.reshape(-1, 4)[:, 1:].flatten())
        
        # Further sample to reduce number of arrows
        n_sample = min(len(unique_point_ids), len(unique_point_ids) // subsample_factor)
        sample_indices = sorted(random.sample(list(unique_point_ids), n_sample))
        
        # Create a point cloud with the sampled points
        sampled_points = points[sample_indices]
        point_cloud = pv.PolyData(sampled_points)
        
        # Add vector data
        point_cloud["transmural"] = transmural_field[sample_indices]
        point_cloud["longitudinal"] = longitudinal_field[sample_indices]
        point_cloud["circumferential"] = circumferential_field[sample_indices]
        
        return point_cloud, sample_indices
    
    # Create sampled vector fields for each surface
    epicardium_vectors, epi_indices = create_sampled_vectors(epicardium_mesh)
    endocardium_vectors, endo_indices = create_sampled_vectors(endocardium_mesh)
    
    print(f"Sampled epicardium points: {epicardium_vectors.n_points}")
    print(f"Sampled endocardium points: {endocardium_vectors.n_points}")
    
    # Left viewport: Epicardium
    plotter.subplot(0, 0)
    plotter.add_text("Epicardium Vector Fields", font_size=16, color="black")
    
    # Add epicardium surface
    plotter.add_mesh(epicardium_mesh, color='lightgray', opacity=0.8)
    
    # Add vector fields for epicardium
    transmural_arrows = epicardium_vectors.glyph(
        orient="transmural",
        scale=False,
        factor=glyph_scale
    )
    plotter.add_mesh(transmural_arrows, color="red", label="Transmural")
    
    longitudinal_arrows = epicardium_vectors.glyph(
        orient="longitudinal",
        scale=False,
        factor=glyph_scale
    )
    plotter.add_mesh(longitudinal_arrows, color="green", label="Longitudinal")
    
    circumferential_arrows = epicardium_vectors.glyph(
        orient="circumferential",
        scale=False,
        factor=glyph_scale
    )
    plotter.add_mesh(circumferential_arrows, color="blue", label="Circumferential")
    
    # Add legend
    plotter.add_legend(size=(0.2, 0.2), loc='upper right')
    
    # Right viewport: Endocardium
    plotter.subplot(0, 1)
    plotter.add_text("Endocardium Vector Fields", font_size=16, color="black")
    
    # Add endocardium surface
    plotter.add_mesh(endocardium_mesh, color='lightgray', opacity=0.8)
    
    # Add vector fields for endocardium
    transmural_arrows = endocardium_vectors.glyph(
        orient="transmural",
        scale=False,
        factor=glyph_scale
    )
    plotter.add_mesh(transmural_arrows, color="red", label="Transmural")
    
    longitudinal_arrows = endocardium_vectors.glyph(
        orient="longitudinal",
        scale=False,
        factor=glyph_scale
    )
    plotter.add_mesh(longitudinal_arrows, color="green", label="Longitudinal")
    
    circumferential_arrows = endocardium_vectors.glyph(
        orient="circumferential",
        scale=False,
        factor=glyph_scale
    )
    plotter.add_mesh(circumferential_arrows, color="blue", label="Circumferential")
    
    # Add legend
    plotter.add_legend(size=(0.2, 0.2), loc='upper right')
    
    # Link the cameras between the two viewports
    plotter.link_views()
    
    # Set camera position
    plotter.camera_position = 'xz'
    plotter.camera.zoom(1.2)
    
    # Show the plot
    plotter.show()

    plotter.screenshot('../figures/cardial_coordinates.png')

def visualize_fibers(points, triangles, triangle_regions, fiber_dirs, 
                    subsample_factor=20, glyph_scale=0.005):
    """
    Create side-by-side visualizations of the fiber directions on epicardium and endocardium.
    Left plot: Endocardium (triangle_regions == 3 or 4) in red
    Right plot: Epicardium (triangle_regions == 2) in blue
    """
    # Set up a nice theme for visualization
    my_theme = themes.DarkTheme()
    my_theme.lighting = True
    my_theme.show_edges = False
    my_theme.background = 'white'
    my_theme.window_size = [1600, 800]
    pv.set_plot_theme(my_theme)
    
    print(f"Original points: {len(points)}")
    print(f"Original triangles: {len(triangles)}")
    
    # Prepare triangle cells for PyVista format
    triangle_cells = np.column_stack(
        (np.full(len(triangles), 3), triangles)
    ).flatten()
    
    # Create full mesh
    full_mesh = pv.PolyData(points, triangle_cells)
    
    # Create a plotter with two side-by-side viewports
    plotter = pv.Plotter(shape=(1, 2))
    
    # Filter triangles by region
    epicardium_triangles = triangles[triangle_regions == 2]
    endocardium_triangles = triangles[np.logical_or(triangle_regions == 3, triangle_regions == 4)]
    
    print(f"Epicardium triangles: {len(epicardium_triangles)}")
    print(f"Endocardium triangles: {len(endocardium_triangles)}")
    
    # Create separate meshes for each surface
    epicardium_cells = np.column_stack(
        (np.full(len(epicardium_triangles), 3), epicardium_triangles)
    ).flatten()
    
    endocardium_cells = np.column_stack(
        (np.full(len(endocardium_triangles), 3), endocardium_triangles)
    ).flatten()
    
    epicardium_mesh = pv.PolyData(points, epicardium_cells)
    endocardium_mesh = pv.PolyData(points, endocardium_cells)
    
    # Function to sample points from a mesh
    def create_sampled_vectors(mesh):
        # Extract unique points from this mesh
        unique_point_ids = np.unique(mesh.faces.reshape(-1, 4)[:, 1:].flatten())
        
        # Further sample to reduce number of arrows
        n_sample = min(len(unique_point_ids), len(unique_point_ids) // subsample_factor)
        sample_indices = sorted(random.sample(list(unique_point_ids), n_sample))
        
        # Create a point cloud with the sampled points
        sampled_points = points[sample_indices]
        point_cloud = pv.PolyData(sampled_points)
        
        # Add vector data
        point_cloud["fibers"] = fiber_dirs[sample_indices]
        
        return point_cloud, sample_indices
    
    # Create sampled vector fields for each surface
    epicardium_vectors, epi_indices = create_sampled_vectors(epicardium_mesh)
    endocardium_vectors, endo_indices = create_sampled_vectors(endocardium_mesh)
    
    print(f"Sampled epicardium points: {epicardium_vectors.n_points}")
    print(f"Sampled endocardium points: {endocardium_vectors.n_points}")
    
    # Left viewport: Endocardium (in red)
    plotter.subplot(0, 0)
    plotter.add_text("Endocardium Fiber Directions", font_size=16, color="black")
    
    # Add endocardium surface
    plotter.add_mesh(endocardium_mesh, color='mistyrose', opacity=0.8)
    
    # Add fiber directions for endocardium
    fiber_arrows = endocardium_vectors.glyph(
        orient="fibers",
        scale=False,
        factor=glyph_scale
    )
    plotter.add_mesh(fiber_arrows, color="red", label="Fiber Direction")
    
    # Add legend
    plotter.add_legend(size=(0.2, 0.2), loc='upper right')
    
    # Right viewport: Epicardium (in blue)
    plotter.subplot(0, 1)
    plotter.add_text("Epicardium Fiber Directions", font_size=16, color="black")
    
    # Add epicardium surface
    plotter.add_mesh(epicardium_mesh, color='lightblue', opacity=0.8)
    
    # Add fiber directions for epicardium
    fiber_arrows = epicardium_vectors.glyph(
        orient="fibers",
        scale=False,
        factor=glyph_scale
    )
    plotter.add_mesh(fiber_arrows, color="blue", label="Fiber Direction")
    
    # Add legend
    plotter.add_legend(size=(0.2, 0.2), loc='upper right')
    
    # Link the cameras between the two viewports
    plotter.link_views()
    
    # Set camera position
    plotter.camera_position = 'xz'
    plotter.camera.zoom(1.2)
    
    # Show the plot
    plotter.show()

    plotter.screenshot('../figures/fiber_directions.png')


def visualize_fascicular_sites(points, triangles, triangle_regions, is_fascicular_site, 
                               fascicular_site_tag, sphere_scale=0.005):
    """
    Visualize fascicular sites on the endocardial surface with color coding by site type.
    
    Parameters:
    -----------
    points : ndarray
        Coordinates of mesh vertices
    triangles : ndarray
        Triangle vertex indices
    triangle_regions : ndarray
        Region labels for each triangle
    is_fascicular_site : ndarray
        Boolean array indicating fascicular sites
    fascicular_site_tag : ndarray
        Integer array with site type labels (1-5)
    sphere_scale : float
        Scale factor for visualization spheres
    """
    # Set up a nice theme for visualization
    my_theme = themes.DarkTheme()
    my_theme.lighting = True
    my_theme.show_edges = False
    my_theme.background = 'white'
    my_theme.window_size = [1000, 800]
    pv.set_plot_theme(my_theme)
    
    print(f"Original points: {len(points)}")
    print(f"Original triangles: {len(triangles)}")
    print(f"Number of fascicular sites: {np.sum(is_fascicular_site)}")
    
    # Filter triangles to get only endocardium
    endocardium_triangles = triangles[np.logical_or(triangle_regions == 3, triangle_regions == 4)]
    print(f"Endocardium triangles: {len(endocardium_triangles)}")
    
    # Create endocardium mesh
    endocardium_cells = np.column_stack(
        (np.full(len(endocardium_triangles), 3), endocardium_triangles)
    ).flatten()
    
    endocardium_mesh = pv.PolyData(points, endocardium_cells)
    
    # Get indices of fascicular sites - use ALL of them, no subsampling
    fascicular_indices = np.where(is_fascicular_site)[0]
    print(f"Displaying all {len(fascicular_indices)} fascicular sites")
    
    # Create a new point cloud for fascicular sites
    fascicular_points = points[fascicular_indices]
    fascicular_cloud = pv.PolyData(fascicular_points)
    
    # Add site tags as a scalar field for coloring
    site_tags = fascicular_site_tag[fascicular_indices]
    fascicular_cloud['site_tag'] = site_tags
    
    # Define colors for each site type
    site_colors = {
        1: 'red',         # LV anterior
        2: 'blue',        # LV posterior
        3: 'green',       # Septal
        4: 'purple',      # RV moderator band
    }
    
    # Create a plotter
    plotter = pv.Plotter()
    plotter.add_text("Endocardium with Fascicular Sites", font_size=16, color="black")
    
    # Add endocardium surface
    plotter.add_mesh(endocardium_mesh, color='mistyrose', opacity=0.7, 
                     label="Endocardial Surface")
    
    # Add spheres for each fascicular site type separately
    for tag, color in site_colors.items():
        # Filter to this site type
        site_indices = np.where(site_tags == tag)[0]
        if len(site_indices) == 0:
            continue
            
        site_points = fascicular_points[site_indices]
        site_cloud = pv.PolyData(site_points)
        
        # Create spheres for this site type
        spheres = site_cloud.glyph(
            geom=pv.Sphere(radius=1),
            scale=False,
            factor=sphere_scale
        )
        
        # Get the site name for the legend
        site_names = {
            1: "LV Anterior",
            2: "LV Posterior",
            3: "Septal",
            4: "RV Moderator Band",
        }
        
        plotter.add_mesh(spheres, color=color, label=site_names[tag])
    
    # Add legend
    plotter.add_legend(size=(0.3, 0.3), loc='upper right')
    
    # Set camera position for a good view
    plotter.camera_position = 'xz'
    plotter.camera.zoom(1.2)

    # Show the plot
    plotter.show()

    plotter.screenshot('../figures/fascicular_sites.png') 
    
    return plotter


# ~~ For converting .vtp and .vtu mesh formats into openCARP formats ~~



# Function to read VTP files
def read_vtp(filename):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    polydata = reader.GetOutput()
    return polydata

# Function to read VTU files
def read_vtu(filename):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    grid = reader.GetOutput()
    return grid

# Function to extract and save Universal Ventricular Coordinates (UVC)
def extract_save_uvc(volume_mesh, output_prefix, output_path):
    # UVC coordinate names to extract
    uvc_names = ["tv", "tm", "rtSin", "rtCos", "rt", "ab"]
    
    # Dictionary to store the coordinates
    coordinates = {}
    
    # Extract each coordinate if available
    for coord_name in uvc_names:
        if volume_mesh.GetPointData().HasArray(coord_name):
            data_array = volume_mesh.GetPointData().GetArray(coord_name)
            coordinates[coord_name] = numpy_support.vtk_to_numpy(data_array)
            print(f"Extracted {coord_name} shape: {coordinates[coord_name].shape}")
            print(f"{coord_name} range: {coordinates[coord_name].min()} to {coordinates[coord_name].max()}")
    
    # Create a pandas DataFrame from the coordinates
    if coordinates:
        df = pd.DataFrame(coordinates)
        # Create full path for outputs
        csv_path = os.path.join(output_path, f"{output_prefix}_UVC.csv")
        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"Saved UVC data to {csv_path}")
    else:
        print("No UVC data found in the volume mesh")

# Function to convert VTK data to openCARP format
def convert_to_opencarp(volume_mesh, surface_mesh, output_prefix, output_path):
    # Extract and save UVC data
    extract_save_uvc(volume_mesh, output_prefix, output_path)
    
    # Generate full paths for output files
    pts_path = os.path.join(output_path, f"{output_prefix}.pts")
    elem_path = os.path.join(output_path, f"{output_prefix}.elem")
    surf_path = os.path.join(output_path, f"{output_prefix}.surf")
    
    # Generate .pts file from volume mesh points
    points = volume_mesh.GetPoints()
    num_points = points.GetNumberOfPoints()
    
    with open(pts_path, "w") as pts_file:
        pts_file.write(f"{num_points}\n")
        for i in range(num_points):
            point = points.GetPoint(i)
            pts_file.write(f"{point[0]} {point[1]} {point[2]}\n")
    
    print(f"Created {pts_path} with {num_points} points")
    
    # Generate .elem file from volume mesh cells (tetrahedra)
    num_cells = volume_mesh.GetNumberOfCells()
    
    with open(elem_path, "w") as elem_file:
        elem_file.write(f"{num_cells}\n")
        
        # Determine if we have cell data for regions
        cell_data = None
        region_id_array_name = None
        
        for i in range(volume_mesh.GetCellData().GetNumberOfArrays()):
            array_name = volume_mesh.GetCellData().GetArrayName(i)
            # Possible names for region IDs in the cell data
            if array_name in ["region", "material", "celldata"]:
                region_id_array_name = array_name
                cell_data = numpy_support.vtk_to_numpy(volume_mesh.GetCellData().GetArray(array_name))
                break
        
        # Default region ID if no cell data is available
        default_region_id = 0
        
        for i in range(num_cells):
            cell = volume_mesh.GetCell(i)
            
            # Check if the cell is a tetrahedron (VTK_TETRA = 10)
            if cell.GetCellType() == 10:  # VTK_TETRA
                # Get the 4 point IDs of the tetrahedron
                point_ids = [cell.GetPointId(j) for j in range(4)]
                
                # Get region ID for this cell if available
                region_id = default_region_id
                if cell_data is not None:
                    region_id = int(cell_data[i])
                
                # Write in Tt format for tetrahedron with region ID
                # Note: openCARP is 0-indexed, same as VTK
                elem_file.write(f"Tt {point_ids[0]} {point_ids[1]} {point_ids[2]} {point_ids[3]} {region_id}\n")
            else:
                print(f"Warning: Ignoring non-tetrahedral cell (type {cell.GetCellType()}) at index {i}")
    
    print(f"Created {output_prefix}.elem with tetrahedral elements")
    
    # Generate .surf file from surface mesh triangles
    # Extract class/region information if available
    region_labels = None
    if surface_mesh.GetPointData().HasArray("class"):
        class_array = surface_mesh.GetPointData().GetArray("class")
        region_labels = numpy_support.vtk_to_numpy(class_array)
    
    # Create a dictionary to organize triangles by region
    triangles_by_region = {}
    
    num_surface_cells = surface_mesh.GetNumberOfCells()
    for i in range(num_surface_cells):
        cell = surface_mesh.GetCell(i)
        
        # Check if the cell is a triangle (VTK_TRIANGLE = 5)
        if cell.GetCellType() == 5:  # VTK_TRIANGLE
            # Get the 3 point IDs of the triangle
            point_ids = [cell.GetPointId(j) for j in range(3)]
            
            # Determine region for this triangle
            # We need to map the surface point IDs to volume point IDs
            # For simplicity, we'll use the most common region among the triangle's points
            region = 0  # Default region
            
            if region_labels is not None:
                # Get the region labels for the points in this triangle
                triangle_regions = [region_labels[point_id] for point_id in point_ids]
                
                # Use the most common region label for this triangle
                from collections import Counter
                region = Counter(triangle_regions).most_common(1)[0][0]
            
            # Add this triangle to the appropriate region
            if region not in triangles_by_region:
                triangles_by_region[region] = []
            
            triangles_by_region[region].append(point_ids)
    
    # Write the .surf file with triangles organized by region
    with open(surf_path, "w") as surf_file:
        # For each region, write the number of triangles in that region followed by Reg ID
        for region, triangles in sorted(triangles_by_region.items()):
            num_triangles_in_region = len(triangles)
            surf_file.write(f"{num_triangles_in_region} Reg {region}\n")
            
            # Write all triangles for this region
            for point_ids in triangles:
                surf_file.write(f"Tr {point_ids[0]} {point_ids[1]} {point_ids[2]}\n")
    
    print(f"Created {output_prefix}.surf with surface triangles")

def scale_pts_file(file_path, scale_factor=10000):
    """
    Read a .pts file, scale the points by the given factor, and save back to the same file.
    
    Args:
        file_path (str): Path to the .pts file
        scale_factor (float): Scaling factor to apply to coordinates (default: 10000 for cm to micrometers)
    """
    print(f"Reading points from {file_path}")
    
    # Read the points file
    with open(file_path, 'r') as f:
        # Read the number of points from the first line
        n_points = int(f.readline().strip())
        
        # Initialize the points array
        points = np.zeros((n_points, 3))
        
        # Read each point
        for i in range(n_points):
            line = f.readline().strip()
            # Parse x, y, z coordinates
            x, y, z = map(float, line.split())
            points[i] = [x, y, z]
    
    print(f"Loaded {n_points} points with shape {points.shape}")
    
    # Scale the points
    scaled_points = points * scale_factor
    print(f"Scaled points by factor of {scale_factor}")
    
    # Write the scaled points back to the file
    with open(file_path, 'w') as f:
        # Write the number of points
        f.write(f"{n_points}\n")
        
        # Write each scaled point
        for i in range(n_points):
            f.write(f"{scaled_points[i, 0]} {scaled_points[i, 1]} {scaled_points[i, 2]}\n")
    
    print(f"Saved scaled points back to {file_path}") 

# Main function to read VTK files and convert to openCARP format
def vtk_to_opencarp(vtp_filename, vtu_filename, output_prefix, output_path):
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Reading surface mesh: {vtp_filename}")
    surface_mesh = read_vtp(vtp_filename)
    
    print(f"Reading volume mesh: {vtu_filename}")
    volume_mesh = read_vtu(vtu_filename)
    
    print(f"Converting to openCARP format. Output directory: {output_path}")
    convert_to_opencarp(volume_mesh, surface_mesh, output_prefix, output_path)

    scale_pts_file(os.path.join(output_path, output_prefix+".pts"), scale_factor=1000) # Convert the output mesh from mm to micrometers for openCARP
    
    print(f"Conversion complete! Files saved to: {output_path}")
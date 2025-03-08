import numpy as np
import re
from collections import defaultdict
from scipy.spatial import KDTree, cKDTree
import argparse
import pandas as pd

class MeshHandler:
    def __init__(self, pts_file, surf_file):
        """
        Initialize the MeshHandler with paths to the pts and surf files.
        
        Args:
            pts_file (str): Path to the .pts file
            surf_file (str): Path to the .surf file
        """
        self.pts_file = pts_file
        self.surf_file = surf_file
        self.points = None
        self.triangles = None
        self.triangle_regions = None
        
    def read_points(self):
        """
        Read the .pts file and return an array of points with dimension (n_points x 3).
        
        Returns:
            numpy.ndarray: Array of 3D points
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
    
    def read_triangles(self):
        """
        Read the .surf file and return an array of triangles with dimension (n_triangles x 3).
        Each triangle contains indices to the points array.
        
        Returns:
            numpy.ndarray: Array of triangle indices
            numpy.ndarray: Array of region IDs for each triangle
        """
        print(f"Reading triangles from {self.surf_file}")
        
        triangles = []
        triangle_regions = []
        
        with open(self.surf_file, 'r') as f:
            lines = f.readlines()
            
            i = 0
            current_region = -1  # Default region if not specified
            
            while i < len(lines):
                line = lines[i].strip()
                i += 1
                
                # Check if this line defines a region
                region_match = re.match(r"(\d+)(?:\s+Reg\s+(\d+))?", line)
                if region_match:
                    n_triangles_in_region = int(region_match.group(1))
                    
                    if region_match.group(2):
                        current_region = int(region_match.group(2))
                    
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
                                triangle_regions.append(current_region)
                else:
                    # Handle the case where there is no region header (just the number of triangles)
                    try:
                        n_triangles = int(line)
                        for j in range(n_triangles):
                            if i < len(lines):
                                tri_line = lines[i].strip()
                                i += 1
                                
                                # Parse triangle (Tr format)
                                if tri_line.startswith('Tr'):
                                    parts = tri_line.split()
                                    # Extract the 3 vertex indices
                                    indices = [int(parts[1]), int(parts[2]), int(parts[3])]
                                    
                                    triangles.append(indices)
                                    triangle_regions.append(current_region)
                    except ValueError:
                        # This line is not a region header or a number of triangles
                        # It might be a triangle directly
                        if line.startswith('Tr'):
                            parts = line.split()
                            # Extract the 3 vertex indices
                            indices = [int(parts[1]), int(parts[2]), int(parts[3])]
                            
                            triangles.append(indices)
                            triangle_regions.append(current_region)
        
        # Convert to NumPy arrays
        triangles = np.array(triangles)
        triangle_regions = np.array(triangle_regions)
        
        self.triangles = triangles
        self.triangle_regions = triangle_regions
        
        print(f"Loaded {len(triangles)} triangles with shape {triangles.shape}")
        return triangles, triangle_regions
    
    def compute_triangle_centroids(self):
        """
        Compute the centroid of each triangle.
        
        Returns:
            numpy.ndarray: Array of centroids with shape (n_triangles, 3)
        """
        if self.triangles is None or self.points is None:
            raise ValueError("Points and triangles must be loaded first")
        
        # Get the coordinates of each vertex in each triangle
        p1 = self.points[self.triangles[:, 0]]
        p2 = self.points[self.triangles[:, 1]]
        p3 = self.points[self.triangles[:, 2]]
        
        # Compute the centroid as the average of the three vertices
        centroids = (p1 + p2 + p3) / 3.0
        
        return centroids
    
    def write_surf_file(self, output_path, triangles=None, triangle_regions=None):
        """
        Write triangles to a .surf file, organized by regions.
        
        Args:
            output_path (str): Path to write the .surf file
            triangles (numpy.ndarray, optional): Triangle indices. If None, use self.triangles
            triangle_regions (numpy.ndarray, optional): Region IDs. If None, use self.triangle_regions
        """
        if triangles is None:
            triangles = self.triangles
        
        if triangle_regions is None:
            triangle_regions = self.triangle_regions
        
        print(f"Writing .surf file to {output_path}")
        
        # Group triangles by region
        triangles_by_region = defaultdict(list)
        for tri_idx, region_id in enumerate(triangle_regions):
            triangles_by_region[region_id].append(triangles[tri_idx])
        
        # Write the .surf file with triangles organized by region
        with open(output_path, "w") as surf_file:
            # For each region, write the number of triangles in that region followed by Reg ID
            for region, region_triangles in sorted(triangles_by_region.items()):
                if region == -1:
                    # Skip the default region if it exists
                    continue
                
                num_triangles_in_region = len(region_triangles)
                surf_file.write(f"{num_triangles_in_region} Reg {region}\n")
                
                # Write all triangles for this region
                for point_ids in region_triangles:
                    surf_file.write(f"Tr {point_ids[0]} {point_ids[1]} {point_ids[2]}\n")


def transfer_regions(original_pts_file, original_surf_file, 
                     downsampled_pts_file, downsampled_surf_file, 
                     output_surf_file):
    """
    Transfer region information from an original mesh to a downsampled mesh.
    
    Args:
        original_pts_file (str): Path to the original .pts file
        original_surf_file (str): Path to the original .surf file with regions
        downsampled_pts_file (str): Path to the downsampled .pts file
        downsampled_surf_file (str): Path to the downsampled .surf file without regions
        output_surf_file (str): Path to write the output .surf file with restored regions
    """
    print("Starting region transfer process...")
    
    # Load the original mesh
    original_mesh = MeshHandler(original_pts_file, original_surf_file)
    original_mesh.read_points()
    original_mesh.read_triangles()
    original_centroids = original_mesh.compute_triangle_centroids()
    
    # Load the downsampled mesh
    downsampled_mesh = MeshHandler(downsampled_pts_file, downsampled_surf_file)
    downsampled_mesh.read_points()
    downsampled_mesh.read_triangles()
    downsampled_centroids = downsampled_mesh.compute_triangle_centroids()
    
    print("Computing nearest triangles for region classification...")
    
    # Build a KD-tree for efficient nearest neighbor search
    kdtree = KDTree(original_centroids)
    
    # For each downsampled triangle, find the nearest original triangle
    distances, indices = kdtree.query(downsampled_centroids)
    
    # Assign regions based on the nearest original triangles
    downsampled_regions = original_mesh.triangle_regions[indices]
    
    print("Region classification complete")
    
    # Count triangles per region for verification
    unique_regions, region_counts = np.unique(downsampled_regions, return_counts=True)
    for region, count in zip(unique_regions, region_counts):
        print(f"Region {region}: {count} triangles")
    
    # Write the downsampled mesh with restored regions
    downsampled_mesh.triangle_regions = downsampled_regions
    downsampled_mesh.write_surf_file(output_surf_file)
    
    print(f"Successfully wrote output mesh with regions to {output_surf_file}")

def read_points(pts_file):
    """
    Read the .pts file and return an array of points with dimension (n_points x 3).
    
    Args:
        pts_file (str): Path to the .pts file
    
    Returns:
        numpy.ndarray: Array of 3D points
    """
    print(f"Reading points from {pts_file}")
    
    with open(pts_file, 'r') as f:
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
    return points

def interpolate_uvcs(input_pts_path, input_UVC_path, output_pts_path, output_UVC_path):
    """
    Interpolate UVC coordinates from input mesh to output mesh.
    
    Args:
        input_pts_path (str): Path to input points file
        input_UVC_path (str): Path to input UVC file
        output_pts_path (str): Path to output points file
        output_UVC_path (str): Path to output UVC file
    """
    print("Starting UVC interpolation...")
    
    # Read input points and UVCs
    input_points = read_points(input_pts_path)
    input_uvcs = pd.read_csv(input_UVC_path)
    
    # Read output points
    output_points = read_points(output_pts_path)
    
    print(f"Input UVCs shape: {input_uvcs.shape}")
    print(f"Input points shape: {input_points.shape}")
    print(f"Output points shape: {output_points.shape}")
    
    # Verify that the number of points matches
    if input_points.shape[0] != input_uvcs.shape[0]:
        raise ValueError(f"Number of points in input_pts ({input_points.shape[0]}) does not match number of UVCs ({input_uvcs.shape[0]})")
    
    # Get UVC column names (headers)
    uvc_columns = input_uvcs.columns.tolist()
    
    # Convert input UVCs to numpy array for faster processing
    input_uvc_values = input_uvcs.values
    
    # Build KD-tree for efficient nearest neighbor search
    print("Building KD-tree for spatial interpolation...")
    kdtree = cKDTree(input_points)
    
    # Find K nearest neighbors for each output point
    k = 8  # Number of neighbors to consider
    print(f"Finding {k} nearest neighbors for each output point...")
    distances, indices = kdtree.query(output_points, k=k)
    
    # Inverse distance weighting interpolation
    print("Performing inverse distance weighting interpolation...")
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-10
    weights = 1.0 / (distances + epsilon)
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    normalized_weights = weights / weights_sum
    
    # Initialize output UVCs array
    output_uvc_values = np.zeros((output_points.shape[0], input_uvc_values.shape[1]))
    
    # Interpolate UVC values
    for i in range(output_points.shape[0]):
        for j in range(input_uvc_values.shape[1]):
            output_uvc_values[i, j] = np.sum(input_uvc_values[indices[i], j] * normalized_weights[i])
    
    # Convert back to pandas DataFrame
    output_uvcs = pd.DataFrame(output_uvc_values, columns=uvc_columns)
    
    # Write the interpolated UVCs to file
    print(f"Writing interpolated UVCs to {output_UVC_path}")
    output_uvcs.to_csv(output_UVC_path, index=False)
    
    print("UVC interpolation completed successfully!")
    return output_uvcs

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description='This script is used to transfer region labels between an input (original) and output (downsampled) mesh in openCARP.')
    # Add arguments
    parser.add_argument('--input_mesh_path', type=str, required=True, help='Path + header for input (original) openCARP mesh. Ex: path/to/mesh, where path/to/mesh.pts and path/to/mesh.surf exists')
    parser.add_argument('--output_mesh_path', type=str, default='output.txt', help='Path + header for output (downsampled) openCARP mesh. Ex: path/to/mesh, where path/to/mesh.pts and path/to/mesh.surf exists')
    
    # Parse the arguments
    args = parser.parse_args()

    input_pts = args.input_mesh_path + ".pts"
    input_surf = args.input_mesh_path + ".surf"
    input_UVCs = args.input_mesh_path + "_UVC.csv"

    output_pts = args.output_mesh_path + ".pts"
    output_surf = args.output_mesh_path + ".surf"
    output_UVCs = args.output_mesh_path + "_UVC.csv"

    transfer_regions(input_pts, input_surf, output_pts, output_surf, output_surf) # Transfer the region labels from the input mesh to the output mesh
    interpolate_uvcs(input_pts, input_UVCs, output_pts, output_UVCs) # Interpolate the UVCs from the input mesh points to the output mesh points
    
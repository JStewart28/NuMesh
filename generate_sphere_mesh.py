import sys
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans

def fibonacci_sphere(radius, num_points):
    """ Generate `num_points` nearly uniform points on a sphere of given `radius` using the Fibonacci lattice. """
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.pi * (1 + 5**0.5)  # Golden angle
    theta = np.arccos(1 - 2 * indices / num_points)
    lon = phi * indices

    x = radius * np.sin(theta) * np.cos(lon)
    y = radius * np.sin(theta) * np.sin(lon)
    z = radius * np.cos(theta)
    
    return np.column_stack((x, y, z))

def spherical_triangulation(points):
    """ Generate a valid 3D triangular connectivity using the convex hull of the points. """
    hull = ConvexHull(points)
    return hull.simplices  # Triangular connectivity

def distribute_mesh(points, connectivity, num_procs):
    """ Partition the mesh into `num_procs` contiguous sections while keeping boundary triangles. """
    kmeans = KMeans(n_clusters=num_procs, n_init=10, random_state=42)
    labels = kmeans.fit_predict(points)  # Cluster points based on spatial proximity

    partitions = []
    ghost_owners = {}  # Map ghost point indices to the rank that owns them
    for i in range(num_procs):
        part_indices = np.where(labels == i)[0]  # Get indices of points in this partition
        part_points = points[part_indices]  # Extract corresponding points

        # Create a global-to-local mapping for owned points
        global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(part_indices)}

        # Track ghost points
        ghost_points = {}
        local_cells = []
        
        for cell in connectivity:
            cell_partitions = {labels[v] for v in cell}  # Set of partitions owning this triangle
            
            if i in cell_partitions:  # Keep the triangle if it has any local points
                local_cell = []
                for v in cell:
                    if v in global_to_local:
                        local_cell.append(global_to_local[v])  # Owned point
                    else:
                        if v not in ghost_points:
                            ghost_points[v] = len(part_points) + len(ghost_points)  # Assign new index
                            ghost_owners[v] = i  # Mark the owner rank for the ghost point
                        local_cell.append(ghost_points[v])  # Ghost point index

                local_cells.append(local_cell)

        # Merge owned and ghost points
        ghost_indices = list(ghost_points.keys())
        full_points = np.vstack([part_points, points[ghost_indices]])

        # Store ghost point indices for writing to file
        ghost_flags = np.zeros(len(full_points), dtype=np.uint8)
        ghost_flags[len(part_points):] = 1  # Mark ghost points

        partitions.append((full_points, np.array(local_cells), len(part_points), ghost_flags))

    return partitions, ghost_owners

def write_vtu(filename, points, connectivity, num_owned, ghost_flags, ghost_owners):
    """ Write a partitioned mesh to a VTU file, marking ghost points. """
    points_vtk = vtk.vtkPoints()
    for p in points:
        points_vtk.InsertNextPoint(p)

    cells_vtk = vtk.vtkCellArray()
    for cell in connectivity:
        triangle = vtk.vtkTriangle()
        for i in range(3):
            triangle.GetPointIds().SetId(i, int(cell[i]))
        cells_vtk.InsertNextCell(triangle)

    # Write the ghost point flag array
    vtk_ghost_array = numpy_to_vtk(ghost_flags)
    vtk_ghost_array.SetName("ghost_points")

    # Write the ghost owner array
    vtk_ghost_array = numpy_to_vtk(ghost_owners)
    vtk_ghost_array.SetName("ghost_owners")

    polydata = vtk.vtkUnstructuredGrid()
    polydata.SetPoints(points_vtk)
    polydata.SetCells(vtk.VTK_TRIANGLE, cells_vtk)
    polydata.GetPointData().AddArray(vtk_ghost_array)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()

def main():
    if len(sys.argv) != 4:
        print("Usage: python generate_vtu_mesh.py <radius> <num_points> <num_procs>")
        sys.exit(1)

    radius = float(sys.argv[1])
    num_points = int(sys.argv[2])
    num_procs = int(sys.argv[3])

    if num_points % num_procs != 0:
        print("Error: num_points must be evenly divisible by num_procs.")
        sys.exit(1)

    points = fibonacci_sphere(radius, num_points)
    connectivity = spherical_triangulation(points)

    partitions, ghost_owners = distribute_mesh(points, connectivity, num_procs)

    for rank, (part_points, part_cells, num_owned, ghost_flags) in enumerate(partitions):
        filename = f"mesh_{rank}.vtu"
        write_vtu(filename, part_points, part_cells, num_owned, ghost_flags, ghost_owners)
        print(f"Written {filename} with {len(part_points)} points ({num_owned} owned, {len(part_points) - num_owned} ghosts) and {len(part_cells)} cells")

    # Optionally, print or return the ghost_owners map for debugging or further use
    # print("Ghost owners map:", ghost_owners)

if __name__ == "__main__":
    main()

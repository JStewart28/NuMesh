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

    # Create global-to-local mappings for each partition
    global_to_local = [{} for _ in range(num_procs)]
    for i in range(num_procs):
        part_indices = np.where(labels == i)[0]  # Get indices of points in this partition
        global_to_local[i] = {global_idx: local_idx for local_idx, global_idx in enumerate(part_indices)}

    partitions = []
    for i in range(num_procs):
        part_indices = np.where(labels == i)[0]  # Get indices of points in this partition
        part_points = points[part_indices]  # Extract corresponding points
        local_cells = []
        ghost_points = {}

        vertex_owner = np.full(len(part_points), i, dtype=int)  # Owner partition for owned points
        vertex_gid = np.array(part_indices, dtype=int)  # Global indices of owned points

        for cell in connectivity:
            cell_partitions = {labels[v] for v in cell}  # Find which partitions own this triangle

            if i in cell_partitions:  # Keep the triangle if it has any local points
                local_cell = []
                for v in cell:
                    if v in global_to_local[i]:
                        local_cell.append(global_to_local[i][v])  # Owned point
                    else:
                        if v not in ghost_points:
                            ghost_points[v] = len(part_points) + len(ghost_points)  # Assign new local index
                        local_cell.append(ghost_points[v])  # Ghost point index

                local_cells.append(local_cell)

        # Merge owned and ghost points
        ghost_indices = list(ghost_points.keys())
        full_points = np.vstack([part_points, points[ghost_indices]])

        # Assign vertex owners and global IDs for ghost points
        ghost_owner = np.array([labels[v] for v in ghost_indices], dtype=int)
        ghost_gid = np.array(ghost_indices, dtype=int)

        # Combined vertex owner and gid arrays
        full_vertex_owner = np.concatenate([vertex_owner, ghost_owner])
        full_vertex_gid = np.concatenate([vertex_gid, ghost_gid])

        # Ghost flags for visualization
        ghost_flags = np.zeros(len(full_points), dtype=np.uint8)
        ghost_flags[len(part_points):] = 1  # Mark ghost points

        partitions.append(
            (full_points, np.array(local_cells), len(part_points), ghost_flags, full_vertex_owner, full_vertex_gid))

    return partitions


def make_gids_contiguous(partitions):
    """ Remap global IDs so that each partition owns a contiguous section of the global ID space. """
    num_procs = len(partitions)

    # Compute new global ID offsets for each partition
    global_offset = 0
    new_global_ids = []
    for i in range(num_procs):
        num_owned = partitions[i][2]  # Number of owned vertices
        new_global_ids.append((global_offset, global_offset + num_owned))
        global_offset += num_owned  # Update offset for next partition

    # Create mapping from old global IDs to new global IDs
    global_id_map = {}
    for i in range(num_procs):
        old_gids = partitions[i][5][:partitions[i][2]]  # Only owned GIDs
        new_start = new_global_ids[i][0]
        for local_idx, old_gid in enumerate(old_gids):
            global_id_map[old_gid] = new_start + local_idx  # Assign contiguous global ID

    # Update partitions with new global IDs
    updated_partitions = []
    for i in range(num_procs):
        part_points, part_cells, num_owned, ghost_flags, vertex_owners, vertex_gids = partitions[i]

        # Remap global IDs
        new_vertex_gids = np.array([global_id_map.get(gid, gid) for gid in vertex_gids], dtype=int)

        updated_partitions.append((part_points, part_cells, num_owned, ghost_flags, vertex_owners, new_vertex_gids))

    return updated_partitions


def write_vtu(filename, points, connectivity, ghost_flags, vertex_owners, vertex_gids):
    """ Write a partitioned mesh to a VTU file, marking ghost points and assigning vertex metadata. """
    points_vtk = vtk.vtkPoints()
    for p in points:
        points_vtk.InsertNextPoint(p)

    cells_vtk = vtk.vtkCellArray()
    for cell in connectivity:
        triangle = vtk.vtkTriangle()
        for i in range(3):
            triangle.GetPointIds().SetId(i, int(cell[i]))
        cells_vtk.InsertNextCell(triangle)

    # Write ghost point flag array
    vtk_ghost_array = numpy_to_vtk(ghost_flags)
    vtk_ghost_array.SetName("ghost_points")

    # Write vertex owner array
    vtk_vertex_owner_array = numpy_to_vtk(vertex_owners)
    vtk_vertex_owner_array.SetName("vertex_owner")

    # Write vertex global ID array
    vtk_vertex_gid_array = numpy_to_vtk(vertex_gids)
    vtk_vertex_gid_array.SetName("vertex_gids")

    polydata = vtk.vtkUnstructuredGrid()
    polydata.SetPoints(points_vtk)
    polydata.SetCells(vtk.VTK_TRIANGLE, cells_vtk)
    polydata.GetPointData().AddArray(vtk_ghost_array)
    polydata.GetPointData().AddArray(vtk_vertex_owner_array)
    polydata.GetPointData().AddArray(vtk_vertex_gid_array)

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

    # if num_points % num_procs != 0:
    #     print("Error: num_points must be evenly divisible by num_procs.")
    #     sys.exit(1)

    points = fibonacci_sphere(radius, num_points)
    connectivity = spherical_triangulation(points)

    partitions = distribute_mesh(points, connectivity, num_procs)

    # Make global IDs contiguous
    partitions = make_gids_contiguous(partitions)

    for rank, (part_points, part_cells, num_owned, ghost_flags, vertex_owners, vertex_gids) in enumerate(partitions):
        filename = f"mesh_{rank}.vtu"
        write_vtu(filename, part_points, part_cells, ghost_flags, vertex_owners, vertex_gids)
        print(f"Wrote {filename} with {len(part_points)} points ({num_owned} owned, {len(part_points) - num_owned} ghosts) and {len(part_cells)} cells")


if __name__ == "__main__":
    main()

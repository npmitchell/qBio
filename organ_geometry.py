import os
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')  # Must come before importing pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
import open3d as o3d
import re
from natsort import natsorted
from scipy.spatial import cKDTree
import pyvista as pv
import copy

# Interactive portion of the tutorial
def align_mesh_icp(meshA, meshB, plotResult=True, ssfactor=5,  Alabel="A", Blabel="B"):
    """Align subsampled meshA to subsampled meshB using ICP. First aligns centroids.

    Parameters
    ----------
    meshA: pyvista mesh object
        Mesh to align, with vertices and faces
    meshB: pyvista mesh object
        Mesh to align to, with vertices and faces
    plotResult: bool (default=True)
        plot the results
    ssfactor: int (1 or larger)
        Subsampling factor. Larger values will speed up computation but potentially be less precise

    Returns
    -------
    meshA_t, T

    """
    vA = np.asarray(meshA.points)
    vA = subsample_vertices(vA, ssfactor)
    vB = np.asarray(meshB.points)
    vB = subsample_vertices(vB, ssfactor)

    if plotResult:
        # --- Save original for "before" visualization ---
        meshA_before = copy.deepcopy(meshA)

    # === Transform A to match B ===
    # First shift the centroids to match. This aids in ICP convergence.
    centroid_shift = vB.mean(axis=0) - vA.mean(axis=0)
    meshA_t = meshA.translate(centroid_shift, inplace=False)
    vA += centroid_shift
    # ICP transform and application
    print('Computing optimal transform via ICP...')
    T, _ = compute_icp_transform_o3d(vA, vB)
    meshA_t.transform(T, inplace=True)

    if plotResult:
        # Plot before alignment
        plotter = pv.Plotter()
        plotter.set_background("white")
        plotter.add_mesh(meshA_before, color="crimson", opacity=0.6, label=Alabel+"(original)")
        plotter.add_mesh(meshB, color="dodgerblue", opacity=0.6, label=Blabel)
        plotter.add_legend()
        plotter.show(interactive_update=True)

        # Plot after alignment
        plotter = pv.Plotter()
        plotter.set_background("white")
        plotter.add_mesh(meshA_t, color="crimson", opacity=0.6, label=Alabel+" (transformed)")
        plotter.add_mesh(meshB, color="dodgerblue", opacity=0.6, label=Blabel)
        plotter.add_legend()
        plotter.show(interactive_update=True)

    return meshA_t, T


# ----------------------
# STEP 4: Subsampling Meshes (for speed)
# ----------------------

def subsample_vertices(vertices, step=10):
    return vertices[::step]


# ----------------------
# STEP 5: Compute ICP Error Between Two Meshes
# ----------------------

def extract_tp(filename):
    """Extract the timepoint (integer) from a filename like mesh_000038_APDV_um.ply"""
    match = re.search(r"mesh_(\d+)_APDV_um\.ply", filename)
    return int(match.group(1)) if match else None

def compute_icp_transform_o3d(source_pts, target_pts, threshold=2.0, init=np.eye(4)):
    """
    Compute rigid ICP transform using Open3D (point-to-point).

    Parameters
    ----------
    source_pts : (N, 3) np.ndarray
        Source point cloud (to be aligned).
    target_pts : (M, 3) np.ndarray
        Target point cloud (fixed).
    threshold : float
        Distance threshold for correspondence matching.
    init : (4, 4) np.ndarray
        Initial transformation (default: identity).

    Returns
    -------
    transformation : (4, 4) np.ndarray
        Best-fit transformation matrix (target ← source).
    rmse : float
        Final root mean square error.
    """
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(source_pts)

    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(target_pts)

    result = o3d.pipelines.registration.registration_icp(
        src, tgt, threshold, init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    return result.transformation, result.inlier_rmse


# ----------------------
# STEP 6: Build ICP Cost Matrix
# ----------------------


def build_icp_cost_matrix(dirA, dirB, ssfactor=10, step=1, preview=False, flipy=False):
    """
    Build a matrix of ICP errors between meshes in two directories.

    Parameters
    ----------
    dirA : str
        Path to the directory containing source meshes (.ply).
    dirB : str
        Path to the directory containing target meshes (.ply).
    ssfactor : int, optional
        Subsampling factor for vertices (default is 10).
    step : int, optional
        Use every `step`-th file in both directories (default is 1, i.e. use all files).

    Returns
    -------
    np.ndarray
        2D matrix of ICP RMSE values, shape (num_A, num_B).
    """
    filesA = natsorted([os.path.join(dirA, f) for f in os.listdir(dirA) if f.endswith('.ply')])[::step]
    filesB = natsorted([os.path.join(dirB, f) for f in os.listdir(dirB) if f.endswith('.ply')])[::step]
    #
    # filesA = sorted([os.path.join(dirA, f) for f in os.listdir(dirA) if f.endswith('.ply')])[::step]
    # filesB = sorted([os.path.join(dirB, f) for f in os.listdir(dirB) if f.endswith('.ply')])[::step]

    num_A, num_B = len(filesA), len(filesB)
    icp_cost = np.full((num_A, num_B), np.nan)

    for i, fA in enumerate(filesA):
        print('Considering', fA)
        meshA = pv.read(fA)
        vA = np.asarray(meshA.points)
        vA = subsample_vertices(vA, ssfactor)

        for j, fB in enumerate(filesB):
            meshB = pv.read(fB)
            if flipy:
                meshB.points[:, 1] *= -1

            vB = np.asarray(meshB.points)
            vB = subsample_vertices(vB, ssfactor)

            centroid_shift = vB.mean(axis=0) - vA.mean(axis=0)
            vA += centroid_shift

            try:
                TAB, _ = compute_icp_transform_o3d(vA, vB, threshold=5.0)
                TBA, _ = compute_icp_transform_o3d(vB, vA, threshold=5.0)

                # Apply transforms
                vA2B = (TAB[:3, :3] @ vA.T).T + TAB[:3, 3]
                vB2A = (TBA[:3, :3] @ vB.T).T + TBA[:3, 3]

                # Match nearest neighbors and compute RMS distances
                treeB = cKDTree(vB)
                distsAB, _ = treeB.query(vA2B)

                treeA = cKDTree(vA)
                distsBA, _ = treeA.query(vB2A)

                # Average symmetric RMSE
                err_sym = 0.5 * (np.sqrt(np.mean(distsAB**2)) + np.sqrt(np.mean(distsBA**2)))

                icp_cost[i, j] = err_sym

                # Show overlay
                if preview:
                    show_icp_overlay_pv(vA, vB, transform=TAB, title=f"ICP Overlay (RMSE = {err_sym:.4f})")

            except Exception as e:
                print(f"Error computing ICP between {fA} and {fB}: {e}")

    return icp_cost


def show_icp_overlay_pv(source: np.ndarray, target: np.ndarray, transform: np.ndarray = None,
                        title: str = "ICP Overlay"):
    """
    Visualize the source and target point clouds using PyVista.

    Parameters
    ----------
    source : (N, 3) np.ndarray
        Source point cloud (will be transformed if `transform` is given).
    target : (M, 3) np.ndarray
        Target point cloud (fixed).
    transform : (4, 4) np.ndarray, optional
        4x4 homogeneous transform matrix to apply to source.
    title : str
        Title of the plot window.
    """
    if transform is not None:
        source_h = np.hstack([source, np.ones((len(source), 1))])
        source = (transform @ source_h.T).T[:, :3]

    cloud_source = pv.PolyData(source)
    cloud_target = pv.PolyData(target)

    plotter = pv.Plotter()
    plotter.add_points(cloud_target, color="blue", point_size=5, render_points_as_spheres=True, label="Target")
    plotter.add_points(cloud_source, color="red", point_size=5, render_points_as_spheres=True, label="Aligned Source")
    plotter.add_legend()
    plotter.add_title(title)
    plotter.show()


# ----------------------
# STEP 7: Visualize Cost Matrix
# ----------------------

def show_icp_matrix(matrix, title="ICP Matrix", tpsA=None, tpsB=None):
    """
    Display a heatmap of an ICP cost matrix using matplotlib.

    Parameters
    ----------
    matrix : np.ndarray
        2D array of ICP costs.
    title : str
        Title for the plot window.
    tpsA : list or np.ndarray, optional
        Labels for the rows (source timepoints).
    tpsB : list or np.ndarray, optional
        Labels for the columns (target timepoints).
"""
    plt.figure(figsize=(8, 6))
    im = plt.imshow(matrix, cmap='inferno', aspect='auto')
    plt.colorbar(im, label="ICP RMSE")

    if tpsA is not None:
        plt.yticks(np.arange(len(tpsA)), tpsA)
    if tpsB is not None:
        plt.xticks(np.arange(len(tpsB)), tpsB, rotation=90)

    plt.xlabel("Target Timepoints (B)")
    plt.ylabel("Source Timepoints (A)")
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


# ----------------------
# STEP 8: Smooth and Match Timepoints
# ----------------------

def smooth_icp_matrix(icp_matrix, window=5, polyorder=1):
    med = np.apply_along_axis(lambda m: savgol_filter(m, window, polyorder), axis=1, arr=icp_matrix)
    return np.apply_along_axis(lambda m: savgol_filter(m, window, polyorder), axis=0, arr=med)


def match_timepoints(smoothed_icp):
    """
    Match timepoints between two series of meshes using a precomputed cost matrix.

    AtoB[i] gives the index in series B that best matches meshA at tpsA[i].
    BtoA[j] gives the index in series A that best matches meshB at tpsB[j].

    Parameters
    ----------
    smoothed_icp : np.ndarray of shape (nA, nB)
        Cost matrix where each element [i, j] represents the matching cost
        (e.g., RMSE or alignment error) between meshA[i] and meshB[j].
        Lower values indicate better matches.

    Returns
    -------
    AtoB : np.ndarray of shape (nA,)
        Indices into mesh series B for best match to each timepoint in A.

    BtoA : np.ndarray of shape (nB,)
        Indices into mesh series A for best match to each timepoint in B.
    """
    AtoB = np.argmin(smoothed_icp, axis=1)
    BtoA = np.argmin(smoothed_icp, axis=0)
    return AtoB, BtoA

def dtw_match(cost_matrix):
    """
    Perform symmetric, monotonic matching using Dynamic Time Warping (DTW).

    Parameters
    ----------
    cost_matrix : np.ndarray of shape (nA, nB)
        ICP or other cost matrix. Lower values = better match.

    Returns
    -------
    path : list of tuple
        Optimal warp path as (i, j) index pairs, with i indexing into tpsA and j indexing into tpsB
    AtoB : np.ndarray
        For each index i in A, gives the B index j it aligns with.
    BtoA : np.ndarray
        For each index j in B, gives the A index i it aligns with.
    """
    nA, nB = cost_matrix.shape
    dtw = np.full((nA + 1, nB + 1), np.inf)
    dtw[0, 0] = 0

    for i in range(1, nA + 1):
        for j in range(1, nB + 1):
            cost = cost_matrix[i - 1, j - 1]
            dtw[i, j] = cost + min(dtw[i - 1, j],     # insertion
                                   dtw[i, j - 1],     # deletion
                                   dtw[i - 1, j - 1]) # match

    # Backtrack to find warp path
    i, j = nA, nB
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        directions = [dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1]]
        move = np.argmin(directions)
        if move == 0:
            i -= 1
        elif move == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    path.reverse()

    # Extract AtoB and BtoA from path (last match per i or j)
    AtoB = np.full(nA, -1, dtype=int)
    BtoA = np.full(nB, -1, dtype=int)
    for i, j in path:
        AtoB[i] = j
        BtoA[j] = i

    return path, AtoB, BtoA


# ----------------------
# STEP 9: PCA Smoothing
# ----------------------
def pca_smooth_correspondences(tpsA, tpsB, AtoB, BtoA, sg_window=11):
    """
    Smooth A↔B timepoint correspondences using PCA decomposition and Savitzky-Golay filtering.

    This function takes the matched timepoints between two series and smooths their relationship
    by projecting them into PCA space, smoothing along the orthogonal component, and reconstructing.

    Parameters
    ----------
    tpsA, tpsB : np.ndarray
        Timepoints from series A and B (e.g., extracted from filenames).
    AtoB, BtoA : np.ndarray
        Match indices: AtoB[i] = j means tpsA[i] matches tpsB[j]; BtoA[j] = i means tpsB[j] matches tpsA[i].
    sg_window : int
        Window size for Savitzky-Golay filter.

    Returns
    -------
    smoothed : np.ndarray of shape (N, 2)
        Smoothed (tpsA, tpsB) pairs, sorted by PCA arc-length.
    """
    # Step 1: collect all matched (tA, tB) pairs
    points = np.vstack((
        np.column_stack((tpsA, tpsB[AtoB])),
        np.column_stack((tpsA[BtoA], tpsB))
    )).astype(float)

    # Step 2: center and PCA
    mean_pts = points.mean(axis=0)
    points_centered = points - mean_pts
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(points_centered)

    # Step 3: sort by PC1 (arc-length direction)
    sort_idx = np.argsort(pcs[:, 0])
    pcs_sorted = pcs[sort_idx]

    # Step 4: smooth PC2 along PC1
    pcs_smoothed = pcs_sorted.copy()
    pcs_smoothed[:, 1] = savgol_filter(pcs_sorted[:, 1], sg_window, polyorder=1)

    # Step 5: invert PCA transform
    smoothed = pca.inverse_transform(pcs_smoothed) + mean_pts

    return smoothed

# ----------------------------------
# STEP 11: Baseline expected RMSE
# ----------------------------------


def mean_point_spacing(points):
    """
    Estimate the mean spacing between points in a point cloud.

    Parameters
    ----------
    points : (N, 3) np.ndarray
        The point cloud.

    Returns
    -------
    float
        Average nearest-neighbor distance.
    """
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)  # k=2: first is self, second is nearest neighbor
    return np.mean(dists[:, 1])  # skip self-distance


# ----------------------------------
# STEP 12: Movie of overlay
# ----------------------------------

def batch_icp_overlay(dirA, dirB, filesA, filesB, AtoB, outdir="icp_frames", ssfactor=10,
                      xyzlim=None, flipy=False, Alabel="A (transformed)", Blabel="B"):
    """
    Save individual PNG images showing A→B matched meshes with ICP transform applied to A.

    Parameters
    ----------
    dirA, dirB : str
        Directories containing .ply mesh files.
    filesA, filesB : list of str
        Sorted lists of .ply filenames.
    AtoB : list or np.ndarray
        Index mapping from A to best-matching B.
    outdir : str
        Directory to save output .png frames.
    ssfactor : int
        Subsampling factor for ICP computation.
    xyzlim : tuple of 3 tuples, optional
        ((xmin, xmax), (ymin, ymax), (zmin, zmax)) to fix plot bounds and avoid jitter.

    """
    os.makedirs(outdir, exist_ok=True)

    for i, j in enumerate(AtoB):
        print(f"Saving overlay image for A[{i}] → B[{j}]")

        # Load meshes
        pathA = os.path.join(dirA, filesA[i])
        pathB = os.path.join(dirB, filesB[j])
        meshA = pv.read(pathA)
        meshB = pv.read(pathB)

        if flipy:
            meshB.points[:, 1] *= -1

        # Subsample point clouds for ICP
        vA = meshA.points[::ssfactor]
        vB = meshB.points[::ssfactor]
        centroid_shift = vB.mean(axis=0) - vA.mean(axis=0)
        meshA_t = meshA.translate(centroid_shift, inplace=False)
        vA += centroid_shift

        # ICP transform and application
        T, _ = compute_icp_transform_o3d(vA, vB)
        meshA_t.transform(T, inplace=True)


        # Plot
        plotter = pv.Plotter(off_screen=True)
        plotter.set_background("white")
        plotter.add_mesh(meshA_t, color="crimson", opacity=0.6, label=Alabel)
        plotter.add_mesh(meshB, color="dodgerblue", opacity=0.6, label=Blabel)

        if xyzlim is not None:
            xlim, ylim, zlim = xyzlim
            bounds_box = pv.Cube(bounds=(xlim[0], xlim[1], ylim[0], ylim[1], zlim[0], zlim[1]))
            plotter.add_mesh(bounds_box, opacity=0.0)

        plotter.add_legend()
        # plotter.view_xy()
        # plotter.camera_position = [(1, 1, np.sqrt(3)/2), (0, 0, 0), (0, 0, 1)]

        # Save screenshot
        # fnA = os.path.splitext(os.path.basename(filesA[i]))[0]
        # fnB = os.path.splitext(os.path.basename(filesB[j]))[0]
        # png_path = os.path.join(outdir, f"overlay_{fnA}_{fnB}.png")
        png_path = os.path.join(outdir, f"overlay_{i:03d}.png")
        plotter.screenshot(png_path)
        plotter.close()

# ----------------------------------
# STEP 13: Color by distance from target mesh
# ----------------------------------
def color_mesh_by_distance(source_mesh: pv.PolyData, target_mesh: pv.PolyData,
                           transform=None, colormap="inferno", outfn=None):
    """
    Color the source mesh by its distance from the target mesh.

    Parameters
    ----------
    source_mesh : pv.PolyData
        Source mesh to be transformed and colored.
    target_mesh : pv.PolyData
        Target mesh (fixed).
    transform : (4, 4) np.ndarray, optional
        Homogeneous transform to apply to source mesh.
    colormap : str
        Name of matplotlib colormap to use.
    outfn : str
        Output filename for figure
    """
    src = source_mesh.copy()
    if transform is not None:
        src.transform(transform, inplace=True)

    src_pts = np.asarray(src.points)
    tgt_pts = np.asarray(target_mesh.points)

    tree = cKDTree(tgt_pts)
    dists, _ = tree.query(src_pts)

    # Normalize distances for colormap scaling (optional)
    norm_dists = dists / (np.max(dists + 1e-8))

    # Apply colormap to distances
    colors = plt.get_cmap(colormap)(norm_dists)[:, :3]  # drop alpha

    src.point_data["distance [µm]"] = dists
    src.point_data["colors"] = colors

    # Plot
    plotter = pv.Plotter()
    plotter.set_background("white")
    plotter.add_mesh(src, scalars="distance [µm]", cmap=colormap, show_scalar_bar=True)
    plotter.add_mesh(target_mesh, color="gray", opacity=0.2)
    plotter.add_title("Source Mesh Colored by Distance to Target")
    if outfn is not None:
        plotter.save_graphic(outfn)

    plotter.show(interactive_update=True)
    return dists, plotter

def compute_global_bounds(dirA, dirB, filesA, filesB):
    """
    Compute global axis-aligned bounds (x, y, z) across all meshes in two datasets.

    This function loads each mesh file in `filesA` and `filesB`, reads their spatial bounds,
    and returns the minimum and maximum extent along each coordinate axis.
    The resulting limits can be used to fix axis ranges in plots or visualizations to avoid
    frame-to-frame jitter.

    Parameters
    ----------
    dirA : str
        Directory containing mesh files for series A.
    dirB : str
        Directory containing mesh files for series B.
    filesA : list of str
        Filenames (not full paths) of .ply meshes in dirA.
    filesB : list of str
        Filenames (not full paths) of .ply meshes in dirB.

    Returns
    -------
    xlim : tuple of float
        (xmin, xmax) across all meshes.
    ylim : tuple of float
        (ymin, ymax) across all meshes.
    zlim : tuple of float
        (zmin, zmax) across all meshes.
    """
    all_bounds = []
    for fA, fB in zip(filesA, filesB):
        meshA = pv.read(os.path.join(dirA, fA))
        meshB = pv.read(os.path.join(dirB, fB))
        all_bounds.append(meshA.bounds)
        all_bounds.append(meshB.bounds)

    all_bounds = np.array(all_bounds)
    xmin = np.min(all_bounds[:, 0])
    xmax = np.max(all_bounds[:, 1])
    ymin = np.min(all_bounds[:, 2])
    ymax = np.max(all_bounds[:, 3])
    zmin = np.min(all_bounds[:, 4])
    zmax = np.max(all_bounds[:, 5])
    return (xmin, xmax), (ymin, ymax), (zmin, zmax)

def batch_color_by_distance(dirA, dirB, filesA, filesB, AtoB,
                            outdir="distance_colored", ssfactor=10,
                            save_vtk=False, save_png=True, colormap="viridis",
                            xyzlim=None, clim=None, flipy=False):
    """
    For each A→B match, color A's mesh by distance to B after ICP alignment and save results.

    Parameters
    ----------
    dirA, dirB : str
        Directories containing source (A) and target (B) .ply meshes.
    filesA, filesB : list of str
        Lists of filenames (just filenames, not full paths).
    AtoB : list or np.ndarray
        Indices of best-matching B mesh for each A.
    outdir : str
        Output directory for saving visualizations and/or VTK files.
    ssfactor : int
        Subsampling factor used during ICP.
    save_vtk : bool
        Whether to save the colored mesh with distance as a VTK file.
    save_png : bool
        Whether to save a rendered PNG of the colored mesh.
    colormap : str
        Matplotlib colormap to use for distance coloring.
    xyzlim : tuple of 3 tuples, optional
        ((xmin, xmax), (ymin, ymax), (zmin, zmax)) to fix plot bounds and avoid jitter.
    clim : tuple of float, optional
        Color range (vmin, vmax) for consistent scalar colormap across all frames.
    """
    os.makedirs(outdir, exist_ok=True)

    for i, j in enumerate(AtoB):
        print(f"[{i}] Coloring meshA {filesA[i]} by distance to meshB {filesB[j]}")

        pathA = os.path.join(dirA, filesA[i])
        pathB = os.path.join(dirB, filesB[j])
        meshA = pv.read(pathA)
        meshB = pv.read(pathB)

        if flipy:
            meshB.points[:, 1] *= -1

        # Subsample point clouds for ICP
        vA = meshA.points[::ssfactor]
        vB = meshB.points[::ssfactor]
        centroid_shift = vB.mean(axis=0) - vA.mean(axis=0)
        meshA_t = meshA.translate(centroid_shift, inplace=False)
        vA += centroid_shift

        # ICP transform and application
        T, _ = compute_icp_transform_o3d(vA, vB)
        meshA_t.transform(T, inplace=True)

        # Compute distances
        treeB = cKDTree(meshB.points)
        dists, _ = treeB.query(meshA_t.points)

        meshA_t["distance"] = dists  # scalar field

        # Save VTK file
        if save_vtk:
            vtk_path = os.path.join(outdir, f"meshA_{i:03d}_colored.vtk")
            meshA_t.save(vtk_path)
            print(f"  → Saved VTK: {vtk_path}")

        # Save PNG rendering
        if save_png:
            plotter = pv.Plotter(off_screen=True)
            plotter.set_background("white")
            plotter.add_mesh(meshA_t, scalars="distance", cmap=colormap,
                             clim=clim, show_scalar_bar=True)
            plotter.add_mesh(meshB, color="gray", opacity=0.3)

            if xyzlim is not None:
                xlim, ylim, zlim = xyzlim
                bounds_box = pv.Cube(bounds=(xlim[0], xlim[1], ylim[0], ylim[1], zlim[0], zlim[1]))
                plotter.add_mesh(bounds_box, opacity=0.0)

            # plotter.camera_position = [(400, -400, 400*np.sqrt(3)/2), (0, 0, 0), (0, 0, 1)]
            plotter.add_title(f"{dirA}[{i}] - {dirB}[{j}]")
            png_path = os.path.join(outdir, f"meshA_{i:03d}_colored.png")

            plotter.screenshot(png_path)
            plotter.close()
            print(f"  → Saved PNG: {png_path}")


# ----------------------
# STEP 10: Full Pipeline (Run this after setup)
# ----------------------

if __name__ == '__main__':
    # Example usage (uncomment and set correct paths):
    dirA = "HandGFPbynGAL4klar_UASmChCAAXHiFP/20240527/"
    dirB = "HandGFPbynGAL4klar_UASmChCAAXHiFP/20240531/"

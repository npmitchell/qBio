from os import mkdir

import matplotlib.pyplot as plt

from organ_geometry import *
import pandas as pd

# WT / OE --> first do without L/R
# Anterior part different
# Black box the details that are field-specific
# Get rid of sphere + ellipsoid


# Parameters
wt_oe = 'oe' # 'wt' 'oe' 'x1' 'x2' 'x3' 'x4'
step = 4      # subsampling in time
ssfactor = 10   # subsampling factor (take 1/N points for the ICP registration).
              # Larger values will run faster but be less precise. This is for speedup.
dt = 2         # timestep in minutes between adjacent timepoints
outdir = './results'
if not os.path.exists(outdir):
    mkdir(outdir)


conditions = ['wt', 'oe', 'x1', 'x2', 'x3', 'x4']

for i, wt_oe in enumerate(conditions):
    if wt_oe == 'wt':
        # dirA = "HandGFPbynGAL4klar_UASmChCAAXHiFP/20240527/"
        # dirB = "HandGFPbynGAL4klar_UASmChCAAXHiFP/20240531/"
        dirA = "wildtype/20240527/"
        dirB = "wildtype/20240531/"
        flipy = False
    elif wt_oe == 'oe':
        dirA = "bynGAL4_UASMyo1C/20240528/"
        dirB = "bynGAL4_UASMyo1C/20240626/"
        flipy = False
    elif wt_oe == 'x1':
        dirA = "wildtype/20240527/"
        dirB = "bynGAL4_UASMyo1C/20240528/"
        flipy = True
    elif wt_oe == 'x2':
        dirA = "wildtype/20240527/"
        dirB = "bynGAL4_UASMyo1C/20240626/"
        flipy = True
    elif wt_oe == 'x3':
        dirA = "wildtype/20240531/"
        dirB = "bynGAL4_UASMyo1C/20240528/"
        flipy = True
    elif wt_oe == 'x4':
        dirA = "wildtype/20240531/"
        dirB = "bynGAL4_UASMyo1C/20240626/"
        flipy = True


    # List and sort .ply files
    filesA = []
    for f in os.listdir(dirA):
        if f.endswith(".ply"):
            filesA.append(f)

    filesA = natsorted(filesA[::step])

    # Do the same for dirB
    filesB = []
    for f in os.listdir(dirB):
        if f.endswith(".ply"):
            filesB.append(f)

    filesB = natsorted(filesB[::step])

    # Extract timepoints from filenames
    tpsA = np.array([extract_tp(f) for f in filesA])
    tpsB = np.array([extract_tp(f) for f in filesB])

    # First verify that we get something
    icp_raw = build_icp_cost_matrix(dirA, dirB, ssfactor=ssfactor, step=step, flipy=flipy)
    # show_icp_matrix(icp_raw, "Raw ICP Error", tpsA, tpsB)
    icp_smooth = smooth_icp_matrix(icp_raw)
    # show_icp_matrix(icp_smooth, "Smoothed ICP Error", tpsA, tpsB)
    AtoB, BtoA = match_timepoints(icp_smooth)

    # Advanced: using dynamic time warping, we optimize with monotonicity
    [path, _, _] = dtw_match(icp_smooth)
    path = np.array(path)
    path_tps = np.array([tpsA[np.array(path)[:, 0]], tpsB[np.array(path)[:, 1]]]).T

    # These should now match the downsampled sizes
    assert tpsB.shape[0] == len(BtoA), "BtoA length must match # of points in B"
    assert tpsA.shape[0] == len(AtoB), "AtoB length must match # of points in A"

    # -----------------------------------------------
    # Plot the result on the RMSE (ICP mismatch) heatmap
    # -----------------------------------------------
    plt.figure()
    plt.imshow(icp_raw, cmap='inferno')
    plt.title("Timeline comparison via ICP")
    plt.xlabel("Time B")
    plt.ylabel("Time A")
    plt.colorbar(label="ICP Error")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


    # -----------------------------------------------
    # Plot the result on the RMSE (ICP mismatch) heatmap
    # -----------------------------------------------
    plt.figure()
    plt.imshow(icp_smooth, cmap='inferno')
    plt.plot(path[:,1], path[:, 0], '.-', lw=2, color='tab:purple')
    plt.plot(AtoB, np.arange(len(tpsA)), '.-', lw=2, color='tab:blue')
    plt.plot(np.arange(len(tpsB)), BtoA, '.-', lw=2, color='tab:orange')
    plt.title("Timepoint Matching")
    plt.xlabel("Time B")
    plt.ylabel("Time A")
    plt.colorbar(label="ICP Error")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'TPMatching.png'))
    plt.show()


    # -----------------------------------------------
    # Plot raw ICP error at matched A→B and B→A pairs
    # -----------------------------------------------
    # Plot the result
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # === Subplot 1: A→B ===
    ax1.plot(tpsA, icp_raw[np.arange(len(tpsA)), AtoB], 'o--', label='A→B raw', color='tab:blue')
    ax1.plot(tpsA, icp_smooth[np.arange(len(tpsA)), AtoB], 'o-', label='A→B smoothed', color='tab:blue')
    # ax1.plot(path_tps[:, 0], icp_smooth[path[:, 0], path[:, 1]], '*-', label='DTW smoothed (A→B)', color='tab:blue')
    ax1.set_xlabel("Timepoint A")
    ax1.set_ylabel("ICP Error")
    ax1.set_title("A→B Matching")
    ax1.grid(True)
    ax1.legend()

    # === Subplot 2: B→A ===
    ax2.plot(tpsB, icp_raw[BtoA, np.arange(len(tpsB))], 'o--', label='B→A raw', color='tab:orange')
    ax2.plot(tpsB, icp_smooth[BtoA, np.arange(len(tpsB))], 'o-', label='B→A smoothed', color='tab:orange')
    # ax2.plot(path_tps[:, 1], icp_smooth[path[:, 0], path[:, 1]], '*-', label='DTW smoothed (B→A)', color='tab:orange')
    ax2.set_xlabel("Timepoint B")
    ax2.set_title("B→A Matching")
    ax2.grid(True)
    ax2.legend()
    plt.savefig(os.path.join(outdir, 'TPMatching_AtoB_BtoA.png'))
    plt.show()


    # ------------------------------------------------
    # Discussion: what is the baseline expected error
    # ------------------------------------------------
    mean_spacing_A = []
    mean_spacing_B = []

    for fA in filesA:
        # Load and subsample point cloud
        vA = pv.read(os.path.join(dirA, fA)).points[::ssfactor]
        # Compute and store mean spacing
        mean_spacing_A.append(mean_point_spacing(vA))

    for fB in filesB:
        # Load and subsample point cloud
        vB = pv.read(os.path.join(dirB, fB)).points[::ssfactor]
        # Compute and store mean spacing
        mean_spacing_B.append(mean_point_spacing(vB))

    mean_spacing_A = np.array(mean_spacing_A)
    mean_spacing_B = np.array(mean_spacing_B)
    expected_rmse_A = mean_spacing_A / np.sqrt(2)
    expected_rmse_B = mean_spacing_B / np.sqrt(2)

    # Plot the result
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # === Subplot 1: A→B ===
    ax1.plot(tpsA, icp_raw[np.arange(len(tpsA)), AtoB], 'o--', label='A→B raw', color='tab:blue')
    ax1.plot(tpsA, icp_smooth[np.arange(len(tpsA)), AtoB], 'o-', label='A→B smoothed', color='tab:blue')
    ax1.plot(tpsA, expected_rmse_A, '.--', label='B→A baseline', color='tab:blue')
    # ax1.plot(path_tps[:, 0], icp_smooth[path[:, 0], path[:, 1]], '*-', label='DTW smoothed (A→B)', color='tab:blue')
    ax1.set_xlabel("Timepoint A")
    ax1.set_ylabel("ICP Error")
    ax1.set_title("A→B Matching")
    ax1.grid(True)
    ax1.legend()

    # === Subplot 2: B→A ===
    ax2.plot(tpsB, icp_raw[BtoA, np.arange(len(tpsB))], 'o--', label='B→A raw', color='tab:orange')
    ax2.plot(tpsB, icp_smooth[BtoA, np.arange(len(tpsB))], 'o-', label='B→A smoothed', color='tab:orange')
    ax2.plot(tpsB, expected_rmse_B, '.--', label='A→B baseline', color='tab:orange')
    # ax2.plot(path_tps[:, 1], icp_smooth[path[:, 0], path[:, 1]], '*-', label='DTW smoothed (B→A)', color='tab:orange')
    ax2.set_xlabel("Timepoint B")
    ax2.set_title("B→A Matching")
    ax2.grid(True)
    ax2.legend()
    plt.savefig(os.path.join(outdir, 'TPMatching_AtoB_BtoA_expectedRMSE.png'))
    plt.show()

    # --------------------------------------
    # Visualize the overlay between two timepoints
    # --------------------------------------
    # Load meshes
    tp2view = int(np.median(tpsA))
    tidx = np.argmin(np.abs(tpsA - tp2view))

    meshA = pv.read(os.path.join(dirA, filesA[tidx]))
    meshB = pv.read(os.path.join(dirB, filesB[AtoB[tidx]]))

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
    plotter = pv.Plotter()
    plotter.set_background("white")
    plotter.add_mesh(meshA_t, color="crimson", opacity=0.6, label="A (transformed)")
    plotter.add_mesh(meshB, color="dodgerblue", opacity=0.6, label="B")
    plotter.add_legend()
    plotter.save_graphic(os.path.join(outdir, 'example_overlay.pdf'))
    plotter.show(interactive_update=True)
    plotter.close()

    # --------------------------------------
    # Color by distance between the two
    # --------------------------------------
    # Show distance coloring
    outfn = os.path.join(outdir, 'example_mesh_by_distance.pdf')
    color_mesh_by_distance(meshA_t, meshB, transform=None, outfn=outfn)

    # --------------------------------------
    # Batch overlays
    # --------------------------------------
    xyzlim = compute_global_bounds(dirA, dirB, filesA, filesB)
    batch_icp_overlay(dirA, dirB, filesA, filesB, AtoB, outdir="icp_overlay_"+wt_oe,
                      ssfactor=ssfactor, xyzlim=xyzlim, flipy=flipy,
                      Alabel=dirA, Blabel=dirB)

    # --------------------------------------
    # Color the meshes by mismatch distance en masse
    # --------------------------------------
    clims = (0, 20)
    batch_color_by_distance(dirA, dirB, filesA, filesB, AtoB,
                            outdir="colored_distance_"+wt_oe,
                            ssfactor=ssfactor, xyzlim=xyzlim,
                            clim=clims, flipy=flipy)


    # --------------------------------------
    # Advanced: an alternative to DTW is using both AtoB and
    # BtoA to create consensus in PCA1 space.
    # --------------------------------------
    smoothed_curve = pca_smooth_correspondences(tpsA, tpsB, AtoB, BtoA)

    plt.figure()
    plt.imshow(icp_smooth, cmap='inferno', extent=[tpsB[0], tpsB[-1], tpsA[0], tpsA[-1]],
               origin='lower', aspect='auto')
    plt.plot(smoothed_curve[:, 1], smoothed_curve[:, 0], 'r.-', lw=2)
    plt.title("PCA-Smoothed Timepoint Matching")
    plt.xlabel("Time B")
    plt.ylabel("Time A")
    plt.colorbar(label="ICP Error")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'TPMatching_PCASm.png'))
    plt.show()
    plt.close()

    # Save the results
    icpRawA = icp_raw[np.arange(len(tpsA)), AtoB]
    icpSmoothA = icp_smooth[np.arange(len(tpsA)), AtoB]
    icpRawB = icp_raw[BtoA, np.arange(len(tpsB))]
    icpSmoothB = icp_smooth[BtoA, np.arange(len(tpsB))]
    df_icp = pd.DataFrame({
        "tpsA": tpsA,
        "AtoB": AtoB,
        "icpRawA": icpRawA,
        "icpSmoothA": icpSmoothA
    })

    df_icp_B = pd.DataFrame({
        "tpsB": tpsB,
        "BtoA": BtoA,
        "icpRawB": icpRawB,
        "icpSmoothB": icpSmoothB
    })

    df_icp.to_csv(os.path.join(outdir, f"icp_error_AtoB_{wt_oe}_step{step}_ss{ssfactor}.csv"), index=False)
    df_icp_B.to_csv(os.path.join(outdir, f"/icp_error_BtoA_{wt_oe}_step{step}_ss{ssfactor}.csv"), index=False)

    np.save(os.path.join(outdir, f"icp_raw_{wt_oe}_step{step}_ss{ssfactor}.npy"), icp_raw)
    np.save(os.path.join(outdir, f"icp_smooth_{wt_oe}_step{step}_ss{ssfactor}.npy"), icp_smooth)
    np.save(os.path.join(outdir, f"expected_rmse_A_{wt_oe}_step{step}_ss{ssfactor}.npy"), expected_rmse_A)
    np.save(os.path.join(outdir, f"expected_rmse_B_{wt_oe}_step{step}_ss{ssfactor}.npy"), expected_rmse_B)
    np.save(os.path.join(outdir, f"path_{wt_oe}_step{step}_ss{ssfactor}.npy"), [path, path_tps])

# ------------------------------------------------------------
# Compare difference between WT and OE against
# diff within WT and within OE
# ------------------------------------------------------------

conds = ['wt', 'oe', 'x1', 'x2', 'x3', 'x4']
icp_dict = {}

for cond in conds:
    fname = f"icp_error_AtoB_{cond}_step{step}_ss{ssfactor}.csv"
    df = pd.read_csv(fname)
    icp_dict[cond] = {
        "tps": df["tpsA"].values,
        "error": df["icpSmoothA"].values
    }

# Align on common timepoints
common_tps = np.intersect1d(icp_dict['wt']['tps'], icp_dict['oe']['tps'])
for key in icp_dict:
    mask = np.isin(icp_dict[key]["tps"], common_tps)
    icp_dict[key]["tps"] = icp_dict[key]["tps"][mask]
    icp_dict[key]["error"] = icp_dict[key]["error"][mask]

# Stack WT vs OE error
wt_err = icp_dict['wt']["error"]
oe_err = icp_dict['oe']["error"]
tps = icp_dict['wt']["tps"]

# Mean between-group difference
diff_wtoe = np.abs(wt_err - oe_err)
mean_diff = np.mean(diff_wtoe)

# Stack X1–X4 errors
x_errors = np.stack([icp_dict[f"x{i}"]["error"] for i in range(1, 5)])
x_mean = np.mean(x_errors, axis=0)
x_std = np.std(x_errors, axis=0)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(tps*dt, wt_err, 'b.-', label="WT↔WT")
plt.plot(tps*dt, oe_err, 'r.-', label="OE↔OE")
plt.plot(tps*dt, x_mean, 'k-', label="WT↔OE (mean x1–x4)")
plt.fill_between(tps*dt, x_mean - x_std, x_mean + x_std, color='gray', alpha=0.3, label="WT↔OE ± std")
plt.xlabel("reference time [min]")
plt.ylabel("ICP RMSE (smoothed)")
plt.title("Within- vs cross-ensemble ICP RMSE")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'cross_ensemble_RMSE.png'))
plt.show()

print(f"Mean WT↔OE diff: {mean_diff:.3f}")
print(f"WT internal std: {np.std(wt_err):.3f}")
print(f"OE internal std: {np.std(oe_err):.3f}")

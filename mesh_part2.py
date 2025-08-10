from organ_geometry import *
import pandas as pd

# Main parameters
step = 3           # downsampling step in units of timepoints
                   # (higher will run faster and consider fewer timepoints)
ssfactor = 10      # spatial subsampling factor for ICP
                   # (higher is faster but potentially less accurate)
wt_oe = 'wt'       # comparison will initially be between two WT datasets
outdir = 'results' # where to store output on disk
dt = 2             # timestep in minutes between adjacent timepoints (this should not be changed, taken from experiments)

if not os.path.exists(outdir):
    os.mkdir(outdir)

dirA = "wildtype/20240527/"
dirB = "wildtype/20240531/"
flipy = False

# --------------------------------------
# --------------------------------------

# Collect all PLY files and extract their timepoint numbers
# List and sort .ply files
filesA = []
for f in os.listdir(dirA):
    if f.endswith(".ply"):
        filesA.append(f)

filesA = natsorted(filesA)[::step]

# Do the same for dirB
filesB = []
for f in os.listdir(dirB):
    if f.endswith(".ply"):
        filesB.append(f)

filesB = natsorted(filesB)[::step]

# Extract timepoints from filenames
tpsA = np.array([extract_tp(f) for f in filesA])
tpsB = np.array([extract_tp(f) for f in filesB])

# Compute ICP RMSE
icp_raw = build_icp_cost_matrix(dirA, dirB, ssfactor=ssfactor, step=step, flipy=flipy)
icp_smooth = smooth_icp_matrix(icp_raw)

# --------------------------------------
# --------------------------------------



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
im1 = ax1.imshow(icp_raw, cmap='inferno')
ax1.set_title("Timeline comparison via ICP")
ax1.set_xlabel("Time B")
ax1.set_ylabel("Time A")
fig.colorbar(im1, ax=ax1, label="ICP Error")

im2 = ax2.imshow(icp_smooth, cmap='inferno')
ax2.set_title("Timeline comparison via ICP")
ax2.set_xlabel("Time B")
ax2.set_ylabel("Time A")
fig.colorbar(im2, ax=ax2, label="Smoothed ICP Error")

plt.show()



# --------------------------------------
# --------------------------------------
# Match timepoints
AtoB, BtoA = match_timepoints(icp_smooth)

# Advanced: using dynamic time warping, we optimize with monotonicity
[path, _, _] = dtw_match(icp_smooth)
path = np.array(path)
path_tps = np.array([tpsA[path[:, 0]], tpsB[path[:, 1]]]).T


# --------------------------------------
# --------------------------------------

plt.figure()
plt.imshow(icp_smooth, cmap='inferno', extent=[tpsB[0], tpsB[-1], tpsA[0], tpsA[-1]],
           origin='lower', aspect='auto')
plt.plot(tpsB[path[:,1]], tpsA[path[:, 0]], '.-', lw=2, color='tab:purple')
plt.plot(tpsB[AtoB], tpsA, '.-', lw=2, color='tab:blue')
plt.plot(tpsB, tpsA[BtoA], '.-', lw=2, color='tab:orange')
plt.title("Timepoint Matching")
plt.xlabel("Time B")
plt.ylabel("Time A")
plt.colorbar(label="Smoothed ICP Error")
plt.axis('equal')
plt.tight_layout()
plt.show()


# --------------------------------------
# --------------------------------------


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
plt.savefig(os.path.join(outdir, f'TPMatching_AtoB_BtoA_{wt_oe}_step{step}_ss{ssfactor}.png'))
plt.show()



# --------------------------------------
# --------------------------------------

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



# --------------------------------------
# --------------------------------------



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
plt.savefig(os.path.join(outdir, f'TPMatching_AtoB_BtoA_expectedRMSE_{wt_oe}_step{step}_ss{ssfactor}.png'))
plt.show()


# --------------------------------------
# --------------------------------------

# --------------------------------------
# Batch overlays
# --------------------------------------
xyzlim = compute_global_bounds(dirA, dirB, filesA, filesB)
batch_icp_overlay(dirA, dirB, filesA, filesB, AtoB,
                  outdir=os.path.join(outdir, f"icp_overlay_{wt_oe}_step{step}_ss{ssfactor}"),
                  ssfactor=ssfactor, xyzlim=xyzlim, flipy=flipy,
                  Alabel=dirA, Blabel=dirB)
# --------------------------------------
# --------------------------------------

# --------------------------------------
# Batch color the meshes by mismatch distance
# --------------------------------------
clims = (0, 20)
batch_color_by_distance(dirA, dirB, filesA, filesB, AtoB,
                        outdir=os.path.join(outdir, f"colored_distance__{wt_oe}_step{step}_ss{ssfactor}"),
                        ssfactor=ssfactor, xyzlim=xyzlim,
                        clim=clims, flipy=flipy)

# --------------------------------------
# --------------------------------------

# Advanced: an alternative to DTW is using both AtoB and
# BtoA to create consensus in PCA1 space.
# --------------------------------------
smoothed_curve = pca_smooth_correspondences(tpsA, tpsB, AtoB, BtoA)


# --------------------------------------
# --------------------------------------
plt.figure()
plt.imshow(icp_smooth, cmap='inferno', extent=[tpsB[0], tpsB[-1], tpsA[0], tpsA[-1]],
           origin='lower', aspect='auto')
plt.plot(path_tps[:,1], path_tps[:, 0], '.-', lw=2, color='tab:purple')
plt.plot(tpsB[AtoB], tpsA, '.-', lw=2, color='tab:blue')
plt.plot(tpsB, tpsA[BtoA], '.-', lw=2, color='tab:orange')
plt.plot(smoothed_curve[:, 1], smoothed_curve[:, 0], 'r.-', lw=2)
plt.title("PCA-Smoothed Timepoint Matching")
plt.xlabel("Time B")
plt.ylabel("Time A")
plt.colorbar(label="ICP Error")
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(outdir, f'TPMatching_PCASm_{wt_oe}_step{step}_ss{ssfactor}.png'))
plt.show()




# --------------------------------------
# --------------------------------------
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

df_icp.to_csv(f"icp_error_AtoB_{wt_oe}_step{step}_ss{ssfactor}.csv", index=False)
df_icp_B.to_csv(f"icp_error_BtoA_{wt_oe}_step{step}_ss{ssfactor}.csv", index=False)

# --------------------------------------
# --------------------------------------
np.save(os.path.join(outdir, f"icp_raw_{wt_oe}_step{step}_ss{ssfactor}.npy"), icp_raw)
np.save(os.path.join(outdir, f"icp_smooth_{wt_oe}_step{step}_ss{ssfactor}.npy"), icp_smooth)
np.save(os.path.join(outdir, f"expected_rmse_A_{wt_oe}_step{step}_ss{ssfactor}.npy"), expected_rmse_A)
np.save(os.path.join(outdir, f"expected_rmse_B_{wt_oe}_step{step}_ss{ssfactor}.npy"), expected_rmse_B)
np.save(os.path.join(outdir, f"path_{wt_oe}_step{step}_ss{ssfactor}.npy"), [path, path_tps])

# --------------------------------------
# --------------------------------------


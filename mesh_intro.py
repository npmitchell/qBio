# Introductory Python Module: Mesh Triangulations and ICP Matching for Biology

# This notebook introduces core concepts in scientific programming using Python,
# with applications to comparing 3D biological shapes across time.
# We will start from the basics and build up to aligning mesh surfaces using ICP.
# After completing this tutorial, advance to Steps 4-onward in main.py

# ----------------------
# STEP 1: Imports and Setup
# ----------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pyvista as pv
from mesh import Mesh

# ----------------------
# STEP 2: Understanding Triangulations
# ----------------------

# ----------------------
# 2a: single triangle
# ----------------------
# Let's visualize a single triangle
x = np.array([0, 1, 0])
y = np.array([0, 0, 1])
faces = [[0, 1, 2]]

# plt.figure()
# plt.triplot(x, y, faces, color='black')
# plt.plot(x, y, 'o', color='blue')
# plt.title('A Single Triangle')
# plt.axis('equal')
# plt.show()

# ----------------------
# 2b: square
# ----------------------
# Let's visualize a simple triangulation of a square (2D)
x = np.array([0, 1, 0, 1])
y = np.array([0, 0, 1, 1])
faces = [[0, 1, 2], [1, 3, 2]]

# plt.figure()
# plt.triplot(x, y, faces, color='black')
# plt.plot(x, y, 'o', color='blue')
# plt.title('2D Triangulation of a Square')
# plt.axis('equal')
# plt.show()


# ----------------------
# 2c: cube
# ----------------------
# Make a triangulation of a cube

# Define the vertices of the cube
vertices = np.array([
    [0, 0, 0],  # 0
    [1, 0, 0],  # 1
    [1, 1, 0],  # 2
    [0, 1, 0],  # 3
    [0, 0, 1],  # 4
    [1, 0, 1],  # 5
    [1, 1, 1],  # 6
    [0, 1, 1]   # 7
])

# Define faces (two per face, 12 faces total)
faces = [
    [0, 2, 1], [0, 3, 2],  # bottom face
    [4, 5, 6], [4, 6, 7],  # top face
    [0, 1, 5], [0, 5, 4],  # front face
    [2, 3, 7], [2, 7, 6],  # back face
    [0, 7, 3], [0, 4, 7],  # left face
    [1, 2, 6], [1, 6, 5]   # right face
]

# Set up the figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a collection of faces for plotting
tri_faces = [[vertices[vertex] for vertex in tri] for tri in faces]

# Plot the triangulated cube
ax.add_collection3d(Poly3DCollection(tri_faces,
                                     facecolors='cyan',
                                     linewidths=1,
                                     edgecolors='black',
                                     alpha=0.8))

# Plot vertices as points
ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], color='blue')

# Set plot limits
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set equal aspect ratio
ax.set_box_aspect([1, 1, 1])
plt.title('3D Triangulation of a Cube')
plt.show()

# ----------------------
# ADVANCED: Check if mesh is watertight
# ----------------------
# Initialize an empty dictionary to count edges
edge_count = {}

# Loop over all faces
for tri in faces:
    # Extract edges from the triangle
    edges = [
        (min(tri[0], tri[1]), max(tri[0], tri[1])),
        (min(tri[1], tri[2]), max(tri[1], tri[2])),
        (min(tri[2], tri[0]), max(tri[2], tri[0]))
    ]

    # Count how many times each edge appears
    for edge in edges:
        if edge in edge_count:
            edge_count[edge] += 1
        else:
            edge_count[edge] = 1

# Collect edges that appear ≠ 2 times (open or non-manifold edges)
open_edges = []
for edge in edge_count:
    if edge_count[edge] != 2:
        open_edges.append((edge, edge_count[edge]))

# Report
if len(open_edges) == 0:
    print("✅ The mesh is closed (watertight).")
else:
    print("❌ The mesh is NOT closed. Problematic edges:")
    for edge, count in open_edges:
        print(f"  Edge {edge} appears {count} times")


# Alternative -- Challenge: how does this check work?
mm = Mesh(vertices, faces)
mm.is_closed()

# ----------------------
# ADVANCED: Check if faces are oriented correctly
# ----------------------
# Compute cube center
center = np.mean(vertices, axis=0)

def is_outward_facing(tri):
    v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
    # Compute normal vector
    normal = np.cross(v1 - v0, v2 - v0)
    # Vector from the center of the triangle to center of the cube
    tri_center = (v0 + v1 + v2) / 3
    to_center = center - tri_center
    # Dot product tells us if the normal is pointing toward or away from the center
    return np.dot(normal, to_center) < 0  # Should be negative if pointing outward

# Check all faces
inward_facing = []
for i, tri in enumerate(faces):
    if not is_outward_facing(tri):
        inward_facing.append(i)

# Report
if len(inward_facing) == 0:
    print("✅ All faces are consistently outward-facing.")
else:
    print("❌ Found inward-facing faces at indices:")
    print(inward_facing)

# --------------
# Alternative -- Challenge: how does this check work?
# --------------
mm = Mesh(vertices, faces)
mm.is_closed()

# To fix face orientations using Mesh() class, do the following:
mm.make_normals()
mm.force_z_normal(direction=1)


# ----------------------
# STEP 3: Working with 3D Meshes
# ----------------------
m = pv.read('./HandGFPbynGAL4klar_UASmChCAAXHiFP/20240527/mesh_000000_APDV_um.ply')
print(m)
m.plot(show_edges=True)
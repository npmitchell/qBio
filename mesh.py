# mesh_modern.py
# Modern Python 3.13-compatible version of Mesh class from mesh.py

import numpy as np
import struct
import os
from typing import Union, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Mesh:
    def __init__(self, points: Optional[np.ndarray] = np.zeros((0, 3)),
                 triangles: Optional[np.ndarray] = None):
        """Initialize Mesh object from points and triangles or from a .ply file."""
        self.points: np.ndarray = np.zeros((0, 3))
        self.triangles: np.ndarray = np.zeros((0, 3), dtype=np.uint32)
        self.normals: Optional[np.ndarray] = None
        self.vertex_normals: Optional[np.ndarray] = None

        if isinstance(points, str):
            ext = os.path.splitext(points)[1].lower()
            if ext == ".ply":
                self._load_ply(points)
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
        else:
            self.points = np.array(points)
            if triangles is not None:
                self.triangles = np.array(triangles, dtype=np.uint32)

    def _load_ply(self, filepath: str):
        """Load mesh data from a .ply file."""
        with open(filepath, 'r') as f:
            if 'ply' not in f.readline():
                raise ValueError("Not a valid PLY file.")

            line = ''
            num_v = num_f = 0
            while 'end_header' not in line:
                line = f.readline()
                if line.startswith('element vertex'):
                    num_v = int(line.split()[-1])
                elif line.startswith('element face'):
                    num_f = int(line.split()[-1])

            self.points = np.zeros((num_v, 3))
            normals_included = False
            for i in range(num_v):
                vals = list(map(float, f.readline().split()))
                self.points[i] = vals[:3]
                if len(vals) == 6:
                    if self.vertex_normals is None:
                        self.vertex_normals = np.zeros_like(self.points)
                    self.vertex_normals[i] = vals[3:]
                    normals_included = True

            self.triangles = np.zeros((num_f, 3), dtype=np.uint32)
            for i in range(num_f):
                vals = list(map(int, f.readline().split()))
                if vals[0] != 3:
                    raise ValueError("Only triangular faces are supported.")
                self.triangles[i] = vals[1:]

    def is_closed(self, tol: float = 1e-12) -> bool:
        """Check if the mesh is watertight (closed) using volume invariance under translation."""
        if self.points.size == 0 or self.triangles.size == 0:
            return False

        m2 = self.copy()
        bounds = np.ptp(self.points, axis=0)
        m2.points += 2 * bounds
        v1 = self.volume()
        v2 = m2.volume()
        return abs((v1 - v2) / v1) < tol

    def copy(self) -> 'Mesh':
        """Return a deep copy of the mesh."""
        return Mesh(self.points.copy(), self.triangles.copy())

    def tps(self, i: int) -> np.ndarray:
        """Return the i-th vertex of each triangle."""
        return self.points[self.triangles[:, i]]

    def volume(self) -> float:
        """Compute the signed volume enclosed by the mesh."""
        px, py, pz = self.tps(0).T
        qx, qy, qz = self.tps(1).T
        rx, ry, rz = self.tps(2).T
        vol = (px * qy * rz + py * qz * rx + pz * qx * ry
             - px * qz * ry - py * qx * rz - pz * qy * rx).sum() / 6.0
        return vol

    def make_normals(self, normalize: bool = True) -> np.ndarray:
        """Compute triangle face normals."""
        n = np.cross(self.tps(2) - self.tps(0), self.tps(1) - self.tps(0)).astype(float)  # <-- cast here
        if normalize:
            norms = np.linalg.norm(n, axis=1, keepdims=True)
            norms[norms == 0] = 1
            n /= norms
        self.normals = n
        return n

    def make_vertex_normals(self) -> np.ndarray:
        """Compute per-vertex normals by averaging adjacent face normals."""
        if self.normals is None:
            self.make_normals()

        self.vertex_normals = np.zeros_like(self.points)
        for i, tri in enumerate(self.triangles):
            for j in tri:
                self.vertex_normals[j] += self.normals[i]
        norms = np.linalg.norm(self.vertex_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.vertex_normals /= norms
        return self.vertex_normals

    def inverted(self) -> 'Mesh':
        """Return a new mesh with triangle winding order reversed."""
        return Mesh(self.points.copy(), self.triangles[:, ::-1].copy())

    def translate(self, offset: np.ndarray) -> 'Mesh':
        """Return a translated copy of the mesh."""
        return Mesh(self.points + offset, self.triangles.copy())

    def scale(self, s: Union[float, np.ndarray]) -> 'Mesh':
        """Return a scaled copy of the mesh."""
        s = np.asarray(s)
        if s.ndim == 0:
            s = np.full(3, s)
        return Mesh(self.points * s, self.triangles.copy())

    def __add__(self, other: 'Mesh') -> 'Mesh':
        """Combine two meshes by concatenating their vertices and triangles."""
        if hasattr(other, 'points') and hasattr(other, 'triangles'):
            return Mesh(
                points=np.vstack((self.points, other.points)),
                triangles=np.vstack((self.triangles, other.triangles + len(self.points)))
            )
        else:
            raise TypeError('Can only add a Mesh to another Mesh')

    def save_ply(self, filepath: str, save_normals: bool = False):
        """Save mesh as an ASCII PLY file."""
        with open(filepath, 'w') as f:
            if save_normals and self.vertex_normals is not None:
                f.write(f"""ply
format ascii 1.0
element vertex {len(self.points)}
property float x
property float y
property float z
property float nx
property float ny
property float nz
element face {len(self.triangles)}
property list uchar int vertex_indices
end_header
""")
                for p, n in zip(self.points, self.vertex_normals):
                    f.write(f"{p[0]} {p[1]} {p[2]} {n[0]} {n[1]} {n[2]}\n")
            else:
                f.write(f"""ply
format ascii 1.0
element vertex {len(self.points)}
property float x
property float y
property float z
element face {len(self.triangles)}
property list uchar int vertex_indices
end_header
""")
                for p in self.points:
                    f.write(f"{p[0]} {p[1]} {p[2]}\n")
            for tri in self.triangles:
                f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

    def save_stl(self, filepath: str):
        """Save mesh as a binary STL file."""
        with open(filepath, 'wb') as f:
            header = b'STL generated by Mesh class' + b' ' * (80 - len('STL generated by Mesh class'))
            f.write(header)
            f.write(struct.pack('<I', len(self.triangles)))
            self.make_normals()
            for tri, normal in zip(self.triangles, self.normals):
                f.write(struct.pack('<3f', *normal))
                for idx in tri:
                    f.write(struct.pack('<3f', *self.points[idx]))
                f.write(struct.pack('<H', 0))

    def force_z_normal(self, direction: int = 1):
        """Ensure face normals point in the positive or negative Z direction."""
        if self.normals is None:
            self.make_normals()
        z_sign = np.sign(self.normals[:, 2])
        flip = np.where(z_sign != np.sign(direction))[0]
        self.triangles[flip] = self.triangles[flip, ::-1]
        self.normals[flip] *= -1

    def show(self):
        """Display the mesh using matplotlib 3D plotting."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        faces = [self.points[tri] for tri in self.triangles]
        mesh = Poly3DCollection(faces, alpha=0.6, edgecolor='k')
        ax.add_collection3d(mesh)
        ax.scatter(*self.points.T, color='blue', s=1)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.tight_layout()
        plt.show()

    def save(self, fn: str, ext: str = None):
        """Save the mesh to file, dispatching by file extension."""
        if ext is None:
            ext = os.path.splitext(fn)[-1][1:]

        ext = ext.lower()
        if ext == 'ply' or fn.endswith('.ply'):
            self.save_ply(fn)
        elif ext == 'stl' or fn.endswith('.stl'):
            self.save_stl(fn)
        elif ext == 'xml' or fn.endswith('.xml'):
            self.save_xml(fn)
        else:
            raise ValueError('Extension should be "stl", "ply", or "xml" for outputting mesh')

    def save_xml(self, fname: str):
        """Save the mesh as an XML file compatible with FEniCS."""
        pts = self.points
        tri = self.triangles

        is2d = pts.shape[1] == 2

        with open(fname, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n\n')
            f.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')
            f.write('  <mesh celltype="triangle" dim="2">\n')
            f.write(f'    <vertices size="{len(pts)}">\n')
            for i, pt in enumerate(pts):
                if is2d:
                    f.write(f'      <vertex index="{i}" x="{pt[0]:.9E}" y="{pt[1]:.9E}"/>\n')
                else:
                    f.write(f'      <vertex index="{i}" x="{pt[0]:.9E}" y="{pt[1]:.9E}" z="{pt[2]:.9E}"/>\n')
            f.write('    </vertices>\n')
            f.write(f'    <cells size="{len(tri)}">\n')
            for i, t in enumerate(tri):
                f.write(f'      <triangle index="{i}" v0="{t[0]}" v1="{t[1]}" v2="{t[2]}"/>\n')
            f.write('    </cells>\n')
            f.write('  </mesh>\n')
            f.write('</dolfin>')

    def relax_z(self, fixed=None, steps=5):
        """Relax mesh geometry along Z using a spring network approximation."""
        oz = self.points[:, 2].copy()
        N = len(self.points)
        K = {}

        def dist(p1, p2):
            return np.linalg.norm(self.points[p1, :2] - self.points[p2, :2])

        for t in self.triangles:
            a = dist(t[1], t[2])
            b = dist(t[2], t[0])
            c = dist(t[0], t[1])
            s = (a + b + c) / 2
            A = max(np.sqrt(s * (s - a) * (s - b) * (s - c)), 1e-8)

            for (i1, i2, term) in [
                (t[1], t[2], (-a**2 + b**2 + c**2)),
                (t[2], t[0], (a**2 - b**2 + c**2)),
                (t[0], t[1], (a**2 + b**2 - c**2))]:
                p1, p2 = sorted((i1, i2))
                K[(p1, p2)] = K.get((p1, p2), 0.) + term / A

        tK = np.zeros(N)
        for (p1, p2), W in K.items():
            tK[p1] += W
            tK[p2] += W

        tK[tK == 0] = 1

        for _ in range(steps):
            z = np.zeros(N)
            for (p1, p2), W in K.items():
                z[p1] += W * oz[p2]
                z[p2] += W * oz[p1]
            z /= tK
            if fixed is not None:
                z[fixed] = self.points[fixed, 2]
            oz = z

        self.points[:, 2] = oz

    def merge_points(self, tol: float = 1e-10, verbose: bool = False):
        """Remove duplicate or near-duplicate vertices using distance threshold."""
        new = np.zeros((len(self.points), 3))
        p_map = np.zeros(len(self.points), dtype=int)
        j = 0
        for i, p in enumerate(self.points):
            if j == 0:
                new[j] = p
                p_map[i] = j
                j += 1
            else:
                dists = np.linalg.norm(new[:j] - p, axis=1)
                j_min = np.argmin(dists)
                if dists[j_min] < tol:
                    p_map[i] = j_min
                else:
                    new[j] = p
                    p_map[i] = j
                    j += 1
        if verbose:
            print(f'Merged {len(self.points) - j} points.')
        self.points = new[:j]
        self.triangles = p_map[self.triangles]

    def project(self, X: np.ndarray, x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray] = None) -> 'Mesh':
        """Project mesh points into new basis defined by (X, x, y, z)."""
        if z is None:
            z = np.cross(x, y)
        QQ = sum(self.points[:, i:i+1] * v for i, v in enumerate((x, y, z))) + X
        return Mesh(QQ, self.triangles.copy())


def shift(a, n=1):
    """Cyclically shift array `a` by `n` positions."""
    return a[(np.arange(len(a)) + n) % len(a)]


def mag(x, axis=None):
    """Compute the magnitude of vector(s) in `x` along a given axis."""
    x = np.asarray(x)
    if x.ndim == 1:
        return np.sqrt((x ** 2).sum())
    else:
        if axis is None:
            axis = x.ndim - 1
        m = np.sqrt((x ** 2).sum(axis))
        ns = list(x.shape)
        ns[axis] = 1
        return m.reshape(ns)


def m1(x):
    """Compute L2 norm across rows (axis 1)."""
    return np.sqrt((x ** 2).sum(axis=1))


def D(x):
    """Discrete central difference approximation."""
    x = np.array(x)
    return 0.5 * (shift(x, +1) - shift(x, -1))


def norm(x):
    """Normalize vectors in `x`."""
    return x / mag(x)


def proj(a, b):
    """Project vector `a` onto vector `b`."""
    b = norm(b)
    return a.dot(b) * b


if __name__ == '__main__':
    import sys
    from mesh import Mesh

    m = Mesh(sys.argv[1])
    x, y, z = m.points.T

    print(f'Points: {len(m.points)}')
    print(f'Triangles: {len(m.triangles)}')

    closed = m.is_closed(tol=1E-12)
    print(f'Volume: {m.volume():.6f} ({"closed and oriented" if closed else "NOT closed or properly oriented"})')
    print(f'Print price: ${m.volume() / 1000. * 0.30:.2f} (assuming units=mm and $0.30/cc)')
    print(f'X extents: ({x.min():.2f}, {x.max():.2f})')
    print(f'Y extents: ({y.min():.2f}, {y.max():.2f})')
    print(f'Z extents: ({z.min():.2f}, {z.max():.2f})')
    try:
        import pyvista as pv

        mesh = pv.PolyData(m.points, np.hstack([np.full((len(m.triangles), 1), 3), m.triangles]).astype(np.int32))
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, show_edges=True, color="lightgray")
        plotter.show()
    except ImportError:
        print("PyVista not installed.")
        try:
            from mayavi import mlab
            mlab.triangular_mesh(x, y, z, m.triangles)
            mlab.show()
        except ImportError:
            print("Mayavi not installed. Skipping visualization.")
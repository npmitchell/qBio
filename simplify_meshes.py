import os
import pyvista as pv

def simplify_mesh(file_path: str, target_reduction: float = 0.5) -> pv.PolyData:
    """
    Load a mesh from file and decimate it to reduce the number of vertices.

    Parameters
    ----------
    file_path : str
        Path to the input mesh file (.ply).
    target_reduction : float
        Fraction of total faces to remove (0.0 = no reduction, 0.9 = aggressive decimation).

    Returns
    -------
    pv.PolyData
        Simplified mesh as a PyVista PolyData object.
    """
    mesh = pv.read(file_path)
    simplified = mesh.decimate(target_reduction)
    return simplified


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Simplify all .ply meshes in a directory.')
    parser.add_argument('directory', type=str, help='Path to directory containing .ply files')
    parser.add_argument('--reduction', type=float, default=0.5,
                        help='Target fraction of faces to remove (e.g., 0.5 = keep 50%)')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix to add to output filenames')
    args = parser.parse_args()

    # Example usage
    # python3 simplify_meshes.py ./wt/20240527/ --reduction 0.5
    # python3 simplify_meshes.py ./wt/20240531/ --reduction 0.5
    # python3 simplify_meshes.py ./oe/20240528/ --reduction 0.5
    # python3 simplify_meshes.py ./oe/20240626/ --reduction 0.5

    ply_files = [f for f in os.listdir(args.directory) if f.endswith('.ply')]
    if not ply_files:
        print(f"No .ply files found in {args.directory}")
        exit(1)

    for fname in ply_files:
        in_path = os.path.join(args.directory, fname)
        out_path = os.path.join(
            args.directory,
            os.path.splitext(fname)[0] + args.suffix + '.ply'
        )

        print(f"Simplifying {fname} → {os.path.basename(out_path)}")
        simplified = simplify_mesh(in_path, target_reduction=args.reduction)
        simplified.save(out_path)

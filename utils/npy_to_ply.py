import argparse
import numpy as np
from pathlib import Path
import open3d as o3d

def get_knot_mesh(ply_path):
    mesh = o3d.io.read_triangle_mesh(ply_path)
    mesh.compute_vertex_normals()
    return mesh



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Initialization")
    parser.add_argument("--reference_path", dest="ref_path", default='./reference_nose_mesh.ply', help="Path to reference ply")
    parser.add_argument("--out_dir", dest="out_dir", default='./npy_to_ply_export', help="Export path")
    parser.add_argument("--pred", dest="pred_file", default="./predictions.npy", help="Predictions numpy" )
    args = parser.parse_args()
    
    Path(f"./{args.out_dir}").mkdir(parents=True, exist_ok=True)
    
    mesh = get_knot_mesh(args.ref_path)
    print(type(mesh.vertices))
    print(np.asarray(mesh.triangles))

    pred_tensor = np.load(args.pred_file)

    una_pred = pred_tensor[-1,:,:] 
    mesh.vertices = o3d.utility.Vector3dVector(una_pred)


    o3d.visualization.draw_geometries([mesh])
    
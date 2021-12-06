import argparse
import numpy as np
from pathlib import Path
import open3d as o3d

def get_knot_mesh(ply_path):
    mesh = o3d.io.read_triangle_mesh(ply_path)
    mesh.compute_vertex_normals()
    return mesh

def render_mesh(vertices, topology_reference_path):
    #Path(topology_reference_path).mkdir(parents=True, exist_ok=True)
    
    mesh = get_knot_mesh(topology_reference_path)
    print(type(mesh.vertices))
    print(np.asarray(mesh.triangles).shape)
    print("VERTICES SHAPE")
    print(vertices.shape) #well => (582, 3), unwell => (583, 3)
    print(f'First vertex: {vertices[0]}, last vertex: {vertices[-1]}')

    #pred_tensor = np.load(args.pred_file)

    #una_pred = pred_tensor[200,:,:] 
    mesh.vertices = o3d.utility.Vector3dVector(vertices) #well => (1094, 3), unwell => (1094, 3)
    o3d.visualization.draw_geometries([mesh])

def save_mesh_img(vertices, topology_reference_path, out_path):
        #Path(topology_reference_path).mkdir(parents=True, exist_ok=True)
    
    mesh = get_knot_mesh(topology_reference_path)
    '''print(type(mesh.vertices))
    print(np.asarray(mesh.triangles).shape)
    print("VERTICES SHAPE")
    print(vertices.shape) #well => (582, 3), unwell => (583, 3)
    print(f'First vertex: {vertices[0]}, last vertex: {vertices[-1]}')'''

    #pred_tensor = np.load(args.pred_file)

    #una_pred = pred_tensor[200,:,:] 
    mesh.vertices = o3d.utility.Vector3dVector(vertices) #well => (1094, 3), unwell => (1094, 3)
    #o3d.visualization.draw_geometries([mesh])
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False) #works for me with False, on some systems needs to be true
    vis.add_geometry(mesh)
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(out_path)
    vis.destroy_window()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Initialization")
    parser.add_argument("--reference_path", dest="ref_path", default='./reference_nose_mesh.ply', help="Path to reference ply")
    parser.add_argument("--out_dir", dest="out_dir", default='./npy_to_ply_export', help="Export path")
    parser.add_argument("--pred", dest="pred_file", default="./test.npy", help="Predictions numpy" )
    args = parser.parse_args()
    
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    pred_tensor = np.load(args.pred_file)
    print(pred_tensor.shape)
    una_pred = pred_tensor[200,:,:] 
    render_mesh(una_pred, args.ref_path)
    
    
    if False:
        mesh = get_knot_mesh(args.ref_path)
        print(type(mesh.vertices))
        print(np.asarray(mesh.triangles))

        pred_tensor = np.load(args.pred_file)

        una_pred = pred_tensor[200,:,:] 
        mesh.vertices = o3d.utility.Vector3dVector(una_pred)
        o3d.visualization.draw_geometries([mesh])
    
    
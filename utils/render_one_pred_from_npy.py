from os import listdir
from os.path import isfile, join

import open3d as o3d
import numpy as np

def get_numpy_from_file(input_file):
    pcd = o3d.io.read_point_cloud(input_file) # Read the point cloud

    #Visualize the point cloud within open3d
    #o3d.visualization.draw_geometries([pcd]) 

    # Convert open3d format to numpy array
    # Here, you have the point cloud in numpy format. 
    numpy_point_cloud = np.asarray(pcd.points) 
    #print(numpy_point_cloud.shape)
    return numpy_point_cloud

input_file = "./../true_prediction.npy"

pred_tensor = np.load(input_file)
print(f"Dimensione tensore {pred_tensor.shape}")

una_pred = pred_tensor[3,:,:]
print(una_pred.shape)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(una_pred)
o3d.visualization.draw_geometries([pcd]) 


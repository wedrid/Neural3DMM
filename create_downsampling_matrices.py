# creazione delle matrici di downsampling, qui è necessario utilizzare mpi-mesh

import numpy as np
import os
import pickle
import mesh_sampling
from shape_data import ShapeData
import argparse
import json

meshpackage = 'mpi-mesh'


def create_downsampling_matrices(args):

    data = dict_path['data']

    # mean.npy and std.npy
    shapedata = ShapeData(nVal=dict_path['nVal'],
                          train_file=data + '/train.npy',
                          test_file=data + '/test.npy',
                          reference_mesh_file=dict_path['reference_mesh_file'],
                          normalization=args.normalization,
                          meshpackage=meshpackage, load_flag=True)

    if not os.path.exists(data + '/mean.npy') or not os.path.exists(data + '/std.npy'):
        np.save(data + '/mean.npy', shapedata.mean)
        np.save(data + '/std.npy', shapedata.std)
    else:
        print("Files ", data + '/mean.npy' + " and " + data + '/std.npy'
              + "already generated --> Let's load them")

        shapedata.mean = np.load(dict_path['data'] + '/mean.npy')
        shapedata.std = np.load(dict_path['data'] + '/std.npy')
        shapedata.n_vertex = shapedata.mean.shape[0]
        shapedata.n_features = shapedata.mean.shape[1]
        print("Load OK ")

    # downsampling_matrices.pkl
    if not os.path.exists(os.path.join(dict_path['downsample_directory'], 'downsampling_matrices.pkl')):

        print("Generating Transform Matrices ..")
        if dict_path['downsample_method'] == 'COMA_downsample':
            M, A, D, U, F = mesh_sampling.generate_transform_matrices(shapedata.reference_mesh, args.ds_factors)

        with open(os.path.join(dict_path['downsample_directory'], 'downsampling_matrices.pkl'), 'wb') as fp:
            M_verts_faces = [(M[i].v, M[i].f) for i in range(len(M))]
            pickle.dump({'M_verts_faces': M_verts_faces, 'A': A, 'D': D, 'U': U, 'F': F}, fp)
    else:
        print("Transform Matrices already generated --> Let's load them")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="mpi-mesh execution")
    parser.add_argument("--ds_factors", dest="ds_factors", default=[4, 4, 4, 4], help="ds_factors")  # [4, 4, 4, 1] per mesh parte alta nasi
    parser.add_argument("--shuffle", dest="shuffle", default=True, help="Shuffle")
    parser.add_argument("--normalization", dest="normalization", default=True, help="Normalization")
    parser.add_argument("--dict", dest="dict_path", default=None, help="Path to the json file containing dict_path")

    args = parser.parse_args()

    with open(args.dict_path) as json_file:
        dict_path = json.load(json_file)

    # mpi-mesh
    create_downsampling_matrices(args)

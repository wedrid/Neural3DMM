import numpy as np
import os
import pickle
import trimesh
from shape_data import ShapeData
from sklearn.metrics.pairwise import euclidean_distances

from spiral_utils import get_adj_trigs, generate_spirals


def matrices_tr(args, reference_points, dict_path):
    meshpackage = 'trimesh'
    # np.random.seed(args['seed'])
    print("Loading data .. ")

    shapedata = ShapeData(nVal=dict_path['nVal'],
                          train_file=dict_path['data'] + '/train.npy',
                          test_file=dict_path['data'] + '/test.npy',
                          reference_mesh_file=dict_path['reference_mesh_file'],
                          normalization=args.normalization,
                          meshpackage=meshpackage, load_flag=False)

    shapedata.mean = np.load(dict_path['data'] + '/mean.npy')
    shapedata.std = np.load(dict_path['data'] + '/std.npy')
    # shapedata.mean = np.load(dict_path['data'] + '/mean.npy', allow_pickle=True)
    # shapedata.std = np.load(dict_path['data'] + '/std.npy', allow_pickle=True)
    shapedata.n_vertex = shapedata.mean.shape[0]
    shapedata.n_features = shapedata.mean.shape[1]

    print("Loading Transform Matrices ..")
    with open(os.path.join(dict_path['downsample_directory'], 'downsampling_matrices.pkl'), 'rb') as fp:
        # downsampling_matrices = pickle.load(fp,encoding = 'latin1')
        downsampling_matrices = pickle.load(fp)

    M_verts_faces = downsampling_matrices['M_verts_faces']
    M = [trimesh.base.Trimesh(vertices=M_verts_faces[i][0], faces=M_verts_faces[i][1], process=False) for i in
         range(len(M_verts_faces))]
    A = downsampling_matrices['A']
    D = downsampling_matrices['D']
    U = downsampling_matrices['U']
    F = downsampling_matrices['F']

    # Needs also an extra check to enforce points to belong to different disconnected component at each hierarchy level
    print("Calculating reference points for downsampled versions..")
    for i in range(len(args.ds_factors)):
        dist = euclidean_distances(M[i + 1].vertices, M[0].vertices[reference_points[0]])
        reference_points.append(np.argmin(dist, axis=0).tolist())

    sizes = [x.vertices.shape[0] for x in M]

    Adj, Trigs = get_adj_trigs(A, F, shapedata.reference_mesh, meshpackage=shapedata.meshpackage)

    spirals_np, spiral_sizes, spirals = generate_spirals(args.step_sizes,
                                                         M, Adj, Trigs,
                                                         reference_points=reference_points,
                                                         dilation=args.dilation, random=False,
                                                         meshpackage=shapedata.meshpackage,
                                                         counter_clockwise=True)

    bU = []
    bD = []
    for i in range(len(D)):
        d = np.zeros((1, D[i].shape[0] + 1, D[i].shape[1] + 1))
        u = np.zeros((1, U[i].shape[0] + 1, U[i].shape[1] + 1))
        d[0, :-1, :-1] = D[i].todense()
        u[0, :-1, :-1] = U[i].todense()
        d[0, -1, -1] = 1
        u[0, -1, -1] = 1
        bD.append(d)
        bU.append(u)

    return shapedata, sizes, bU, bD, spirals_np, spiral_sizes, spirals



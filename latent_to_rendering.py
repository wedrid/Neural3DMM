#python3 latent_to_rendering.py --dict ./TMP/dict_path.json --checkpoint_file checkpoint290
# script can be executed either with or without arguments, in particular init has optional arguments to change dict path and checkpoint.

from random import sample
import numpy as np
import os
from numpy.lib.npyio import save
import torch
from models import SpiralAutoencoder
import argparse
import json
from transform_matrices import matrices_tr
from tqdm import tqdm
import copy
from torchsummary import summary
from utils import render_mesh
import open3d as o3d
from pathlib import Path

#python main.py --dict /home/egrappolini/CG3D/prova1/dict_path.json --mode 'test' --checkpoint_file /home/egrappolini/CG3D/prova1/results/spirals_\ autoencoder/checkpoints/checkpoint490
#python3 latent_to_pointcloud.py --dict ./TMP/dict_path.json --checkpoint_file ./TMP/checkpoints/checkpoint290.pth.tar
#python3 latent_to_pointcloud.py --dict ./TMP/dict_path.json --checkpoint_file ./TMP/checkpoints/checkpoint290


from torch.utils.data import Dataset, DataLoader
from autoencoder_dataset import *

ds_factors = [4, 4, 4, 4]
step_sizes = [2, 2, 1, 1, 1]
filter_size_enc = [[3, 16, 32, 64, 128], [[], [], [], [], []]]
filter_size_dec = [[128, 64, 32, 32, 16], [[], [], [], [], 3]]

dilation_flag = True
if dilation_flag:
    dilation = [2, 2, 1, 1, 1]
else:
    dilation = None
#FIXME reference_point!! Forse dovrebbe essere il vertice del capo del naso ? proviamo con 450
#reference_points = [[414]]  # 414  [[3567,4051,4597]] used for COMA with 3 disconnected components
reference_points = [[414]]

def save_predictions(pred_tensor_mod, out_path): 
    print("size")
    print(pred_tensor_mod.size())
    #gets rid of dummy node for correct rendering (with shading)
    final_preds = np.zeros((pred_tensor_mod.shape[0], pred_tensor_mod.shape[1] + 1, pred_tensor_mod.shape[2]), dtype=np.float32)
    final_preds[:, :-1, :] = pred_tensor_mod.cpu().detach().numpy()
    #one_pred = final_preds[1] 
    #print(one_pred)
    
    if False: # turn true for point cloud visualization in a window
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(one_pred)
        o3d.visualization.draw_geometries([pcd])
    print(final_preds.shape)
    #render_mesh.render_mesh(one_pred[:-1,:], "./utils/reference_nose_mesh.ply")
    for (i,one_pred) in enumerate(final_preds):
        Path(out_path).mkdir(parents=True, exist_ok=True)
        render_mesh.save_mesh_img(one_pred[:-1,:], "./utils/reference_nose_mesh.ply", out_path + f"rendering_{i}.png")
    
    
    print("Saved")
    return

def render_pred(pred_tensor_mod, ref_topology): 
    print("size")
    print(pred_tensor_mod.size())
    #gets rid of dummy node for correct rendering (with shading)
    final_preds = np.zeros((pred_tensor_mod.shape[0], pred_tensor_mod.shape[1] + 1, pred_tensor_mod.shape[2]), dtype=np.float32)
    final_preds[:, :-1, :] = pred_tensor_mod.cpu().detach().numpy()
    one_pred = final_preds[0] 
    #print(one_pred)
    
    print(final_preds.shape)
    render_mesh.render_mesh(one_pred[:-1,:], ref_topology)
    
    return

def decode_and_render_latent(latent, model, shapedata_mean, shapedata_std, ref_topo):
    if latent.ndim == 1: 
        latent = np.expand_dims(latent, axis=0)
    mm_constant = 1000
    latent = np.expand_dims(latent, axis=0)
    pred = model.decode(torch.tensor(latent))
    pred = pred[:, :-1] #gets rid of dummy node
    pred_tensor_mod = (pred * shapedata_std + shapedata_mean) * mm_constant
    render_pred(pred_tensor_mod, ref_topo) 

def decode_and_save_latent(latent, model, shapedata_mean, shapedata_std, out_path):
    mm_constant = 1000
    if latent.ndim == 1: 
        latent = np.expand_dims(latent, axis=0)
    pred = model.decode(torch.tensor(latent))
    pred = pred[:, :-1] #gets rid of dummy node
    pred_tensor_mod = (pred * shapedata_std + shapedata_mean) * mm_constant
    save_predictions(pred_tensor_mod, out_path) 
    

def decode_latent_segment(lat1, lat2, step_size, model, shapedata_mean, shapedata_std, additional_path = ""):
    assert lat1.ndim == 1
    assert lat2.ndim == 1
    mm_constant = 1000
    print("Latents 1 and 2")
    #.... hackerman
    if np.linalg.norm(lat1) < np.linalg.norm(lat2):
        latent1 = copy.deepcopy(lat1)
        latent2 = copy.deepcopy(lat2)
    else:
        latent1 = copy.deepcopy(lat2)
        latent2 = copy.deepcopy(lat1)
        
    versor = (latent2 - latent1) / (np.linalg.norm(latent2 - latent1)) # versor goes from latent 1 to latent 2
    additive_vector = versor * step_size

    sample_point = latent1
    print("ENTERING ITERATIONS")
    latents = []
    while np.linalg.norm(sample_point) < np.linalg.norm(latent2):
        print("Iteration")
        latents.append(copy.deepcopy(sample_point))
        sample_point += additive_vector
        #print(sample_point)
    latents.append(copy.deepcopy(latent2))
   
    
    latents = np.array(latents)
    latents = torch.tensor(latents).float()
    print(f"Segment: {latents}")

    if False:
        latents = (np.random.rand(10,16) + 20)
        latents = torch.tensor(latents).float()

    pred = model.decode(latents)
    print("DECODED")
    print(pred.shape)
    pred = pred[:, :-1] #gets rid of dummy node
    pred_tensor_mod = (pred * shapedata_std + shapedata_mean) * mm_constant
    save_predictions(pred_tensor_mod, "./renderings/"+additional_path) 

def init(checkpoint = False, dict_path = False):
    parser = argparse.ArgumentParser(description="neural 3DMM ...")
    parser.add_argument("--GPU", dest="GPU", default=True, help="GPU is available")
    parser.add_argument("--device", dest="device_idx", default='0', help="choose GPU")

    parser.add_argument("--seed", dest="seed", default=2, help="Seed ...")
    parser.add_argument("--loss", dest="loss", default='l1', help="Loss type")
    parser.add_argument("--batch_size", dest="batch_size", default=16, help="Batch size")
    parser.add_argument("--epochs", dest="epochs", default=300, help="Number of epochs")
    parser.add_argument("--eval_f", dest="eval_frequency", default=200, help="Eval frequency")
    parser.add_argument("--num_w", dest="num_workers", default=4, help="Number of workers")
    # pause
    # parser.add_argument("--filter_s_enc", dest="filter_sizes_enc", default=f_sizes_enc_list,
    #                     help="Sizes of filter encoder", type=list)
    # parser.add_argument("--filter_s_dec", dest="filter_sizes_dec", default=f_sizes_dec_list,
    #                     help="Sizes of filter decoder", type=list)
    parser.add_argument("--nz", dest="nz", default=16, help="Nz")
    parser.add_argument("--ds_factors", dest="ds_factors", default=ds_factors, help="ds_factors")
    parser.add_argument("--step_sizes", dest="step_sizes", default=step_sizes, help="step_sizes")
    parser.add_argument("--dilation", dest="dilation", default=dilation, help="dilation")
    parser.add_argument("--lr", dest="lr", default=1e-3, help="lr")
    parser.add_argument("--regularization", dest="regularization", default=5e-5, help="Regularization")
    parser.add_argument("--scheduler", dest="scheduler", default=True, help="Scheduler")
    parser.add_argument("--decay_rate", dest="decay_rate", default=0.99, help="decay_rate")
    parser.add_argument("--decay_steps", dest="decay_steps", default=1, help="decay_steps")
    parser.add_argument("--resume", dest="resume", default=False, help="decay_steps")
    parser.add_argument("--mode", dest="mode", default='train', help="Mode")
    parser.add_argument("--shuffle", dest="shuffle", default=True, help="Shuffle")
    parser.add_argument("--normalization", dest="normalization", default=True, help="Normalization")
    parser.add_argument("--checkpoint_file", dest="checkpoint_file", default=checkpoint, help="checkpoint_file")
    #if not specified, takes in the argument of the function
    parser.add_argument("--dict", dest="dict_path", default=dict_path, help="Path to the json file containing dict_path")


    args = parser.parse_args()
    print(args.GPU)

    with open(args.dict_path) as json_file:
        dict_path = json.load(json_file)

    GPU = args.GPU
    data = dict_path['data']
    if os.path.exists(data + '/mean.npy') or not os.path.exists(data + '/std.npy'):
        np.random.seed(args.seed)
        shapedata, sizes, bU, bD, spirals_np, spiral_sizes, spirals = matrices_tr(args, reference_points, dict_path)
        # Model
        torch.manual_seed(args.seed)

        if GPU:
            # device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        print(device)

        tspirals = [torch.from_numpy(s).long().to(device) for s in spirals_np]
        tD = [torch.from_numpy(s).float().to(device) for s in bD]
        tU = [torch.from_numpy(s).float().to(device) for s in bU]

        if 'autoencoder' in dict_path['generative_model']:
            model = SpiralAutoencoder(filters_enc=filter_size_enc,
                                      filters_dec=filter_size_dec,
                                      latent_size=args.nz,
                                      sizes=sizes,
                                      spiral_sizes=spiral_sizes,
                                      spirals=tspirals,
                                      D=tD, U=tU, device=device).to(device)

        checkpoint_dict = torch.load(os.path.join(dict_path['checkpoint_path'], args.checkpoint_file + '.pth.tar'),
                                         map_location=device)
        model.load_state_dict(checkpoint_dict['autoencoder_state_dict'])
        print("")

        model.eval()
        mm_constant=1000

        shapedata_mean = torch.Tensor(shapedata.mean).to(device)
        shapedata_std = torch.Tensor(shapedata.std).to(device)

        return (model, shapedata_mean, shapedata_std)

        #tx = one_data['points'].to(device)
    else:
        return False

def main():
    # FOR COMA :#model, shapedata_mean, shapedata_std = init(dict_path="./TMP/dict_path.json", checkpoint="checkpoint290")
    print("LATENT: ")
    #FOR COMA TOP: model, shapedata_mean, shapedata_std = init(dict_path="./trained_models/filtri_coma_TOP/dict_path.json", checkpoint="checkpoint290")
    # FOR COMA latents: # latents = np.load("mylatents.npy")
    # FOR COMA TOP: # latents = np.load("./latents/latents_COMA_top/test_COMA_top_latents.npy")
    
    model, shapedata_mean, shapedata_std = init(dict_path="./trained_models/new_dataset/dict_path.json", checkpoint="checkpoint290")
    latents = np.load("./latents/latents_complete_new_dataset/test_newdataset_latents.npy")

    print(latents.shape)
    norms = []
    for item in latents:
        norms.append(np.linalg.norm(item))
    norms = np.array(norms)
    print(f"Min: {norms[np.argmin(norms)]}, Max: {norms[np.argmax(norms)]}")
    #latents = torch.tensor(latents[4000:4010,:])
    #decode_latent_segment(latents[np.argmax(norms)], latents[np.argmin(norms)], 3, model, shapedata_mean, shapedata_std)
    #decode_latent_segment(latents[0], latents[500], 3, model, shapedata_mean, shapedata_std)
    
    #decode_and_render_latent(torch.zeros(16)+10, model, shapedata_mean, shapedata_std)
    #decode_and_render_latent(latents[120], model, shapedata_mean, shapedata_std, "./utils/reference_nose_mesh.ply")
    # FOR COMA TOP: decode_and_render_latent(latents[10], model, shapedata_mean, shapedata_std, "./rendering_references/COMA_top_reference.ply")
    decode_and_render_latent(latents[10], model, shapedata_mean, shapedata_std, "./rendering_references/new_dataset_reference.ply")
    return 
    latents = torch.tensor(latents[1000:1005,:])
    print(f"SHAPEH LATENTS: {latents.size()}")

    if False:
        latents = (np.random.rand(10,16) + 20)
        latents = torch.tensor(latents).float()
        
    pred = model.decode(latents)
    print("DECODED")
    print(pred.shape)
    pred = pred[:, :-1]
    pred_tensor_mod = (pred * shapedata_std + shapedata_mean) * mm_constant
    save_predictions(pred_tensor_mod, "./renderings/") 
    return
 

if __name__ == '__main__':
    lc = main()
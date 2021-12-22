import numpy as np
import os
import torch
from models import SpiralAutoencoder
import argparse
import json
from transform_matrices import matrices_tr
from tqdm import tqdm
import copy
from torchsummary import summary

from torch.utils.data import Dataset, DataLoader
from autoencoder_dataset import *

#python extract_latent.py --dict /home/egrappolini/CG3D/filtri_coma/dict_path.json --checkpoint_file checkpoint290
#python3 extract_latent.py --dict ./trained_models/filtri_coma_TOP/dict_path.json --checkpoint_file checkpoint290

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
reference_points = [[414]]  # 414  [[3567,4051,4597]] used for COMA with 3 disconnected components
#162 per più piccolo

def main():
    torch.cuda.empty_cache()
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
    parser.add_argument("--checkpoint_file", dest="checkpoint_file", default='checkpoint', help="checkpoint_file")
    parser.add_argument("--dict", dest="dict_path", default=None, help="Path to the json file containing dict_path")


    args = parser.parse_args()

    train_test = "train"

    with open(args.dict_path) as json_file:
        dict_path = json.load(json_file)

    GPU = False
    data = dict_path['data']
    if os.path.exists(data + '/mean.npy') or not os.path.exists(data + '/std.npy'):
        np.random.seed(args.seed)
        shapedata, sizes, bU, bD, spirals_np, spiral_sizes, spirals = matrices_tr(args, reference_points, dict_path)
        # Model
        torch.manual_seed(args.seed)
        #GPU = False
        
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
        #summary(model)
        # print(model)
        # for name, layer in model.named_modules():
        #     # if isinstance(layer, torch.nn.Conv2d):
        #     print(name, layer)

        # model = model.fc_latent_enc
        # print("boh: ", boh)s
        #summary(model)
        

        # python model_extraction.py --dict
        # C:/Users/chiar/PycharmProjects/Neural3DMM_noses/TMP/dict_path.json --checkpoint_file
        # C:/Users/chiar/PycharmProjects/Neural3DMM_noses/TMP/checkpoints/checkpoint290
        dataset_test = autoencoder_dataset(root_dir=data, points_dataset="test", #miao
                                           shapedata=shapedata,
                                           normalization=args.normalization)

        #dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,
        #                             shuffle=False, num_workers=args.num_workers)
        dataloader_test = DataLoader(dataset_test, batch_size=int(8),
                                     shuffle=False, num_workers=args.num_workers)

        '''subset_indices = [0] # select your indices here as a list
        subset = torch.utils.data.Subset(dataloader_test, subset_indices)
        testloader_subset = torch.utils.data.DataLoader(subset, batch_size=1, num_workers=0, shuffle=False)
        print(testloader_subset)'''
        all_data = dataset_test.getWholeProcessedDataset("./trained_models/new_dataset/preprocessed/test.npy")

        #all_data = dataset_test.getWholeProcessedDataset(f"./dataset_npy/{train_test}.npy")
        #one_data = next(iter(dataloader_test))   
        print("")
        #print(one_data)

        model.eval()
        l1_loss = 0
        l2_loss = 0
        mm_constant=1000
        shapedata_mean = torch.Tensor(shapedata.mean).to(device)
        shapedata_std = torch.Tensor(shapedata.std).to(device)

        #tx = one_data['points'].to(device)
        tx = all_data[:,:,:].to(device)
        print("Inside encode")
        #print(tx)
        print(tx.size())
        latent_code = model.encode(tx)
        print("LATENT: ")
        #print(latent_code)
        #print(latent_code.shape)
        
        #pred = model.decode(latent_code)
        #print("DECODED")
        #print(pred)
        #print(pred.shape)

        one_latent_code = latent_code[0]
        numpy_latents = latent_code.cpu().detach().numpy()
        print(numpy_latents[0])
        #saves latents
        with open("./latents/latents_COMA_top/test_newdataset_latents.npy", 'wb') as file:
            np.save(file, numpy_latents)
        
        '''
        with torch.no_grad():
            for i, sample_dict in enumerate(tqdm(dataloader_test)):
                tx = sample_dict['points'].to(device)
                prediction = model(tx)
                if i == 0:
                    predictions = copy.deepcopy(prediction)
                    testset = copy.deepcopy(tx[:, :-1]) #fixed rendering
                else:
                    predictions = torch.cat([predictions, prediction], 0)
                    testset = torch.cat([testset, tx[:, :-1]], 0) #fixed rendering

                if dataloader_test.dataset.dummy_node:
                    x_recon = prediction[:, :-1]
                    x = tx[:, :-1]
                else:
                    x_recon = prediction
                    x = tx
                l1_loss += torch.mean(torch.abs(x_recon - x)) * x.shape[0] / float(len(dataloader_test.dataset))

                x_recon = (x_recon * shapedata_std + shapedata_mean) * mm_constant
                x = (x * shapedata_std + shapedata_mean) * mm_constant
                l2_loss += torch.mean(torch.sqrt(torch.sum((x_recon - x) ** 2, dim=2))) * x.shape[0] / float(
                    len(dataloader_test.dataset))

            testset = (testset * shapedata_std + shapedata_mean) * mm_constant #fixed rendering
            predictions = predictions.cpu().numpy()
            l1_loss = l1_loss.item()
            l2_loss = l2_loss.item()
        '''
        return 

if __name__ == '__main__':
    lc = main()
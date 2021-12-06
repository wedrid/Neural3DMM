import os
from train import train
from test import test
import torch
from torch.utils.data import DataLoader
from models import SpiralAutoencoder
from autoencoder_dataset import autoencoder_dataset
from transform_matrices import matrices_tr
import argparse
import numpy as np
import json
#from test_funcs import test_autoencoder_dataloader


ds_factors = [4, 4, 4, 4]
step_sizes = [2, 2, 1, 1, 1]
filter_size_enc = [[3, 16, 32, 64, 128], [[], [], [], [], []]]
filter_size_dec = [[128, 64, 32, 32, 16], [[], [], [], [], 3]]

# f_sizes_enc_list = list([3, 16, 32, 64, 128])
# f_sizes_dec_list = list([128, 64, 32, 32, 16])

# without 3

#
# filter_sizes_enc = [[64, 64, 64, 128], [[],[],[],[]]]
# filter_sizes_dec = [[128, 64, 64, 64], [[],[],[],[]]]

# DFAUST
# f_sizes_enc_list = list([16, 32, 64, 128])
# f_sizes_dec_list = list([128, 64, 32, 32, 16])

# COMA
# f_sizes_enc_list = list([64, 64, 64, 128])
# f_sizes_dec_list = list([128, 64, 64, 64])

dilation_flag = True
if dilation_flag:
    dilation = [2, 2, 1, 1, 1]
else:
    dilation = None
#FIXME reference_point!! Forse dovrebbe essere il vertice del capo del naso ? proviamo con 450
reference_points = [[414]]  # 414  [[3567,4051,4597]] used for COMA with 3 disconnected components

# python main.py --dict /home/egrappolini/CG3D/prova1/dict_path.json
# --checkpoint_file /home/egrappolini/CG3D/prova1/results/spirals_\ autoencoder/checkpoints/checkpoint490

def load_model(path_del_checkpoint):
    parser = argparse.ArgumentParser(description="neural 3DMM ...")
    parser.add_argument("--GPU", dest="GPU", default=True, help="GPU is available")
    parser.add_argument("--device", dest="device_idx", default='0', help="choose GPU")
    parser.add_argument("--seed", dest="seed", default=2, help="Seed ...")
    parser.add_argument("--num_w", dest="num_workers", default=4, help="Number of workers")
    parser.add_argument("--nz", dest="nz", default=16, help="Nz")
    parser.add_argument("--ds_factors", dest="ds_factors", default=ds_factors, help="ds_factors")
    parser.add_argument("--dilation", dest="dilation", default=dilation, help="dilation")
    parser.add_argument("--scheduler", dest="scheduler", default=True, help="Scheduler")
    parser.add_argument("--resume", dest="resume", default=False, help="decay_steps")
    parser.add_argument("--mode", dest="mode", default='test', help="Mode")
    parser.add_argument("--shuffle", dest="shuffle", default=True, help="Shuffle")
    parser.add_argument("--checkpoint_file", dest="checkpoint_file", default='checkpoint', help="checkpoint_file")
    parser.add_argument("--dict", dest="dict_path", default=None, help="Path to the json file containing dict_path")

    args = parser.parse_args()

    with open(args.dict_path) as json_file:
        dict_path = json.load(json_file)

    GPU = args.GPU
    # device_idx = args.device_idx
    # torch.cuda.get_device_name(device_idx)

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

        # Building model, optimizer, and loss function
        '''dataset_train = autoencoder_dataset(root_dir=data, points_dataset='train',
                                            shapedata=shapedata,
                                            normalization=args.normalization)

        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                      shuffle=args.shuffle, num_workers=args.num_workers)
        '''

        dataset_val = autoencoder_dataset(root_dir=data, points_dataset='val',
                                          shapedata=shapedata,
                                          normalization=args.normalization)

        dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers)

        dataset_test = autoencoder_dataset(root_dir=data, points_dataset='test',
                                           shapedata=shapedata,
                                           normalization=args.normalization)

        dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers)

        if 'autoencoder' in dict_path['generative_model']:
            model = SpiralAutoencoder(filters_enc=filter_size_enc,
                                      filters_dec=filter_size_dec,
                                      latent_size=args.nz,
                                      sizes=sizes,
                                      spiral_sizes=spiral_sizes,
                                      spirals=tspirals,
                                      D=tD, U=tU, device=device).to(device)

        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.regularization)
        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optim, args.decay_steps, gamma=args.decay_rate)
        else:
            scheduler = None

        if args.loss == 'l1':
            def loss_l1(outputs, targets):
                L = torch.abs(outputs - targets).mean()
                return L

            loss_fn = loss_l1

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total number of parameters is: {}".format(params))
        print(model)
        # print(M[4].v.shape)

        #### from here on we use the model
        print('loading checkpoint from file %s' % (os.path.join(dict_path['checkpoint_path'], args.checkpoint_file + '.pth.tar')))
        checkpoint_dict = torch.load(os.path.join(dict_path['checkpoint_path'], args.checkpoint_file + '.pth.tar'), map_location=device)
        model.load_state_dict(checkpoint_dict['autoencoder_state_dict'])
        
        #Failed trial of loading the model without dict, il problema è che servono le varie matrici ed i vari parametri come mean, etc. 
        #print("Loading from " + path_del_checkpoint)
        #checkpoint_dict = torch.load(path_del_checkpoint, map_location=device)
        #model.load_state_dict(checkpoint_dict['autoencoder_state_dict'])

        # In previous line, the model should be complitely loaded
        return model
        '''
        predictions, testset, norm_l1_loss, l2_loss = test_autoencoder_dataloader(device, model, dataloader_test,
                                                                        shapedata, mm_constant=1000) #fixed rendering
        np.save(os.path.join(dict_path['prediction_path'], 'predictions'), predictions)
        np.save(os.path.join(dict_path['prediction_path'], 'testset'), testset) #fixed rendering

        print('autoencoder: normalized loss', norm_l1_loss)
        print('autoencoder: euclidean distance in mm=', l2_loss)'''

    else:
        raise NotImplementedError('Run script create_downsampling_matrices.py first to generate file')


if __name__ == '__main__':
    main()


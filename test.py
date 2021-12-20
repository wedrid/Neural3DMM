import numpy as np
import os
from test_funcs import test_autoencoder_dataloader
import torch


def test(model, device, args, dict_path, dataloader_test, shapedata):
    print('loading checkpoint from file %s' % (os.path.join(dict_path['checkpoint_path'], args.checkpoint_file + '.pth.tar')))
    checkpoint_dict = torch.load(os.path.join(dict_path['checkpoint_path'], args.checkpoint_file + '.pth.tar'),
                                 map_location=device)
    model.load_state_dict(checkpoint_dict['autoencoder_state_dict'])
    #true prediction è la versione 'normalizzata'
    predictions, true_prediction, norm_l1_loss, l2_loss = test_autoencoder_dataloader(device, model, dataloader_test,
                                                                     shapedata, mm_constant=1000) #fixed rendering
    np.save(os.path.join(dict_path['prediction_path'], 'predictions'), predictions)
    np.save(os.path.join(dict_path['prediction_path'], 'true_prediction'), true_prediction) #fixed rendering

    print('autoencoder: normalized loss', norm_l1_loss)
    print('autoencoder: euclidean distance in mm=', l2_loss)
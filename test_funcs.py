import torch
import copy
from tqdm import tqdm
import numpy as np

# testset ...


def test_autoencoder_dataloader(device, model, dataloader_test, shapedata, mm_constant=1000):
    model.eval()
    l1_loss = 0
    l2_loss = 0
    shapedata_mean = torch.Tensor(shapedata.mean).to(device)
    shapedata_std = torch.Tensor(shapedata.std).to(device)
    with torch.no_grad():
        for i, sample_dict in enumerate(tqdm(dataloader_test)):
            tx = sample_dict['points'].to(device)
            prediction = model(tx)
            if i == 0:
                predictions = copy.deepcopy(prediction)
                #testset = copy.deepcopy(tx[:, :-1]) #fixed rendering
            else:
                predictions = torch.cat([predictions, prediction], 0)
                #testset = torch.cat([testset, tx[:, :-1]], 0) #fixed rendering

            if dataloader_test.dataset.dummy_node:
                x_recon = prediction[:, :-1]
                x = tx[:, :-1]
            else:
                x_recon = prediction
                x = tx
            l1_loss += torch.mean(torch.abs(x_recon - x)) * x.shape[0] / float(len(dataloader_test.dataset))

            x_recon = (x_recon * shapedata_std + shapedata_mean) * mm_constant
            if i == 0:
                true_predictions = copy.deepcopy(x_recon) #fixed rendering
            else:
                true_predictions = torch.cat([true_predictions, x_recon], 0) #fixed rendering
            x = (x * shapedata_std + shapedata_mean) * mm_constant
            l2_loss += torch.mean(torch.sqrt(torch.sum((x_recon - x) ** 2, dim=2))) * x.shape[0] / float(
                len(dataloader_test.dataset))

        #testset = (testset * shapedata_std + shapedata_mean) * mm_constant #fixed rendering
        predictions = predictions.cpu().numpy()
        l1_loss = l1_loss.item()
        l2_loss = l2_loss.item()

    return predictions, true_predictions.cpu().numpy(), l1_loss, l2_loss #fixed rendering

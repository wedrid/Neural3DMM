import json
import os
import copy
from train_funcs import train_autoencoder_dataloader
import torch
from tensorboardX import SummaryWriter


def train(model, device, dataloader_train, dataloader_val, optim, loss_fn, args, dict_path,
          scheduler, shapedata):
    writer = SummaryWriter(dict_path['summary_path'])
    with open(os.path.join(dict_path['checkpoints'], args.name + '_params.json'), 'w') as fp:
        saveparams = copy.deepcopy(args)  # TODO args non è un dizionario
        json.dump(saveparams, fp)

    if args.resume:
        print('loading checkpoint from file %s' % (os.path.join(dict_path['checkpoint_path'], args.checkpoint_file)))
        checkpoint_dict = torch.load(os.path.join(dict_path['checkpoint_path'], args.checkpoint_file + '.pth.tar'),
                                     map_location=device)
        start_epoch = checkpoint_dict['epoch'] + 1
        model.load_state_dict(checkpoint_dict['autoencoder_state_dict'])
        optim.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
        print('Resuming from epoch %s' % (str(start_epoch)))
    else:
        start_epoch = 0

    if args.generative_model == 'autoencoder':
        train_autoencoder_dataloader(dataloader_train, dataloader_val,
                                     device, model, optim, loss_fn,
                                     bsize=args.batch_size,  # args['batch_size']
                                     start_epoch=start_epoch,
                                     n_epochs=args.num_epochs,  # args['num_epochs']
                                     eval_freq=args.eval_frequency,  # args['eval_frequency']
                                     scheduler=scheduler,
                                     writer=writer,
                                     save_recons=True,
                                     shapedata=shapedata,
                                     metadata_dir=dict_path['checkpoint_path'], samples_dir=dict_path['samples_path'],
                                     checkpoint_path=args.checkpoint_file)
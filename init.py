from tqdm import tqdm
import numpy as np
import os


def create_directory_hierarchy(args, dict_path):
    if os.path.exists(os.path.join(args.root_dir)):
        print('Created folders:')

        if not os.path.exists(os.path.join(args.root_dir, 'preprocessed')):
            os.makedirs(os.path.join(args.root_dir, 'preprocessed'))
            print('\t\tProcessed')

        if not os.path.exists(os.path.join(dict_path['downsample_directory'])):
            os.makedirs(dict_path['downsample_directory'])
            print(f'\t\tTemplate and downsample_directory')

        ###
        if not os.path.exists(os.path.join(dict_path['points_train_path'])):
            os.makedirs(dict_path['points_train_path'])
            print(f'\t\tPoints_train')

        if not os.path.exists(dict_path['points_val_path']):
            os.makedirs(dict_path['points_val_path'])
            print('\t\tPoints_val')

        if not os.path.exists(dict_path['points_test_path']):
            os.makedirs(dict_path['points_test_path'])
            print('\t\tPoints_test')

        ###
        if not os.path.exists(dict_path['summary_path']):
            os.makedirs(dict_path['summary_path'])
            print('\t\tSummary')

        if not os.path.exists(dict_path['checkpoint_path']):
            os.makedirs(dict_path['checkpoint_path'])
            print('\t\tCheckpoint')

        if not os.path.exists(dict_path['samples_path']):
            os.makedirs(dict_path['samples_path'])
            print('\t\tSamples')

        if not os.path.exists(dict_path['prediction_path']):
            os.makedirs(dict_path['prediction_path'])
            print('\t\tPrediction')

    else:
        raise NotImplementedError('Create a root_dataset folder')


def data_generation_func(args, dict_path, data):
    train = np.load(data + '/train.npy')

    for i in tqdm(range(len(train) - args.nVal)):
        np.save(os.path.join(dict_path['points_train_path'], '{0}.npy'.format(i)), train[i])
    for i in range(len(train) - args.nVal, len(train)):
        np.save(os.path.join(dict_path['points_val_path'], '{0}.npy'.format(i)), train[i])

    test = np.load(data + '/test.npy')
    for i in range(len(test)):
        np.save(os.path.join(dict_path['points_test_path'], '{0}.npy'.format(i)), test[i])

    files = []
    for r, d, f in os.walk(dict_path['points_train_path']):
        for file in f:
            if '.npy' in file:
                files.append(os.path.splitext(file)[0])
    np.save(os.path.join(data, 'paths_train.npy'), files)

    files = []
    for r, d, f in os.walk(dict_path['points_val_path']):
        for file in f:
            if '.npy' in file:
                files.append(os.path.splitext(file)[0])
    np.save(os.path.join(data, 'paths_val.npy'), files)

    files = []
    for r, d, f in os.walk(dict_path['points_test_path']):
        for file in f:
            if '.npy' in file:
                files.append(os.path.splitext(file)[0])
    np.save(os.path.join(data, 'paths_test.npy'), files)


def init_func(args, dict_path, data):
    create_directory_hierarchy(args, dict_path)
    data_generation_func(args, dict_path, data)

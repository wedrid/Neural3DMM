from tqdm import tqdm
import numpy as np
import os
import argparse
import json
import platform


def write_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def right_slash(path):
    if type(path) is str and '\\' in path:
        path = path.replace('\\', '/')
        print("Changing '\\' slashes")
        # print("path traformato: ", path)

    return path


# def create_directory_hierarchy(args, dict_path):
def create_directory_hierarchy(dict_path):
    if os.path.exists(os.path.join(dict_path['root_dir'])):  # args.root_dir
        print('Created folders:')

        if not os.path.exists(os.path.join(dict_path['root_dir'], 'preprocessed')):  # args.root_dir
            os.makedirs(os.path.join(dict_path['root_dir'], 'preprocessed'))  # args.root_dir
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


# def data_generation_func(args, dict_path, data):
def data_generation_func(dict_path):
    data = dict_path['data']
    train = np.load(data + '/train.npy')

    for i in tqdm(range(len(train) - dict_path['nVal'])):  # args.nVal
        np.save(os.path.join(dict_path['points_train_path'], '{0}.npy'.format(i)), train[i])
    for i in range(len(train) - dict_path['nVal'], len(train)):  # args.nVal
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


# def init_func(args, dict_path, data):
#     create_directory_hierarchy(args, dict_path)
#     data_generation_func(args, dict_path, data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Initialization")
    parser.add_argument("--root_dir", dest="root_dir", default=None, help="Root data directory location")
    parser.add_argument("--name", dest="name", default='', help="Name")
    parser.add_argument("--downsample_m", dest="downsample_method", default='COMA_downsample',
                        help="Name of downsample method")
    parser.add_argument("--generative_m", dest="generative_model", default='autoencoder', help="autoencoder... type ?")
    parser.add_argument("--nVal", dest="nVal", default=100, help="nVal")
    parser.add_argument("--bool", dest="bool", default=None, help="0 to run create_directory_hierarchy, "
                                                                 "1 to run data_generation_func")

    args = parser.parse_args()

    results_folder = os.path.join(args.root_dir, 'results/spirals_ ' + args.generative_model)
    data = os.path.join(args.root_dir, 'preprocessed', args.name)

    dict_path = {
        'reference_mesh_file': os.path.join(args.root_dir, 'template', 'template.obj'),
        'downsample_directory': os.path.join(args.root_dir, 'template', args.downsample_method),
        'summary_path': os.path.join(results_folder, 'summaries', args.name),
        'checkpoint_path': os.path.join(results_folder, 'checkpoints', args.name),
        'samples_path': os.path.join(results_folder, 'samples', args.name),
        'prediction_path': os.path.join(results_folder, 'predictions', args.name),
        'points_train_path': os.path.join(data, 'points_train'),
        'points_val_path': os.path.join(data, 'points_val'),
        'points_test_path': os.path.join(data, 'points_test'),
        # Added
        'root_dir': args.root_dir, 'name': args.name, 'downsample_method': args.downsample_method,
        'generative_model': args.generative_model, 'nVal': args.nVal,
        'data': data, 'results_folder': results_folder
    }

    if args.bool == '0':
        if platform.system() == 'Windows':
            dic = {}
            for i in dict_path.keys():
                print('key: ', i)
                p = right_slash(dict_path[i])
                dict_path[i] = p
        write_json(dict_path, os.path.join(args.root_dir, 'dict_path.json'))  # salvo dict_path su un file json
        create_directory_hierarchy(dict_path)
    else:
        data_generation_func(dict_path)

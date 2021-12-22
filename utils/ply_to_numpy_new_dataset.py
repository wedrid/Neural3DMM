from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd

from ply_to_numpy import get_numpy_from_file
import argparse
import os
import platform


# per path di windows creati con join che hanno slash \
def right_slash(path):
    if platform.system() == 'Windows' and '\\' in path:
        path = path.replace('\\', '/')
        # print("Changing '\\' slashes")
        # print("path traformato: ", path)

    return path


def get_heading(filename):
    if filename.startswith('b'):
        # print("bosphorus_noLm_SLC dataset heading")
        spl = filename.split('.')[0]  # nome senza estensione ply
        f = spl.split('_')[0]  # estraggo la parte che considero come intestazione
    else:
        # print("FRGC_noLm_SLC dataset heading")
        f = filename[:filename.index('d') + 1]  # estraggo la parte fino a 'd' compresa

    return f


def create_npy_csv_files(directory, dest_directory, train_perc, dest_filename):
    # due sottocartelle : bosphorus_noLm_SLC, FRGC_noLm_SLC
    subdir_list = [f for f in sorted(listdir(directory)) if not isfile(join(directory, f))]

    # primo elemento di una delle due sottocartelle
    sample_file = directory + "/" + subdir_list[1] + "/bs000_CAU_A22A25_0.ply"
    print("sample file: ", sample_file)

    complete_train = np.zeros(get_numpy_from_file(sample_file).shape)
    complete_train = np.expand_dims(complete_train, axis=0)

    complete_test = np.zeros(get_numpy_from_file(sample_file).shape)
    complete_test = np.expand_dims(complete_test, axis=0)

    train = []  # nomi mesh
    test = []
    tot = 0
    for i in range(len(subdir_list)):
        print("Cartella: ", subdir_list[i])
        temp_subdir = directory + "/" + subdir_list[i]
        files = [f for f in sorted(listdir(temp_subdir)) if isfile(right_slash(join(temp_subdir, f)))]

        files_head = []
        for i in range(len(files)):
            f = files[i]
            f = get_heading(f)
            files_head.append(f)

        unici = np.unique(files_head)

        size = len(unici)  # numero di 'entità'
        indices = np.arange(size)

        train_size = int(round((train_perc * size) / 100, 0))
        test_size = size - train_size

        print("\tTRAIN SIZE: ", train_size)
        print("\tTEST SIZE: ", test_size)

        # SPLIT
        # np.random.shuffle(indices)
        # per ora fissiamo lo split (le prime 326 entità stanno nel train, le restanti nel test
        train_i, test_i = indices[:train_size], indices[train_size:(train_size + test_size)]  # inidici di train e test

        # mi prendo le mesh corrispondenti agli indici che ho creato
        # controllando le intestazioni

        for k in range(10):  # range(len(files)):  # ciclo su files
            f = get_heading(files[k])

            name_file = temp_subdir + "/" + files[k]
            print("NAME: ", name_file)
            temp = get_numpy_from_file(name_file)
            temp = np.expand_dims(temp, axis=0)

            # mi devo prendere l'indice di unici e vedere dove si trova se in train o test
            ind = np.where(unici == f)[0][0]  # where restituisce una tupla

            if ind in train_i:
                # train
                complete_train = np.concatenate((complete_train, temp), axis=0)  # npy
                train.append(files[k])  # nome mesh
            else:
                # test
                complete_test = np.concatenate((complete_test, temp), axis=0)
                test.append(files[k])

            tot += 1

    print(f"Finish: shape of train is {complete_train.shape} final shape is ")
    complete_train = complete_train[1:, :, :]
    print(complete_train.shape)

    print(f"Finish: shape of test is {complete_test.shape} final shape is ")
    complete_test = complete_test[1:, :, :]
    print(complete_test.shape)

    print(f"Total number of meshes considered is: {tot}")

    myMetadata_train = pd.DataFrame(
        {
            'mesh_file_name': train
        }
    )

    myMetadata_test = pd.DataFrame(
        {
            'mesh_file_name': test
        }
    )

    name_train = "train_" + dest_filename + ".npy"
    name_test = "test_" + dest_filename + ".npy"
    csv_train = "train_" + dest_filename + "_metadata.csv"
    csv_test = "test_" + dest_filename + "_metadata.csv"

    if platform.system() != 'Windows':
        name_train = dest_directory + name_train
        name_test = dest_directory + name_test
        csv_train = dest_directory + csv_train
        csv_test = dest_directory + csv_test

    with open(name_train, 'wb') as file:
        np.save(file, complete_train)

    with open(name_test, 'wb') as file:
        np.save(file, complete_test)

    myMetadata_train.to_csv(csv_train)
    myMetadata_test.to_csv(csv_test)


if __name__ == '__main__':
    # vogliamo: train.npy (number_meshes, number_vertices, 3) (no Faces because they all share topology)
    # ------ args:

    parser = argparse.ArgumentParser(description="Split dataset")

    parser.add_argument("--directory", dest="directory", default=None,
                        help="Path to the folder that contains the NEW dataset")
    parser.add_argument("--dest_directory", dest="dest_directory", default='',
                        help="Path to the folder where we want to save the csv and npy files")
    parser.add_argument("--train_p", dest="train_p", default='',
                        help="% of training samples, expressed as integer")
    parser.add_argument("--dest_f", dest="dest_f", default='',
                        help="Destination filename (for npy e csv to combine)")

    args = parser.parse_args()

    train_p = int(args.train_p)
    dest_dir = args.dest_directory
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    create_npy_csv_files(directory=args.directory, dest_directory=dest_dir, train_perc=train_p,
                         dest_filename=args.dest_f)
    # no slash for directory,
    # yes slash for dest_directory

    # Es: python ./utils/ply_to_numpy_new_dataset.py --directory C:/Users/chiar/Desktop/FRGC_Bosph_registeredMeshes_TPAMI_noses
    # --dest_directory C:/Users/PycharmProjects/Neural3DMM_noses/npy_nuovo_dataset/ --train_p 70 --dest_f pino





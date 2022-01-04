"""" Nel caso in cui si abbiano già i csv (del nuovo dataset) con mesh_file_name ed expr_info,
si può runnare questo script per aggiungere la colonna 'gender'. """

from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import argparse
import os


def add_genders_column(directory, dest_directory, dest_filename, csv_train, csv_test):
    subdir_list = [f for f in sorted(listdir(directory)) if not isfile(join(directory, f))]

    g_train = []  # genders di train
    g_test = []  # genders di test
    for i in range(len(subdir_list)):
        temp_subdir = directory + "/" + subdir_list[i]

        print("Cartella: ", subdir_list[i])

        files = [f.rstrip('.ply') for f in sorted(listdir(temp_subdir)) if isfile(right_slash(join(temp_subdir, f)))]

        [u, g] = get_genders(subdir_list[i][0], files)  # la funzione distingue tra F e b

        # devo splittare tra train e test
        train_perc = 70
        size = len(u)  # numero di 'entità'
        indices = np.arange(size)

        train_size = int(round((train_perc * size) / 100, 0))
        test_size = size - train_size

        print("\tTRAIN SIZE: ", train_size)
        print("\tTEST SIZE: ", test_size)

        train_i, test_i = indices[:train_size], indices[train_size:(train_size + test_size)]  # inidici di train e test

        for k in range(len(files)):  # ciclo su files
            print("K: ", k)
            f = get_heading(files[k])

            name_file = temp_subdir + "/" + files[k]
            print("NAME: ", name_file)

            # mi devo prendere l'indice di unici e vedere dove si trova se in train o test
            ind = np.where(u == f)[0][0]  # where restituisce una tupla
            print("IND: ", ind)

            if ind in train_i:
                # train
                g_train.append(g[k])  # genere
            else:
                # test
                g_test.append(g[k])

    # Train csv
    csv_file = csv_train
    data = pd.read_csv(csv_file, names=["mesh_file_name", "expr_info"])

    names = list(data["mesh_file_name"])
    expr_info = list(data["expr_info"])

    # Update csv train

    csv_new_train = dest_directory + "train_" + dest_filename + "_metadata.csv"
    myMetadata_train = pd.DataFrame(
        {
            'mesh_file_name': names[1:],
            'expr_info': expr_info[1:],
            'gender': g_train
        }
    )
    myMetadata_train.to_csv(csv_new_train)

    # Test csv
    csv_file = csv_test
    data = pd.read_csv(csv_file, names=["mesh_file_name", "expr_info"])

    names_test = list(data["mesh_file_name"])
    expr_info_test = list(data["expr_info"])

    # Update csv test

    csv_new_test = dest_directory + "test_" + dest_filename + "_metadata.csv"
    myMetadata_test = pd.DataFrame(
        {
            'mesh_file_name': names_test[1:],
            'expr_info': expr_info_test[1:],
            'gender': g_test
        }
    )
    myMetadata_test.to_csv(csv_new_test)


if __name__ == '__main__':
    from ply_to_numpy import right_slash
    from add_info import get_heading, get_genders

    parser = argparse.ArgumentParser(description="Add genders info to train/test csv of the new dataset")

    parser.add_argument("--directory", dest="directory", default=None,
                        help="Path to the folder that contains the NEW dataset")
    parser.add_argument("--dest_directory", dest="dest_directory", default='',
                        help="Path to the folder where we want to save the csv files.")
    parser.add_argument("--dest_f", dest="dest_f", default='',
                        help="Destination filename (between train/test and metadata)")

    parser.add_argument("--csv_train", dest="csv_train", default='',
                        help="Path to the train csv file (FRGCV + bosphorus). We want to add it a column "
                             "with gender info. ")
    parser.add_argument("--csv_test", dest="csv_test", default='',
                        help="Path to the test csv file (FRGCV + bosphorus). We want to add it a column "
                             "with gender info. ")


    args = parser.parse_args()

    dest_dir = args.dest_directory
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    add_genders_column(directory=right_slash(args.directory), dest_directory=right_slash(dest_dir),
                       dest_filename=args.dest_f, csv_train=right_slash(args.csv_train),
                       csv_test=right_slash(args.csv_test))

    # no slash for directory
    # yes slash for dest_directory

    # Es: python expand_csv.py --directory D:\Progetto_CG3D\FRGC_Bosph_registeredMeshes_TPAMI_noses
    # --dest_directory C:\Users\chiar\PycharmProjects\Neural3DMM_noses\npy_nuovo_dataset\top_cut_with_gt\UPDATE\
    # --dest_f nasi
    # --csv_train C:\Users\chiar\PycharmProjects\Neural3DMM_noses\npy_nuovo_dataset\with_gt\train_nasi_metadata.csv
    # --csv_test C:\Users\chiar\PycharmProjects\Neural3DMM_noses\npy_nuovo_dataset\with_gt\test_nasi_metadata.csv

    # per parte alta: o rieseguo cambiando directory etc oppure semplicemnte copio i file già generati
    # per i nasi completi (la gerarchia è la stessa e dentro ci sono solo i nomi delle mesh che sono uguali)
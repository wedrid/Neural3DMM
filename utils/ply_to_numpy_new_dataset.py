from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd

from ply_to_numpy import get_numpy_from_file, right_slash
from add_info import get_neutrals
import argparse
import os
import platform


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

    ## GT
    complete_train_gt = np.zeros(get_numpy_from_file(sample_file).shape)
    complete_train_gt = np.expand_dims(complete_train_gt, axis=0)

    complete_test_gt = np.zeros(get_numpy_from_file(sample_file).shape)
    complete_test_gt = np.expand_dims(complete_test_gt, axis=0)

    ##

    train = []  # nomi mesh
    test = []
    train_gt = []  # nomi mesh gt
    test_gt = []

    expr_train = []
    expr_test = []

    tot = 0
    for i in range(len(subdir_list)):

        print("Cartella: ", subdir_list[i])
        temp_subdir = directory + "/" + subdir_list[i]
        files = [f for f in sorted(listdir(temp_subdir)) if isfile(right_slash(join(temp_subdir, f)))]

        files_head = []
        for p in range(len(files)):
            f = files[p]
            f = get_heading(f)
            files_head.append(f)

        [unici, counts] = np.unique(files_head, return_counts=True)

        size = len(unici)  # numero di 'entità'
        indices = np.arange(size)

        train_size = int(round((train_perc * size) / 100, 0))
        test_size = size - train_size

        print("\tTRAIN SIZE: ", train_size)
        print("\tTEST SIZE: ", test_size)

        # SPLIT
        # np.random.shuffle(indices)
        # per ora fissiamo lo split (le prime tot entità stanno nel train, le restanti nel test
        train_i, test_i = indices[:train_size], indices[train_size:(train_size + test_size)]  # inidici di train e test

        # mi prendo le mesh corrispondenti agli indici che ho creato
        # controllando le intestazioni

        # indici = filtered_list_replicata # [1, 1, 1, 21, 21, 21]  # stessa dimensione di files
        [indici, comments_list] = get_neutrals(dataset=subdir_list[0][i], files=files, unici=unici, counts=counts)

        # inizializzo il rappresentante
        rappr = temp_subdir + "/" + files[indici[0]]
        head_rappr = get_heading(files[indici[0]])
        temp_gt = get_numpy_from_file(rappr)
        temp_gt = np.expand_dims(temp_gt, axis=0)
        ind_gt = np.where(unici == head_rappr)[0][0]

        for k in range(len(files)):  # ciclo su files
            print("K, indici[k]: ", k, indici[k])
            f = get_heading(files[k])

            name_file = temp_subdir + "/" + files[k]
            print("NAME: ", name_file)
            temp = get_numpy_from_file(name_file)
            temp = np.expand_dims(temp, axis=0)

            # mi devo prendere l'indice di unici e vedere dove si trova se in train o test
            ind = np.where(unici == f)[0][0]  # where restituisce una tupla
            print("IND: ", ind)

            ## GT
            if f != head_rappr:
                print("Update rappresentante")
                rappr = temp_subdir + "/" + files[indici[k]]
                head_rappr = get_heading(files[indici[k]])
                temp_gt = get_numpy_from_file(rappr)
                temp_gt = np.expand_dims(temp_gt, axis=0)
                ind_gt = np.where(unici == head_rappr)[0][0]
                print("IND GT: ", ind_gt)
            ###

            if ind in train_i:
                # train
                complete_train = np.concatenate((complete_train, temp), axis=0)  # npy
                train.append(files[k])  # nome mesh
                expr_train.append(comments_list[k])  # espressione mesh
            else:
                # test
                complete_test = np.concatenate((complete_test, temp), axis=0)
                test.append(files[k])
                expr_test.append(comments_list[k])

            ## GT
            if ind_gt in train_i:
                # train
                complete_train_gt = np.concatenate((complete_train_gt, temp_gt), axis=0)  # npy
                train_gt.append(files[indici[k]])  # nome mesh
            else:
                # test
                complete_test_gt = np.concatenate((complete_test_gt, temp_gt), axis=0)
                test_gt.append(files[indici[k]])
            ##

            tot += 1

    print(f"Finish: shape of train is {complete_train.shape} final shape is ")
    complete_train = complete_train[1:, :, :]
    print(complete_train.shape)

    print(f"Finish: shape of test is {complete_test.shape} final shape is ")
    complete_test = complete_test[1:, :, :]
    print(complete_test.shape)

    ### GT
    print(f"Finish: shape of train GT is {complete_train_gt.shape} final shape is ")
    complete_train_gt = complete_train_gt[1:, :, :]
    print(complete_train_gt.shape)

    print(f"Finish: shape of test GT is {complete_test_gt.shape} final shape is ")
    complete_test_gt = complete_test_gt[1:, :, :]
    print(complete_test_gt.shape)
    ###

    print(f"Total number of meshes considered is: {tot}")

    myMetadata_train = pd.DataFrame(
        {
            'mesh_file_name': train,
            'expr_info': expr_train

        }
    )

    myMetadata_test = pd.DataFrame(
        {
            'mesh_file_name': test,
            'expr_info': expr_test
        }
    )

    name_train = dest_directory + "train_" + dest_filename + ".npy"
    name_test = dest_directory + "test_" + dest_filename + ".npy"
    csv_train = dest_directory + "train_" + dest_filename + "_metadata.csv"
    csv_test = dest_directory + "test_" + dest_filename + "_metadata.csv"

    with open(name_train, 'wb') as file:
        np.save(file, complete_train)

    with open(name_test, 'wb') as file:
        np.save(file, complete_test)

    myMetadata_train.to_csv(csv_train)
    myMetadata_test.to_csv(csv_test)

    ## GT

    myMetadata_train_gt = pd.DataFrame(
        {
            'mesh_file_name': train_gt
        }
    )

    myMetadata_test_gt = pd.DataFrame(
        {
            'mesh_file_name': test_gt
        }
    )

    name_train_gt = dest_directory + "train_" + dest_filename + "_gt.npy"
    name_test_gt = dest_directory + "test_" + dest_filename + "_gt.npy"

    csv_train_gt = dest_directory + "train_" + dest_filename + "_metadata_gt.csv"
    csv_test_gt = dest_directory + "test_" + dest_filename + "_metadata_gt.csv"

    with open(name_train_gt, 'wb') as file:
        np.save(file, complete_train_gt)

    with open(name_test_gt, 'wb') as file:
        np.save(file, complete_test_gt)

    myMetadata_train_gt.to_csv(csv_train_gt)
    myMetadata_test_gt.to_csv(csv_test_gt)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Split dataset")

    parser.add_argument("--directory", dest="directory", default=None,
                        help="Path to the folder that contains the NEW dataset")
    parser.add_argument("--dest_directory", dest="dest_directory", default='',
                        help="Path to the folder where we want to save the csv and npy files")
    parser.add_argument("--train_p", dest="train_p", default=70,
                        help="% of training samples, expressed as integer")
    parser.add_argument("--dest_f", dest="dest_f", default='',
                        help="Destination filename (for npy e csv to combine)")

    args = parser.parse_args()

    train_p = int(args.train_p)
    dest_dir = args.dest_directory
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    create_npy_csv_files(directory=right_slash(args.directory), dest_directory=right_slash(dest_dir),
                         train_perc=train_p, dest_filename=args.dest_f)
    # no slash for directory,
    # yes slash for dest_directory

    # Es: python ./utils/ply_to_numpy_new_dataset.py --directory D:/Progetto_CG3D/FRGC_Bosph_registeredMeshes_TPAMI_noses
    # --dest_directory C:/Users/PycharmProjects/Neural3DMM_noses/npy_nuovo_dataset/ --train_p 70 --dest_f pino


    # per esecuzione da console ...
    # parser.add_argument("--directory", dest="directory", default='D:/Progetto_CG3D/FRGC_Bosph_registeredMeshes_TPAMI_noses',
    #                     help="Path to the folder that contains the NEW dataset")
    # parser.add_argument("--dest_directory", dest="dest_directory",
    #                     default='C:/Users/PycharmProjects/Neural3DMM_noses/macche/',
    #                     help="Path to the folder where we want to save thecsv and npy files")
    # parser.add_argument("--train_p", dest="train_p", default=70, help="% of training samples, expressed as integer")
    # parser.add_argument("--dest_f", dest="dest_f", default='pallas', help="Destination filename (for npy e csv to combine)")
    #
    # args = parser.parse_args()
    #
    # train_p = int(args.train_p)
    # dest_dir = args.dest_directory
    #
    # dest_filename = args.dest_f
    # directory = right_slash(args.directory)
    # dest_directory = right_slash(dest_dir)
    # train_perc = train_p

    # se eseguo da console (e quindi da dentro la cartella utils)
    # file_txt = "./info_files/Geometrix_Exp3_ExpressionCategories.txt"
    # neutral_txt = "./info_files/list_frgcv2_neutral.txt"
    # in add_info.py

    # copio il contenuto della funzione

    # # due sottocartelle : bosphorus_noLm_SLC, FRGC_noLm_SLC
    # subdir_list = [f for f in sorted(listdir(directory)) if not isfile(join(directory, f))]
    #
    # # primo elemento di una delle due sottocartelle
    # sample_file = directory + "/" + subdir_list[1] + "/bs000_CAU_A22A25_0.ply"
    # print("sample file: ", sample_file)
    #
    # complete_train = np.zeros(get_numpy_from_file(sample_file).shape)
    # complete_train = np.expand_dims(complete_train, axis=0)
    #
    # complete_test = np.zeros(get_numpy_from_file(sample_file).shape)
    # complete_test = np.expand_dims(complete_test, axis=0)
    #
    # ## GT
    # complete_train_gt = np.zeros(get_numpy_from_file(sample_file).shape)
    # complete_train_gt = np.expand_dims(complete_train_gt, axis=0)
    #
    # complete_test_gt = np.zeros(get_numpy_from_file(sample_file).shape)
    # complete_test_gt = np.expand_dims(complete_test_gt, axis=0)
    #
    # ##
    #
    # train = []  # nomi mesh
    # test = []
    # train_gt = []  # nomi mesh gt
    # test_gt = []
    #
    # expr_train = []
    # expr_test = []
    #
    # tot = 0
    # # for i in range(len(subdir_list)):
    # i = 1
    #
    # print("Cartella: ", subdir_list[i])
    # temp_subdir = directory + "/" + subdir_list[i]
    # files = [f for f in sorted(listdir(temp_subdir)) if isfile(right_slash(join(temp_subdir, f)))]
    #
    # files_head = []
    # for p in range(len(files)):
    #     f = files[p]
    #     f = get_heading(f)
    #     files_head.append(f)
    #
    # [unici, counts] = np.unique(files_head, return_counts=True)
    #
    # size = len(unici)  # numero di 'entità'
    # indices = np.arange(size)
    #
    # train_size = int(round((train_perc * size) / 100, 0))
    # test_size = size - train_size
    #
    # print("\tTRAIN SIZE: ", train_size)
    # print("\tTEST SIZE: ", test_size)
    #
    # # SPLIT
    # # np.random.shuffle(indices)
    # # per ora fissiamo lo split (le prime tot entità stanno nel train, le restanti nel test
    # train_i, test_i = indices[:train_size], indices[train_size:(train_size + test_size)]  # inidici di train e test
    #
    # # mi prendo le mesh corrispondenti agli indici che ho creato
    # # controllando le intestazioni
    #
    # # indici = filtered_list_replicata # [1, 1, 1, 21, 21, 21]  # stessa dimensione di files
    # [indici, comments_list] = get_neutrals(dataset=subdir_list[0][i], files=files, unici=unici, counts=counts)
    #
    # # inizializzo il rappresentante
    # rappr = temp_subdir + "/" + files[indici[0]]
    # head_rappr = get_heading(files[indici[0]])
    # temp_gt = get_numpy_from_file(rappr)
    # temp_gt = np.expand_dims(temp_gt, axis=0)
    # ind_gt = np.where(unici == head_rappr)[0][0]
    #
    # for k in range(len(files)):  # ciclo su files
    #     print("K, indici[k]: ", k, indici[k])
    #     f = get_heading(files[k])
    #
    #     name_file = temp_subdir + "/" + files[k]
    #     print("NAME: ", name_file)
    #     temp = get_numpy_from_file(name_file)
    #     temp = np.expand_dims(temp, axis=0)
    #
    #     # mi devo prendere l'indice di unici e vedere dove si trova se in train o test
    #     ind = np.where(unici == f)[0][0]  # where restituisce una tupla
    #     print("IND: ", ind)
    #
    #     ## GT
    #     if f != head_rappr:
    #         print("Update rappresentante")
    #         rappr = temp_subdir + "/" + files[indici[k]]
    #         head_rappr = get_heading(files[indici[k]])
    #         temp_gt = get_numpy_from_file(rappr)
    #         temp_gt = np.expand_dims(temp_gt, axis=0)
    #         ind_gt = np.where(unici == head_rappr)[0][0]
    #         print("IND GT: ", ind_gt)
    #     ###
    #
    #     if ind in train_i:
    #         # train
    #         complete_train = np.concatenate((complete_train, temp), axis=0)  # npy
    #         train.append(files[k])  # nome mesh
    #         expr_train.append(comments_list[k])  # espressione mesh
    #     else:
    #         # test
    #         complete_test = np.concatenate((complete_test, temp), axis=0)
    #         test.append(files[k])
    #         expr_test.append(comments_list[k])
    #
    #     ## GT
    #     if ind_gt in train_i:
    #         # train
    #         complete_train_gt = np.concatenate((complete_train_gt, temp_gt), axis=0)  # npy
    #         train_gt.append(files[indici[k]])  # nome mesh
    #     else:
    #         # test
    #         complete_test_gt = np.concatenate((complete_test_gt, temp_gt), axis=0)
    #         test.append(files[indici[k]])
    #     ##
    #
    #     tot += 1
    #
    # print(f"Finish: shape of train is {complete_train.shape} final shape is ")
    # complete_train = complete_train[1:, :, :]
    # print(complete_train.shape)
    #
    # print(f"Finish: shape of test is {complete_test.shape} final shape is ")
    # complete_test = complete_test[1:, :, :]
    # print(complete_test.shape)
    #
    # ### GT
    # print(f"Finish: shape of train GT is {complete_train_gt.shape} final shape is ")
    # complete_train_gt = complete_train_gt[1:, :, :]
    # print(complete_train_gt.shape)
    #
    # print(f"Finish: shape of test GT is {complete_test_gt.shape} final shape is ")
    # complete_test_gt = complete_test_gt[1:, :, :]
    # print(complete_test_gt.shape)
    # ###

























# def create_npy_csv_files(directory, dest_directory, train_perc, dest_filename):
#     # due sottocartelle : bosphorus_noLm_SLC, FRGC_noLm_SLC
#     subdir_list = [f for f in sorted(listdir(directory)) if not isfile(join(directory, f))]
#
#     # primo elemento di una delle due sottocartelle
#     sample_file = directory + "/" + subdir_list[1] + "/bs000_CAU_A22A25_0.ply"
#     print("sample file: ", sample_file)
#
#     complete_train = np.zeros(get_numpy_from_file(sample_file).shape)
#     complete_train = np.expand_dims(complete_train, axis=0)
#
#     complete_test = np.zeros(get_numpy_from_file(sample_file).shape)
#     complete_test = np.expand_dims(complete_test, axis=0)
#
#     train = []  # nomi mesh
#     test = []
#     tot = 0
#     for i in range(len(subdir_list)):
#         print("Cartella: ", subdir_list[i])
#         temp_subdir = directory + "/" + subdir_list[i]
#         files = [f for f in sorted(listdir(temp_subdir)) if isfile(right_slash(join(temp_subdir, f)))]
#
#         files_head = []
#         for i in range(len(files)):
#             f = files[i]
#             f = get_heading(f)
#             files_head.append(f)
#
#         unici = np.unique(files_head)
#
#         size = len(unici)  # numero di 'entità'
#         indices = np.arange(size)
#
#         train_size = int(round((train_perc * size) / 100, 0))
#         test_size = size - train_size
#
#         print("\tTRAIN SIZE: ", train_size)
#         print("\tTEST SIZE: ", test_size)
#
#         # SPLIT
#         # np.random.shuffle(indices)
#         # per ora fissiamo lo split (le prime 326 entità stanno nel train, le restanti nel test
#         train_i, test_i = indices[:train_size], indices[train_size:(train_size + test_size)]  # inidici di train e test
#
#         # mi prendo le mesh corrispondenti agli indici che ho creato
#         # controllando le intestazioni
#         for k in range(len(files)):  # ciclo su files
#             f = get_heading(files[k])
#
#             name_file = temp_subdir + "/" + files[k]
#             print("NAME: ", name_file)
#             temp = get_numpy_from_file(name_file)
#             temp = np.expand_dims(temp, axis=0)
#
#             # mi devo prendere l'indice di unici e vedere dove si trova se in train o test
#             ind = np.where(unici == f)[0][0]  # where restituisce una tupla
#
#             if ind in train_i:
#                 # train
#                 complete_train = np.concatenate((complete_train, temp), axis=0)  # npy
#                 train.append(files[k])  # nome mesh
#             else:
#                 # test
#                 complete_test = np.concatenate((complete_test, temp), axis=0)
#                 test.append(files[k])
#
#             tot += 1
#
#     print(f"Finish: shape of train is {complete_train.shape} final shape is ")
#     complete_train = complete_train[1:, :, :]
#     print(complete_train.shape)
#
#     print(f"Finish: shape of test is {complete_test.shape} final shape is ")
#     complete_test = complete_test[1:, :, :]
#     print(complete_test.shape)
#
#     print(f"Total number of meshes considered is: {tot}")
#
#     myMetadata_train = pd.DataFrame(
#         {
#             'mesh_file_name': train
#         }
#     )
#
#     myMetadata_test = pd.DataFrame(
#         {
#             'mesh_file_name': test
#         }
#     )
#
#     name_train = dest_directory + "train_" + dest_filename + ".npy"
#     name_test = dest_directory + "test_" + dest_filename + ".npy"
#     csv_train = dest_directory + "train_" + dest_filename + "_metadata.csv"
#     csv_test = dest_directory + "test_" + dest_filename + "_metadata.csv"
#
#     with open(name_train, 'wb') as file:
#         np.save(file, complete_train)
#
#     with open(name_test, 'wb') as file:
#         np.save(file, complete_test)
#
#     myMetadata_train.to_csv(csv_train)
#     myMetadata_test.to_csv(csv_test)

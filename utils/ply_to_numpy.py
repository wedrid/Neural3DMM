############################
"""
Funziona solo con il dataset COMA ad ora, e' sufficiente cambiare il valore della variabile test_set, che contiene
gli indici dei volti che vogliamo come test.
"""

from os import listdir
from os.path import isfile, join

import open3d as o3d
import numpy as np

from tqdm import tqdm
import pandas as pd
import argparse
import os
import platform


def get_numpy_from_file(input_file):
    pcd = o3d.io.read_point_cloud(input_file)  # Read the point cloud

    # Visualize the point cloud within open3d
    # o3d.visualization.draw_geometries([pcd])

    # Convert open3d format to numpy array
    # Here, you have the point cloud in numpy format. 
    numpy_point_cloud = np.asarray(pcd.points)
    # print(numpy_point_cloud.shape)
    return numpy_point_cloud


# per path di windows creati con join che hanno slash \
def right_slash(path):
    if platform.system() == 'Windows' and '\\' in path:
        path = path.replace('\\', '/')
        # print("Changing '\\' slashes")
        # print("path traformato: ", path)

    return path


def create_npy_csv_files(directory, dest_directory, dest_filename):
    # Lo split è fissato (cartelle 0, 1, 2 test, le restanti train)
    test_set = {0, 1, 2}

    # dato che la gerarchia è la stessa per tutti i dataset di stampo 'coma', questo nome va sempre bene
    # nel senso che rappresenterà sempre una mesh di riferimento generica (naso intero o parte alta che sia)
    sample_file = directory + "/" + "FaceTalk_170725_00137_TA/bareteeth/bareteeth.000001.ply"

    complete_train = np.zeros(get_numpy_from_file(sample_file).shape)
    complete_train = np.expand_dims(complete_train, axis=0)

    complete_test = np.zeros(get_numpy_from_file(sample_file).shape)
    complete_test = np.expand_dims(complete_test, axis=0)

    subdir_list = [f for f in sorted(listdir(directory)) if not isfile(join(directory, f))]  # soggetti

    temp_subdir = directory + "/" + subdir_list[0]
    subsubdir_list = [f for f in sorted(listdir(temp_subdir)) if not isfile(join(temp_subdir, f))]  # espressioni
    expr_dict = {}
    i = 0
    for item in subsubdir_list:
        expr_dict[item] = i
        i += 1

    i = 0
    tot = 0

    individuals_test = []
    expressions_test = []
    indiv_name_test = []
    expr_name_test = []
    mesh_file_name_test = []  # nomi completi delle mesh di test (es: bareteeth.000001.ply)

    individuals_train = []
    expressions_train = []
    indiv_name_train = []
    expr_name_train = []
    mesh_file_name_train = []  # nomi completi delle mesh di train

    print("SHAPE")
    print(complete_train.shape)

    for subdir in tqdm(subdir_list):
        # print(f"Iteration {i} of {len(subdir_list)}")
        # print(subdir)

        temp_subdir = directory + "/" + subdir
        subsubdir_list = [f for f in sorted(listdir(temp_subdir)) if not isfile(join(temp_subdir, f))]
        for subsubdir in tqdm(subsubdir_list, leave=False):
            # print(subsubdir)
            temp_subsubdir = temp_subdir + "/" + subsubdir
            files_list = [f for f in sorted(listdir(temp_subsubdir)) if isfile(join(temp_subsubdir, f))]
            # print(files_list)
            for item in tqdm(files_list, leave=False):
                input_file = temp_subsubdir + "/" + item
                # print(input_file)
                temp = get_numpy_from_file(input_file)
                temp = np.expand_dims(temp, axis=0)
                if i in test_set:
                    complete_test = np.concatenate((complete_test, temp), axis=0)
                    individuals_test.append(i)
                    expressions_test.append(expr_dict[subsubdir])
                    indiv_name_test.append(subdir)
                    expr_name_test.append(subsubdir)
                    mesh_file_name_test.append(item)
                else:
                    complete_train = np.concatenate((complete_train, temp), axis=0)
                    individuals_train.append(i)
                    expressions_train.append(expr_dict[subsubdir])
                    indiv_name_train.append(subdir)
                    expr_name_train.append(subsubdir)
                    mesh_file_name_train.append(item)
                tot += 1

        i += 1

    print(f"Finish: shape of train is {complete_train.shape} final shape is ")
    complete_train = complete_train[1:, :, :]
    print(complete_train.shape)

    print(f"Finish: shape of test is {complete_test.shape} final shape is ")
    complete_test = complete_test[1:, :, :]
    print(complete_test.shape)

    print(f"Total number of meshes considered is: {tot}")
    myMetadata_test = pd.DataFrame(
        {
            'individual_id': individuals_test,
            'expression_id': expressions_test,
            'individual_name': indiv_name_test,
            'expression_name': expr_name_test,
            'mesh_file_name': mesh_file_name_test
        }
    )

    myMetadata_train = pd.DataFrame(
        {
            'individual_id': individuals_train,
            'expression_id': expressions_train,
            'individual_name': indiv_name_train,
            'expression_name': expr_name_train,
            'mesh_file_name': mesh_file_name_train
        }
    )

    # Save npy files
    name_train = dest_directory + "train_" + dest_filename + ".npy"
    name_test = dest_directory + "test_" + dest_filename + ".npy"
    csv_train = dest_directory + "train_" + dest_filename + "_metadata.csv"
    csv_test = dest_directory + "test_" + dest_filename + "_metadata.csv"

    with open(name_train, 'wb') as file:
        np.save(file, complete_train)

    with open(name_test, 'wb') as file:
        np.save(file, complete_test)

    # Save csv files
    myMetadata_train.to_csv(csv_train)
    myMetadata_test.to_csv(csv_test)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Split dataset COMA")

    parser.add_argument("--directory", dest="directory", default=None,
                        help="Path to the folder that contains the NEW dataset")
    parser.add_argument("--dest_directory", dest="dest_directory", default='',
                        help="Path to the folder where we want to save the csv and npy files")
    parser.add_argument("--dest_f", dest="dest_f", default='',
                        help="Destination filename (for npy e csv to combine)")

    args = parser.parse_args()

    dest_dir = args.dest_directory
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    create_npy_csv_files(directory=right_slash(args.directory), dest_directory=right_slash(dest_dir),
                         dest_filename=args.dest_f)

    # no slash for directory,
    # yes slash for dest_directory

    # Es: python ./utils/ply_to_numpy.py --directory C:/Users/chiar/Desktop/FRGC_Bosph_registeredMeshes_TPAMI_noses
    # --dest_directory C:/Users/PycharmProjects/Neural3DMM_noses/npy_nuovo_dataset/ --dest_f pino







# def main():
#     # vogliamo: train.npy (number_meshes, number_vertices, 3) (no Faces because they all share topology)
#     # ------ args:
#     # TODO da convertire in args o da calcolare....
#     train_pct = None
#     test_set = {0, 1, 2}
#     num_individui = 12  # indici vanno da 0 a 11 #questo è possibile ricavarlo.....
#     # -------- end args
#
#     ''' per fare il test solo con parte alta dei nasi ... directory dovrà essere la cartella con solo
#     la parte alta dei nasi. per il resto tutto dovrebbe essere uguale'''
#
#     # directory = "../../COMA_data_noses"
#     directory = "/home/egrappolini/CG3D/COMA_data_noses_TOP"
#
#     dest_directory = "/home/egrappolini/CG3D/Neural3DMM/data/parte_alta_nasi_COMA/"
#
#     total_set = set(range(12))
#
#     train_set = total_set - test_set
#
#     sample_file = directory + "/" + "FaceTalk_170725_00137_TA/bareteeth/bareteeth.000001.ply"
#     complete = np.zeros(get_numpy_from_file(sample_file).shape)
#     complete = np.expand_dims(complete, axis=0)
#
#     complete_train = np.zeros(get_numpy_from_file(sample_file).shape)
#     complete_train = np.expand_dims(complete_train, axis=0)
#
#     complete_test = np.zeros(get_numpy_from_file(sample_file).shape)
#     complete_test = np.expand_dims(complete_test, axis=0)
#
#     subdir_list = [f for f in sorted(listdir(directory)) if not isfile(join(directory, f))]  # soggetti
#
#     temp_subdir = directory + "/" + subdir_list[0]
#     subsubdir_list = [f for f in sorted(listdir(temp_subdir)) if not isfile(join(temp_subdir, f))]  # espressioni
#     expr_dict = {}
#     i = 0
#     for item in subsubdir_list:
#         expr_dict[item] = i
#         i += 1
#
#     i = 0
#     tot = 0
#     subdir_index = 0
#
#     individuals_test = []
#     expressions_test = []
#     indiv_name_test = []
#     expr_name_test = []
#     mesh_file_name_test = []  # nomi completi delle mesh di test (es: bareteeth.000001.ply)
#
#     individuals_train = []
#     expressions_train = []
#     indiv_name_train = []
#     expr_name_train = []
#     mesh_file_name_train = []  # nomi completi delle mesh di train
#
#     print("SHAPE")
#     print(complete_train.shape)
#
#     tests = ['FaceTalk_170731_00024_TA', 'FaceTalk_170904_00128_TA', 'FaceTalk_170811_03275_TA']
#     test_flag = False
#     for subdir in tqdm(subdir_list):
#         # print(f"Iteration {i} of {len(subdir_list)}")
#         # print(subdir)
#
#         if subdir in tests:
#             print(f"Subdir {subdir} is in test")
#             test_flag = True
#         else:
#             print(f"Subdir {subdir} is in train")
#             test_flag = False
#         temp_subdir = directory + "/" + subdir
#         subsubdir_list = [f for f in sorted(listdir(temp_subdir)) if not isfile(join(temp_subdir, f))]
#         for subsubdir in tqdm(subsubdir_list, leave=False):
#             # print(subsubdir)
#             temp_subsubdir = temp_subdir + "/" + subsubdir
#             files_list = [f for f in sorted(listdir(temp_subsubdir)) if isfile(join(temp_subsubdir, f))]
#             # print(files_list)
#             for item in tqdm(files_list, leave=False):
#                 input_file = temp_subsubdir + "/" + item
#                 # print(input_file)
#                 temp = get_numpy_from_file(input_file)
#                 temp = np.expand_dims(temp, axis=0)
#                 if test_flag:
#                     complete_test = np.concatenate((complete_test, temp), axis=0)
#                     individuals_test.append(i)
#                     expressions_test.append(expr_dict[subsubdir])
#                     indiv_name_test.append(subdir)
#                     expr_name_test.append(subsubdir)
#                     mesh_file_name_test.append(item)
#                 else:
#                     complete_train = np.concatenate((complete_train, temp), axis=0)
#                     individuals_train.append(i)
#                     expressions_train.append(expr_dict[subsubdir])
#                     indiv_name_train.append(subdir)
#                     expr_name_train.append(subsubdir)
#                     mesh_file_name_train.append(item)
#                 tot += 1
#
#         i += 1
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
#     myMetadata_test = pd.DataFrame(
#         {
#             'individual_id': individuals_test,
#             'expression_id': expressions_test,
#             'individual_name': indiv_name_test,
#             'expression_name': expr_name_test,
#             'mesh_file_name': mesh_file_name_test
#         }
#     )
#
#     myMetadata_train = pd.DataFrame(
#         {
#             'individual_id': individuals_train,
#             'expression_id': expressions_train,
#             'individual_name': indiv_name_train,
#             'expression_name': expr_name_train,
#             'mesh_file_name': mesh_file_name_train
#         }
#     )
#     # print(myMetadata_train.head(5))
#
#     # print(f"Finish: temp shape is {complete.shape}, final shape is: ")
#     # complete = complete[1:, :, :]
#     # print(complete.shape)
#
#     name_train = dest_directory + "train_nasi_top.npy"
#     name_test = dest_directory + "test_nasi_top.npy"
#
#     with open(name_train, 'wb') as file:
#         np.save(file, complete_train)
#
#     with open(name_test, 'wb') as file:
#         np.save(file, complete_test)
#
#     myMetadata_test.to_csv(dest_directory + "test_nasi_top_metadata.csv")
#     myMetadata_train.to_csv(dest_directory + "train_nasi_top_metadata.csv")
#
#     # with open("train_nasi.npy", 'wb') as file:
#     #     np.save(dest_directory + file, complete_train)
#     #
#     # with open("test_nasi.npy", 'wb') as file:
#     #     np.save(dest_directory + file, complete_test)
#
#     # myMetadata_test.to_csv('test_nasi_metadata.csv')
#     # myMetadata_train.to_csv('train_nasi_metadata.csv')
#
#
# if __name__ == '__main__':
#     main()
#
#     '''
#     files_list = [f for f in listdir(directory) if isfile(join(directory, f))]
#     print(files_list)
#
#     input_file = directory + "/" + files_list[0]
#     complete = get_numpy_from_file(input_file)
#     complete = np.expand_dims(complete, axis=0)
#     print(complete.shape)
#     files_list.pop(0) # ATTENZIONE, NON USARE PIU' QUESTA LISTA PERCHE' IL PRIMO ELEMENTO VIENE DISTRUTTO (poco elegante ma pragmatico)
#
#     for item in subdir_list:
#         input_file = directory + "/" + item
#         print(input_file)
#         #temp = get_numpy_from_file(input_file)
#         #temp = np.expand_dims(temp, axis=0)
#         #complete = np.concatenate((complete, temp), axis=0)
#
#     with open("train.npy", 'wb') as file:
#         np.save(file, complete)
#     '''
#     # test
#     # data = np.load('train.npy')
#     # print(data.shape)
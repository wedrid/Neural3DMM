""" Cerca di ricavare info aggiuntive per il clustering (per etichettare meglio le mesh) """

import numpy as np
from collections import Counter


def get_comments_b(files):  # per bosphorus dataset
    comment_list = []
    for k in range(len(files)):
        name = files[k]  # bs00....ply
        spl = name.split('_')
        if spl[1] == 'E':
            comment_list.append(spl[1] + '_' + spl[2])  # E_SURPRISE etc
        else:
            comment_list.append(spl[1])  # CAU, LFAU etc

    return comment_list


def get_comments_f(voting_matrix):  # per FRGCV dataset
    comment_list = []

    # la voting matrix ha tante righe quante sono le mesh considerate (per ognuno guardo gli 1 ed etichetto)
    for k in range(len(voting_matrix)):
        mesh = voting_matrix[k]  # [0, 1, 1, 0]
        if mesh[0] == 1 and mesh[1] == 1:
            # print("NN")
            comment_list.append('NN')
        elif mesh[1] == 1 and mesh[2] == 1:
            # print("NS")
            comment_list.append('NS')
        elif mesh[1] == 1 and mesh[3] == 1:
            # print("NL")
            comment_list.append('NL')
        elif mesh[2] == 1:
            # print("S")
            comment_list.append('S')
        elif mesh[3] == 1:
            # print("L")
            comment_list.append('L')
        elif mesh[0] == 1 or mesh[1] == 1:
            # print("N")
            comment_list.append('N')
        else:
            print("Eroore")
            break

    return comment_list


def get_neutrals(dataset, files, unici, counts):

    files = [f.rstrip('.ply') for f in files]
    if dataset == 'F':
            print("FRGCV dataset")

            file_txt = "./utils/info_files/Geometrix_Exp3_ExpressionCategories.txt"
            neutral_txt = "./utils/info_files/list_frgcv2_neutral.txt"
            # se eseguo da console
            # file_txt = "./info_files/Geometrix_Exp3_ExpressionCategories.txt"
            # neutral_txt = "./info_files/list_frgcv2_neutral.txt"

            # righe file Geometrix_Exp3_ExpressionCategories.txt
            lines = []  # contiene tutte le righe non vuote del file
            with open(file_txt, "r") as f:
                for line in f:
                    if line != '\n':
                        lines.append(line.rstrip('\n'))  # rimuove \n alla fine di ogni riga letta

            n_index = lines.index('Neutral expressions:')
            n_num = lines[n_index + 1]  # numero di righe che contengono mesh 'neutral'
            neutrals = lines[n_index + 2: n_index + 2 + int(n_num)]

            s_index = lines.index('Small expressions:')
            s_num = lines[s_index + 1]  # numero di righe che contengono mesh 'small'
            smalls = lines[s_index + 2: s_index + 2 + int(s_num)]

            l_index = lines.index('Large expressions:')
            l_num = lines[l_index + 1]  # numero di righe che contengono mesh 'large'
            larges = lines[l_index + 2: l_index + 2 + int(l_num)]

            # righe file list_frgcv2_neutral.txt
            lines_n = []
            with open(neutral_txt, "r") as f:
                for line in f:
                    if line != '\n':
                        lines_n.append(line.rstrip('\n'))  # rimuove \n alla fine di ogni riga letta

            # matrice di 'votazione', per ogni mesh salvo 1 se il file 'vota' una categoria
            # ['neutrals', 'lines_n', 'smalls', 'larges'], 0 altrimenti
            voting_matrix = (np.zeros((len(files), 4)))  # tutte le mesh del dataset in esame

            for k in range(len(files)):
                name = files[k]
                if name in neutrals:
                    voting_matrix[k, 0] = 1
                if name in lines_n:
                    voting_matrix[k, 1] = 1
                if name in smalls:
                    voting_matrix[k, 2] = 1
                if name in larges:
                    voting_matrix[k, 3] = 1

            ext_comments_list = get_comments_f(voting_matrix)

            current_ind = 0
            filtered_list = []
            comments_list = []  # aiuta a capire la composizione di filtered_list, per ogni mesh c'è la categoria cui appartiene (NN, S, U, NS)
            # u_list = []  # come comment list ma solo per le mesh 'U' (voglio capire che tipo sono loro)
            repl_list = []  # replico tante volte quante sono le mesh per ogni individuo la mesh neutrale che trovo
            for i in range(len(unici)):
                print("individuo: ", unici[i])
                num_mesh_individuo = counts[i]
                print("indices: ", current_ind, current_ind + num_mesh_individuo)

                # sottomatrice da esaminare
                mat = voting_matrix[current_ind:current_ind + num_mesh_individuo, :]

                ind0 = np.where(mat[:, 0] == 1)  # indici delle righe che hanno 1 nella colonna 0 (neutrals)  (è una tupla)
                ind1 = np.where(mat[:, 1] == 1)  # indici delle righe che hanno 1 nella colonna 1 (lines_n)

                intersection = np.intersect1d(ind0[0], ind1[0])  # intersezione (è un ndarray)
                if len(intersection):
                    print("lista piena")
                    # devo sommare current_ind per rimappare gli indici
                    ind = current_ind + intersection[0]  # prendo il primo elemento
                    filtered_list.append(ind)
                    comments_list.append('NN')
                    #####
                    for k in range(num_mesh_individuo):
                        repl_list.append(ind)
                    #####
                else:

                    # se la mesh è unica ovviamente devo prendere quella mesh (neutrale o non che sia)
                    if num_mesh_individuo == 1:
                        print('una sola mesh')
                        ind = current_ind
                        filtered_list.append(ind)
                        comments_list.append('U')
                        #####
                        repl_list.append(ind)
                        #####
                        # if len(ind0[0]) != 0:
                        #     u_list.append('N')
                        # else:
                        #     ind2 = np.where(mat[:, 2] == 1)
                        #     if len(ind2) != 0:
                        #         u_list.append('S')
                        #     else:
                        #         u_list.append('L')
                    else:
                        print("provo con una condizione più leggera N e S")
                        ind2 = np.where(mat[:, 2] == 1)  # indici delle righe che hanno 1 nella colonna 2 (smalls)
                        inter = np.intersect1d(ind1[0], ind2[0])  # lines_n e small
                        if len(inter):
                            ind = current_ind + inter[0]  # prendo il primo elemento
                            filtered_list.append(ind)
                            comments_list.append('NS')
                            #####
                            for k in range(num_mesh_individuo):
                                repl_list.append(ind)
                            #####
                        else:
                            print("provo con una condizione ancora più leggera N e L")
                            ind3 = np.where(mat[:, 3] == 1)  # indici delle righe che hanno 1 nella colonna 3 (larges)
                            inte = np.intersect1d(ind1[0], ind3[0])  # lines_n e larges

                            if len(inte):
                                ind = current_ind + inte[0]  # prendo il primo elemento
                                filtered_list.append(ind)
                                comments_list.append('NL')
                                #####
                                for k in range(num_mesh_individuo):
                                    repl_list.append(ind)
                                #####
                            else:
                                if len(ind2[0]):
                                    print("solo small")
                                    ind = current_ind + ind2[0][0]  # prendo il primo elemento
                                    filtered_list.append(ind)
                                    comments_list.append('S')
                                    #####
                                    for k in range(num_mesh_individuo):
                                        repl_list.append(ind)
                                    #####

                                elif len(ind3[0]):
                                    print("solo large")
                                    ind = current_ind + ind3[0][0]  # prendo il primo elemento
                                    filtered_list.append(ind)
                                    comments_list.append('L')
                                    #####
                                    for k in range(num_mesh_individuo):
                                        repl_list.append(ind)
                                    #####
                                else:
                                    print("lista vuota !!!!")
                                    break
                current_ind = current_ind + num_mesh_individuo

            counter = Counter(comments_list)
            print("Counter of comments: ", counter)

            if len(filtered_list) != len(unici):
                raise NotImplementedError("ERRORE: non ho una mesh 'neutrale' per ogni individuo")

            r_list = repl_list
            c_list = ext_comments_list

    else:
        print("Bosphorus dataset")

        current_ind = 0
        b_filtered_list = []
        b_repl_list = []
        comm_list = get_comments_b(files)
        for m in range(len(unici)):  # ciclo sul numero di individui
            print("indici: ", current_ind, current_ind + counts[m])
            current_list = files[current_ind:current_ind + counts[m]]
            ind = [i for i, elem in enumerate(current_list) if 'N_N' in elem]
            if len(ind) != 0:
                b_filtered_list.append(current_ind + ind[0])  # devo rimappare con current ind
                # replico tante volte l'indice della mesh neutrale per l'individuo corrente
                # quante sono le mesh dell'individuo corrente
                for k in range(counts[m]):
                    b_repl_list.append(current_ind + ind[0])
            else:
                print("problemi")
                break
            # update
            current_ind = current_ind + counts[m]

        r_list = b_repl_list
        c_list = comm_list

    return r_list, c_list  # voting_matrix


# if __name__ == '__main__':
#     from ply_to_numpy import right_slash
#     from ply_to_numpy_new_dataset import get_heading
#     from os import listdir
#     from os.path import isfile, join
#
#     directory = 'D:/Progetto_CG3D/FRGC_Bosph_registeredMeshes_TPAMI_noses'
#     subdir_list = [f for f in sorted(listdir(directory)) if not isfile(join(directory, f))]
#     temp_subdir = directory + "/" + subdir_list[1]
#     files = [f for f in sorted(listdir(temp_subdir)) if isfile(right_slash(join(temp_subdir, f)))]
#
#     files_head = []
#     for i in range(len(files)):
#         f = files[i]
#         f = get_heading(f)
#         files_head.append(f)
#
#     [unici, counts] = np.unique(files_head, return_counts=True)
#
#     ###### FUNZIONE
#     dataset = subdir_list[0][1]  # 0 o 1 nella seconda parentesi
#     [r, c] = get_neutrals(dataset, files, unici, counts)  # v




# def get_neutrals(dataset, csv_file):
#
#     data = pd.read_csv(csv_file, names=["mesh_file_name"])
#
#     names = list(data["mesh_file_name"])
#
#     f_names = []  # nomi di FRGCV
#     b_names = []  # nomi di boshporus
#     for name in names[1:]:  # l'elemento 0 è la label della colonna
#         if 'bs' not in name:
#             f_names.append(name.rstrip('.ply'))
#         else:
#             b_names.append(name.rstrip('.ply'))
#
#     # poi tutto uguale, ma devo usare f_names e b_names al posto di files e ricalcolare per entrambi unici e counts

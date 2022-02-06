%%% Funzione per calcolare alcune distanze geodetiche tra
% vertici fissati, per tutte le mesh del dataset B-FRGC. 
% I valori delle distanze calcolate vengono salvati nei file csv di train e
% test, insieme agli altri metadati associati ad ogni mesh del dataset.

% pathToDataset: dove si trova la cartella del dataset % no slash finale
% pathToTrainCsv, pathToTestCsv: path al file csv (train/test) in cui sono
% salvati i metadati associati ad ogni mesh del dataset

% Es:
% pathToDataset = '...\Progetto_CG3D\FRGC_Bosph_registeredMeshes_TPAMI_noses_TOP'); 
% pathToTrainCsv = '...\train_top_metadata.csv';
% pathToTestCsv = '...\test_top_metadata.csv';
% indici dei vertici (inizio e fine) di cui si vuole calcolare le distanze

% Start points geodetiche (scelti sulla parte alta del naso)
sp = [438, 473, 311]; % up, down, vertical
% End points geodetiche (scelti sulla parte alta del naso)
ep = [471, 73, 58]; % up, down, vertical

% % Invertiti
% % Start points geodetiche
% sp = [471, 73, 58]; % up, down, vertical
% % End points geodetiche
% ep = [438, 473, 311]; % up, down, vertical

% Controllo dataset
% if contains(pathToDataset, 'TOP')
%     disp("Parte alta nasi");
% else
%     disp("Nasi");
%     % dato che i vertici sp ed ep sono stati scelti sulla mesh di
%     % riferimento della parte alta del naso, dobbiamo rimapparli su quella
%     % di riferimento del naso per poter calcolare le distanze sui nasi
%     % (in realtà le distanze sono le stesse... mi basta calcolarle sulla
%     % parte alta e poi copiare i csv anche per i nasi) 
%     pathToRefMesh = 'C:/Users/chiar/Desktop/da_selezionare/nuovi_dataset/ref_mesh_naso_new_cut.ply'; % parte alta
%     pathToOriginalMesh = 'C:/Users/chiar/Desktop/da_selezionare/nuovi_dataset/ref_mesh_naso_new.ply'; % naso
%     [vertex_cut, face_cut] = extract_vertex_face(pathToRefMesh); 
%     [vertex_naso, ~] = extract_vertex_face(pathToOriginalMesh);
%     remapping_vertices = get_remapping_vertices(vertex_cut, vertex_naso);
%     for k=1:length(sp)
%         sp(k) = remapping_vertices(sp(k)); 
%         ep(k) = remapping_vertices(ep(k)); 
%     end
% end

% Carico csv train
t = readtable(pathToTrainCsv);
train_names = table2array(t(:, 2));  % è un cell array

% Carico csv test
t1 = readtable(pathToTestCsv);
test_names = table2array(t1(:, 2));  % è un cell array

% Inizializzo gli array per le geodetiche
up_geods_train = zeros(length(train_names), 1); % tante righe quante quelle del csv di train
down_geods_train = zeros(length(train_names), 1);
vertical_geods_train = zeros(length(train_names), 1);

up_geods_test = zeros(length(test_names), 1);
down_geods_test = zeros(length(test_names), 1);
vertical_geods_test = zeros(length(test_names), 1);

% Recupero i nomi delle cartelle
d = dir(pathToDataset); % ho due sotto cartelle che fungono da due dataset 
for i=1:length(d)
    el = d(i).name; % F o B
    if startsWith(el, 'F') || startsWith(el, 'b')
        fprintf("Sottocartella: %s \n", el);
        pi = fullfile(pathToDataset, el); 
        files = dir(pi); 
        for j=1:length(files)
            if files(j).isdir == 0
                pathToFile = fullfile(pi, files(j).name);  
                fprintf("pathToFile: %s \n", pathToFile)
                
                % Leggo la mesh
                [vertex, face, options] = extract_vertex_face(pathToFile);
                
                % Calcolo geodesiche
                % chiamiamo la funzione 3 volte (una per ogni coppia di
                % punti tra cui vogliamo calcolare la distanza)
                up_geod = compute_2vertex_geod(vertex, face, sp(1), ep(1), options, 1); % valore distanza
                down_geod = compute_2vertex_geod(vertex, face, sp(2), ep(2), options, 2); 
                vertical_geod = compute_2vertex_geod(vertex, face, sp(3), ep(3), options, 3); 
                
                % Controllo valori alti
                sub_up = 0;
                if up_geod > 40
                    disp("Calcolo la distanza UP invertendo i vertici di inizio e fine");
                    up_geod_inv = compute_2vertex_geod(vertex, face, ep(1), sp(1), options, 1);
                    if up_geod_inv < up_geod
                        disp("Sostituisco UP");
                        up_geod = up_geod_inv;
                        sub_up = 1; 
                    end
                end
                
                sub_down = 0; 
                if down_geod > 40
                    disp("Calcolo la distanza DOWN invertendo i vertici di inizio e fine");
                    down_geod_inv = compute_2vertex_geod(vertex, face, ep(2), sp(2), options, 2);
                    if down_geod_inv < down_geod
                        disp("Sostituisco DOWN");
                        down_geod = down_geod_inv;
                        sub_down = 1; 
                    end
                end
                
                sub_vert = 0; 
                if vertical_geod > 40
                    disp("Calcolo la distanza VERTICAL invertendo i vertici di inizio e fine");
                    vertical_geod_inv = compute_2vertex_geod(vertex, face, ep(3), sp(3), options, 3);
                    if vertical_geod_inv < vertical_geod
                        disp("Sostituisco VERTICAL");
                        vertical_geod = vertical_geod_inv;
                        sub_vert = 1; 
                    end
                end
                
                % controllo i valori ... se sono troppo bassi provo a
                % ricalcolare invertendo i vertici e poi confronto i valori
                % ... prendendo il massimo dei due 
                if up_geod < 20
                    disp("Calcolo la distanza UP invertendo i vertici di inizio e fine");
                    if sub_up == 1 % significa che avevo già fatto una sostituzione
                        disp("Riprendo la distanza UP originale");
                        up_geod_inv = compute_2vertex_geod(vertex, face, sp(1), ep(1), options, 1);
                    else
                        up_geod_inv = compute_2vertex_geod(vertex, face, ep(1), sp(1), options, 1);
                    end
                    if up_geod_inv > up_geod
                        disp("Sostituisco UP");
                        up_geod = up_geod_inv;
                    end
                end
                
                if down_geod < 20
                    disp("Calcolo la distanza DOWN invertendo i vertici di inizio e fine");
                    if sub_down == 1 % significa che avevo già fatto una sostituzione
                        disp("Riprendo la distanza DOWN originale");
                        down_geod_inv = compute_2vertex_geod(vertex, face, sp(2), ep(2), options, 2);
                    else
                        down_geod_inv = compute_2vertex_geod(vertex, face, ep(2), sp(2), options, 2);
                    end
                    
                    if down_geod_inv > down_geod
                        disp("Sostituisco DOWN");
                        down_geod = down_geod_inv;
                    end
                end
                
                if vertical_geod < 20
                    disp("Calcolo la distanza VERTICAL invertendo i vertici di inizio e fine");
                    if sub_vert == 1 % significa che avevo già fatto una sostituzione
                        disp("Riprendo la distanza UP originale");
                        vertical_geod_inv = compute_2vertex_geod(vertex, face, sp(3), ep(3), options, 3);
                    else
                        vertical_geod_inv = compute_2vertex_geod(vertex, face, ep(3), sp(3), options, 3);
                    end
                    
                    if vertical_geod_inv > vertical_geod
                        disp("Sostituisco VERTICAL");
                        vertical_geod = vertical_geod_inv;
                    end
                end
                
                % controllo se train o test
                ind = find(strcmp(files(j).name, train_names));  % indice per il csv
                if isempty(ind) == 0 % se non è vuoto --> train
                    fprintf("TRAIN mesh: %s, index in train_names: %d \n", files(j).name, ind);

                    up_geods_train(ind) = up_geod; 
                    down_geods_train(ind) = down_geod;
                    vertical_geods_train(ind) = vertical_geod;
                else
                    fprintf("TEST mesh: %s, ", files(j).name);
                    ind = find(strcmp(files(j).name, test_names));
                    fprintf("index in test_names: %d \n", ind);
                     
                    up_geods_test(ind) = up_geod;
                    down_geods_test(ind) = down_geod;
                    vertical_geods_test(ind) = vertical_geod;
                end

            end
        end
    end

end

% Add columns to csv (t è train, t1 è test)

% Train (la tabella è t)
ct = size(t, 2);
t.(ct+1) = up_geods_train;
t.Properties.VariableNames{ct+1} = 'u_geod';
t.(ct+2) = down_geods_train;
t.Properties.VariableNames{ct+2} = 'd_geod';
t.(ct+3) = vertical_geods_train;
t.Properties.VariableNames{ct+3} = 'v_geod';

% Save new train csv 
[pathStr, name, ext] = fileparts(pathToTrainCsv);
pathToNewTrainCsv = fullfile(pathStr, name + "_geods_ctrl_2" + ext);
writetable(t, pathToNewTrainCsv)

% Test (la tabella è t1)

ct1 = size(t1, 2); 
t1.(ct1+1) = up_geods_test;
t1.Properties.VariableNames{ct1+1} = 'u_geod';
t1.(ct1+2) = down_geods_test;
t1.Properties.VariableNames{ct1+2} = 'd_geod';
t1.(ct1+3) = vertical_geods_test;
t1.Properties.VariableNames{ct1+3} = 'v_geod';

% Save new test csv
[pathStr, name, ext] = fileparts(pathToTestCsv);
pathToNewTestCsv = fullfile(pathStr, name + "_geods_ctrl_2" + ext);
writetable(t1, pathToNewTestCsv)


%%% Calcolo distanza geodetica

function [L, paths] = compute_2vertex_geod(vertex, faces, start_point, end_point, options, c_ind)
[D,~,~] = perform_fast_marching_mesh(vertex, faces, start_point, options);

% precompute some usefull information about the mesh
options.v2v = compute_vertex_ring(faces);
options.e2f = compute_edge_face_ring(faces);
% extract the geodesics
options.method = 'continuous';
options.verb = 0;
paths = compute_geodesic_mesh(D, vertex, faces, end_point, options);

% scommentare per display

% if c_ind == 1  % per rappresentare le 3 curve in una stessa figure
%     figure;
% end
% options.colorfx = 'equalize';
% plot_fast_marching_mesh(vertex, faces, D, paths, options);
% shading interp;

% calculate curve length
 
x = paths(1, :);
y = paths(2, :);
z = paths(3, :);
L = sum(sqrt(diff(x).^2 + diff(y).^2 + diff(z).^2));

end

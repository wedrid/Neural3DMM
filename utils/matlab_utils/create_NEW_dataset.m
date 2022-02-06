% Funzione per replicare il dataset B-FRGC, ma con solo mesh di nasi/parte alta
% nasi

% Per replicare dataset NASI:

% pathToRefMesh: path alla mesh di riferimento (naso) (file .ply) % estratta ed esportata con Meshlab o altro
% pathToOriginalMesh: path alla mesh della faccia da cui pathToRefMesh è stata estratta (file .ply)
% pathToDataset: path alla cartella del dataset CoMA (facce) % no slash finale
% pathToNewDataset: path alla cartella in cui si vuole salvare il nuovo
% dataset % no slash finale

% Per replicare dataset PARTE ALTA NASI:

% pathToRefMesh: path alla mesh di riferimento (parte alta naso) (file .ply) % estratta ed esportata con Meshlab o altro
% pathToOriginalMesh: path alla mesh del naso da cui pathToRefMesh è stata estratta(file .ply)
% pathToDataset: path alla cartella del dataset CoMA (nasi) % no slash finale
% pathToNewDataset: path alla cartella in cui si vuole salvare il nuovo
% dataset % no slash finale

% ES: 
% pathToRefMesh = '...\ref_mesh_naso_new.ply';
% pathToOriginalMesh = '..\ref_faccia_new.ply';
% pathToDataset = '...\Progetto_CG3D\FRGC_Bosph_registeredMeshes_TPAMI_PLY';
% pathToNewDataset = '...\Progetto_CG3D\FRGC_Bosph_registeredMeshes_TPAMI_noses';



% Estrarre vertici e facce dalla mesh naso di riferimento 
[vertex_nose, face_nose] = extract_vertex_face(pathToRefMesh); 
% vertex_nose = round(vertex_nose, 6); % solo quando creo il dataset dei
% nasi (per la parte alta dei nasi no !)

% Estrarre i vertici dalla mesh da cui è stato estratto il naso di
% riferimento
[vertex, ~] = extract_vertex_face(pathToOriginalMesh);  
% risultano arrotondati (a 6 cifre significative) rispetto a modFinal (la
% corrispondente matrice di vertici letta direttamente dal file .mat)

% Seleziono gli indici corrispondenti al naso, nella mesh originale
remapping_vertices = get_remapping_vertices(vertex_nose, vertex);

% Recupero i nomi delle cartelle
d = dir(pathToDataset); % ho due sotto cartelle che fungono da due dataset
d_folders = ([]); 
j = 1; 
for i=1:length(d)
    el = d(i).name;
    if startsWith(el, 'F') || startsWith(el, 'b')
        d_folders(j).name = el; 
        fprintf("Sottocartella: %s \n", el);
        j = j + 1; 
    else
    end
end

% Creo il nuovo dataset 
if isfolder(pathToNewDataset) ~= 1
    mkdir(pathToNewDataset);
end

for i=1:length(d_folders)
    p = fullfile(pathToNewDataset, d_folders(i).name); 
    fprintf("Cartella: %s \n", p); 
    if isfolder(p) ~= 1
        mkdir(p);
    end
    
    pi = fullfile(pathToDataset, d_folders(i).name); 
    files = dir(pi); 
    for j=1:length(files)
        if files(j).isdir == 0
            pathToFile = fullfile(pi, files(j).name);  
            fprintf("pathToFile: %s \n", pathToFile)
          
            pathToDest =  p; % dove salvo la nuova mesh
            if exist(pathToDest, 'file') == 2
                disp("Naso già presente");
            else
                disp("Naso NON presente");
                extract_nose(face_nose, remapping_vertices, pathToFile, pathToDest)
            end
            
            
        end
    end
    
end





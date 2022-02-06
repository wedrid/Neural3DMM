% Funzione per replicare la gerarchia del dataset CoMA, ma con mesh di nasi/parte alta
% nasi

% Per replicare dataset NASI:

% pathToRefMesh: path alla mesh di riferimento (naso) (file .ply)  % estratta ed esportata con Meshlab o altro
% pathToOriginalMesh: path alla mesh della faccia da cui pathToRefMesh è stata estratta (file .ply)
% pathToDataset: path alla cartella del dataset CoMA (facce) % no slash finale
% pathToNewDataset: path alla cartella in cui si vuole salvare il nuovo
% dataset % no slash finale

% Per replicare dataset PARTE ALTA NASI:

% pathToRefMesh: path alla mesh di riferimento (parte alta naso) (file .ply)  % estratta ed esportata con Meshlab o altro
% pathToOriginalMesh: path alla mesh del naso da cui pathToRefMesh è stata estratta(file .ply)
% pathToDataset: path alla cartella del dataset CoMA (nasi) % no slash finale
% pathToNewDataset: path alla cartella in cui si vuole salvare il nuovo
% dataset % no slash finale

% ES: 
% pathToRefMesh = '...\ref_mesh_naso.ply';
% pathToOriginalMesh = '...\Progetto_CG3D\COMA_data\FaceTalk_170725_00137_TA\bareteeth\bareteeth.000001.ply';
% pathToDataset = '...\Progetto_CG3D\COMA_data';
% pathToNewDataset = '...\Progetto_CG3D\COMA_data_noses';


% Estrarre vertici e facce dalla mesh naso di riferimento 
[vertex_nose, face_nose] = extract_vertex_face(pathToRefMesh); 

% Estrarre i vertici dalla mesh da cui è stato estratto il naso di
% riferimento
[vertex, ~] = extract_vertex_face(pathToOriginalMesh);

% Seleziono gli indici corrispondenti al naso, nella mesh originale
remapping_vertices = get_remapping_vertices(vertex_nose, vertex);

% Recupero i nomi delle cartelle FaceTalk (soggetti) 
d = dir(pathToDataset); 
d_folders = ([]); % facetalk
j = 1; 
for i=1:length(d)
    el = d(i).name;
    if startsWith(el, 'F')
        d_folders(j).name = el; 
        j = j + 1; 
    else
    end
end


% Recupero i nomi delle cartelle delle espressioni
f = fullfile(pathToDataset, d_folders(1).name); % mi basta il primo tanto sono tutte uguali
e = dir(f); 
e_folders = ([]); % espressioni
j = 1; 
for i=1:length(e)
    el = e(i).name;
    if startsWith(el, '.') ~= 1 && startsWith(el, '..') ~= 1 
        e_folders(j).name = el; 
        j = j + 1; 
    else
    end
end

% Creo il nuovo dataset 
if isfolder(pathToNewDataset) ~= 1
    mkdir(pathToNewDataset);
end

for i=1:length(d_folders) % FaceTalk
    p = fullfile(pathToNewDataset, d_folders(i).name); 
    fprintf("Cartella FaceTalk: %s \n", p); 
    if isfolder(p) ~= 1
        mkdir(p);
    end
    
    for j=1:length(e_folders) % espressioni 
        pe = fullfile(p, e_folders(j).name);
        fprintf("Cartella espressioni: %s \n", pe); 
        if isfolder(pe) ~= 1
            mkdir(pe);
        end
        
        pathToFolder = fullfile(pathToDataset, d_folders(i).name, e_folders(j).name);
        ply_folder = dir(pathToFolder); 
        
        for k=1:length(ply_folder)
            el = ply_folder(k).name;    
            if ply_folder(k).isdir == 0 % prendo solo i file .ply
                pathToNewMesh = fullfile(pathToFolder, el);
                fprintf("Mesh da processare: %s \n", pathToNewMesh);
                pathToDest = pe; % dove salvo la nuova mesh 
                extract_nose(face_nose, remapping_vertices, pathToNewMesh, pathToDest);
            end
                
        end
    
    end
    
end
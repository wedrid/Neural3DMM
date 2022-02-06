%%% Funzione che crea il dataset B-FRGC con file .ply, a partire dal
%%% dataset di file.mat e dal modello di triangolazione delle mesh avgModel_bh_1779_NE_tri.mat

% pathToDataset: path al dataset FRGC_Bosph_registeredMeshes_TPAMI (file in
% .mat)
% pathToNewDataset: path dove si vuole salvare il nuovo dataset (file .ply)
% pathToFaceModel: path al modello di triangolazione delle facce

% Es:
% pathToDataset = '...\Progetto_CG3D\FRGC_Bosph_registeredMeshes_TPAMI'
% pathToNewDataset = '...\Progetto_CG3D\FRGC_Bosph_registeredMeshes_TPAMI_PLY'
% pathToFaceModel = '...\avgModel_bh_1779_NE_tri.mat'

load(pathToFaceModel, 'Tnew')
face = Tnew; 

% Recupero i nomi dei due sotto dataset
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
pathToNewDataset = fullfile(root, 'FRGC_Bosph_registeredMeshes_TPAMI_PLY'); 
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
            pathToFile = fullfile(pi, files(j).name);  % che è uun file .mat
            fprintf("pathToFile: %s \n", pathToFile)
            
            % Controllo se la nuova mesh è già presente
            pathToDest =  p; % dove salvo la nuova mesh
            spl = split(pathToFile, filesep); 
            dest_name = spl{length(spl)}; % nome.mat ma io voglio salvarlo in .ply
            spl1 = split(dest_name, '.');
            name = spl1{length(spl1) - 1}; % nome senza estensione
            pathToFilename = fullfile(pathToDest, name + ".ply"); 
            
            if exist(pathToFilename, 'file') == 2
                disp("File già presente"); 
            else
                var = load(pathToFile);
                names = fieldnames(var);
                if sum(contains(names, 'modFinal')) == 1
                    disp("modFinal presente !");
                    vertex_new = var.modFinal; % vertici faccia
                    fprintf("Path file salvato: %s \n\n", pathToFilename); 
                    plywrite(pathToFilename, face, vertex_new); 
                else
                    disp("modFinal NON presente");
                end
                    
            end
              
        end
    end
    
end

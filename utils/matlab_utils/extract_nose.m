%%% Funzione per estrarre da una qualunque mesh di una faccia, la mesh del naso, data una mesh di naso di riferimento

% Input: 
% pathToRefMesh: path al file .ply corrispondente alla mesh del naso di
% riferimento (estratta ed esportata con Meshlab o altro) 

% pathToOriginalMesh: path al file .ply corrispondente alla mesh della
% faccia da cui è stata estratta la mesh naso di riferimento

% pathToNewMesh: path al file.ply della mesh della faccia da cui si vuole
% estrarre il naso 

% pathToDest: path alla cartella in cui si vuole salvare la nuova mesh naso

function extract_nose(face_nose, remapping_vertices, pathToNewMesh, pathToDest)

% Estraggo i vertici della nuova mesh da cui voglio prendere il naso
[vertex_new, ~] = extract_vertex_face(pathToNewMesh);

% Estraggo gli indici dei vertici corrispondenti alla zona naso
sel_vertices = vertex_new(remapping_vertices, :);

% Salvo la nuova mesh naso 
spl = split(pathToNewMesh, filesep); 
dest_name = spl{length(spl)}; 
pathToFilename = fullfile(pathToDest, dest_name); 
fprintf("Path file salvato: %s \n\n", pathToFilename); 

plywrite(pathToFilename, face_nose, sel_vertices); 

end
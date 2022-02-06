% Funzione che estrae i vertici e le facce da una mesh (file.ply)

% pathToPly: path al file .ply contenente la mesh da cui si vogliono
% estrarre vertici e facce

function [vertex, face] = extract_vertex_face(pathToPly) 

% Load the mesh
name = pathToPly; 
options.name = name; 
[vertex,face] = read_mesh(name); 

end
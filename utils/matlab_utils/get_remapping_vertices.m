% Funzione che permette di 'rimappare' i vertici della mesh di riferimento
% (naso/parte alta naso) su quelli della mesh (faccia/naso) da cui è stata
% estratta

% vertex_nose: vertici della mesh di riferimento % estratti usando la funzione extract_vertex_face
% vertex: vertici della mesh da cui la mesh di riferimento è stata estratta % estratti usando la funzione extract_vertex_face

% remapping_vertices: lista dei vertici della mesh originale (faccia/naso)
% che corrispondono alla zona (naso/parte alta naso) di interesse

function remapping_vertices = get_remapping_vertices(vertex_nose, vertex)
remapping_vertices = []; 
j = 1; 
for i=1:length(vertex_nose)
    % i è il vertice in vertex_nose che deve essere rimappato in vertex
    row = vertex_nose(i, :);
    [r, ~, ~] = find(row == vertex); 
    if isempty(r) ~= 1  % se r non è vuoto 
        ind_map = r(1); % indice del vertice in vertex
        remapping_vertices(j) = ind_map;
        j = j + 1; 
    end
end

remapping_vertices = remapping_vertices'; 
end
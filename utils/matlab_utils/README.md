# Matlab utilities

We suggest (for simplicity) to save all the datasets (faces, noses, top part of noses for CoMA and B-FRGC) in a common 'base' folder. Here the new generated noses and top part of noses datasets are differentiated from the original face datasets with  `_noses ` and `_noses_TOP`. 

## Before starting:
- unzip *matlab_utils/functions* folder
- include *matlab_utils* folder in your Matlab path (Home/Environment-Set Path/Add folder with subfolders). Not all the subfolders are really used, however we suggest to add all of them to your path for compatibility. If you get problems, unset the conflict folders.
- Compile .mex file for the toolbox in *functions* folder. Probably *toolbox_graph* and *toolbox_general* are sufficient - start from them and then if you find some errors in the execution of the other scripts, try to compile the other  `_toolbox ` folders). We want to specify that in perform_front_propagation_3d_mex.cpp e in fm2dAniso.cpp in toolbook_graph/mex/ we have replaced `int* dims = new int[par->ndim];` with `mwSize* dims = new mwSize[par->ndim];` ([error](https://it.mathworks.com/matlabcentral/answers/462538-mxcreatenumericarray-error-cannot-convert-int-to-const-size_t-aka-const-long-long-unsigned-in)), in order to complete the the compile execution. If you get new errors, search for them on the Internet. 

## About the scripts:
- Inside all the files in *matlab_utils*, inputs, outputs and 'goal' of the file are specified. 
- *create_dataset* is for CoMA dataset, while *create_NEW_dataset* is for B-FRGC dataset (Bosphorus + FRGC).
- *get_statistics* compute some geodesic distances and it is set to be used only with B-FRGC dataset. It can also be adapted to CoMA with some modifications.
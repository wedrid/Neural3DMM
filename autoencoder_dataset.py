from torch.utils.data import Dataset
import torch
import numpy as np
import os


class autoencoder_dataset(Dataset):

    def __init__(self, root_dir, points_dataset, shapedata, normalization = True, dummy_node = True):
        
        self.shapedata = shapedata
        self.normalization = normalization
        self.root_dir = root_dir
        self.points_dataset = points_dataset
        self.dummy_node = dummy_node
        self.paths = np.load(os.path.join(root_dir, 'paths_'+points_dataset+'.npy'))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        basename = self.paths[idx]
        
        #print("BASENAME:")
        #print(basename) #di fatto è un numero, perchè i file in preprocessed/points_test/ sono del tipo [numero].npy
        #print("ospathjoin:")
        #print(os.path.join(self.root_dir,'points'+'_'+self.points_dataset, basename+'.npy'))
        #NOTA: verts_init ha solo UNA mesh... in particolare prende code del tipo: preprocessed/points_test/3956.npy
        verts_init = np.load(os.path.join(self.root_dir,'points'+'_'+self.points_dataset, basename+'.npy'))
        
        #print("VERTS INIT:")
        #print(type(verts_init))
        #print(verts_init.shape)
        #print(verts_init)
        if self.normalization:
            verts_init = verts_init - self.shapedata.mean
            verts_init = verts_init/self.shapedata.std
        verts_init[np.where(np.isnan(verts_init))]=0.0
        
        verts_init = verts_init.astype('float32')
        
        #NOTE: here there's a dummy node
        if self.dummy_node:
            verts = np.zeros((verts_init.shape[0]+1,verts_init.shape[1]),dtype=np.float32)
            verts[:-1,:] = verts_init
            verts_init = verts
        #print("Here")
        #print(verts_init.shape)
        verts = torch.Tensor(verts_init)
        

        sample = {'points': verts}
        #print("There")
        #print(verts.size())
        return sample
    
    def getWholeProcessedDataset(self, npy_filepath):
        # Takes in: path to the npy file containing the nodes of the mesh
        # Spits out: all of the data, processed in the same way it is processed in the __getItem__() method
        # i.e. if dummy node is needed it is added, basically.. per controllare basta guardare l'ultima riga di ogni subtensore
        all_data = np.load(npy_filepath)

        #processed data tensor
        if self.dummy_node:
            data_tensor = torch.zeros((all_data.shape[0], all_data.shape[1]+1, all_data.shape[2])) #NOTA: la seconda dimensione è a causa del 'dummy node'
        else:
            data_tensor = torch.zeros((all_data.shape[0], all_data.shape[1], all_data.shape[2])) #NOTA: la seconda dimensione è a causa del 'dummy node'
 
        #print("MY TENSOR")
        #print(data_tensor)
        #print("HELLO INSIDE")
        #print(all_data[1].shape)
        
        #print("VERTS INIT:")
        #print(type(verts_init))
        #print(verts_init.shape)
        #print(verts_init)

        #NOTA: il ciclo seguente è tutto fuorchè efficiente.. se dovesse diventare proibitivo, è da considerare vettorizzazione
        for i in range(all_data.shape[0]):
            verts_init = all_data[i]
            if self.normalization:
                verts_init = verts_init - self.shapedata.mean
                verts_init = verts_init/self.shapedata.std
            verts_init[np.where(np.isnan(verts_init))]=0.0
            
            verts_init = verts_init.astype('float32')
            
            #NOTE: here there's a dummy node
            if self.dummy_node:
                verts = np.zeros((verts_init.shape[0]+1,verts_init.shape[1]),dtype=np.float32)
                verts[:-1,:] = verts_init
                verts_init = verts
            #print("Here")
            #print(verts_init.shape)
            verts = torch.Tensor(verts_init)
            data_tensor[i] = verts

        #print("MY NEW TENSOR")
        #print(data_tensor)
        
        #print("There")
        #print(verts.size())
        return data_tensor

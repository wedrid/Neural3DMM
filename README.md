Commenti ADG
Per avviare: 

-> init.py: si occupa di creare la gerarchia delle directory, data_generation (ha bisogno di train.npy, test.npy, template.obj). Questa si chiama, in generale, una sola volta. 

-> create_downsampling_matrices.py: crea le matrici di downsampling. Serve mpi-mesh. Qui si può andare ad agire, volessimo cambiare il downsampling. 

-> main.py: si avvia il training.. da definirsi il dettaglio. 

## Sequenza chiamate (per ora): 
- `python init.py --root_dir path_to_root_dir --bool 0`  // crea la gerarchia delle cartelle sotto root_dir
- Inserisco train.npy e test.npy in preprocessed. Inserisco template.obj in template.
  (qui poi in pratica questi file dovranno essere generabili ... abbiamo bisogno di una funzione che ci faccia lo split
  e poi in base a quello li generi )
- `python init.py --root_dir path_to_root_dir --bool 1` // genera i file paths_train.npy etc 
- [Se eseguito in locale per Chiara e Niki] `python rename_paths.py --json_file path_to_dict_path_json_file --heading /mnt/c/`
- `python create_downsampling_matrices --dict path_to_dict_path_json` // se eseguito in locale per Chiara e Niki - usare il file con wls
- `python main.py ...`

# Su Yoda
- Crea root_dir ! (nell'esempio 'prova1')

- `python init.py --root_dir /home/egrappolini/CG3D/prova1 --bool 0`

- `python init.py --root_dir /home/egrappolini/CG3D/prova1 --bool 1`

- `python create_downsampling_matrices.py --dict /home/egrappolini/CG3D/prova1/dict_path.json`

### train
- `python main.py --dict /home/egrappolini/CG3D/prova1/dict_path.json --epochs 500`
### test
- `python main.py --dict /home/egrappolini/CG3D/prova1/dict_path.json --mode 'test' --checkpoint_file /home/egrappolini/CG3D/prova1/results/spirals_\ autoencoder/checkpoints/checkpoint490`

======== 
Comando per avviare model_extraction.py
python model_extraction.py --dict /home/egrappolini/CG3D/filtri_coma/dict_path.json --checkpoint_file checkpoint290



-------------------------------------------
![Neural3DMM architecture](images/architecture_figure1.png "Neural3DMM architecture")

# Project Abstract 
*Generative models for 3D geometric data arise in many important applications in 3D computer vision and graphics. In this paper, we focus on 3D deformable shapes that share a common topological structure, such as human faces and bodies. Morphable Models and their variants, despite their linear formulation, have been widely used for shape representation, while most of the recently proposed nonlinear approaches resort to intermediate representations, such as 3D voxel grids or 2D views. In this work, we introduce a novel graph convolutional operator, acting directly on the 3D mesh, that explicitly models the inductive bias
of the fixed underlying graph. This is achieved by enforcing consistent local orderings of the vertices of the graph,
through the spiral operator, thus breaking the permutation invariance property that is adopted by all the prior work
on Graph Neural Networks. Our operator comes by construction with desirable properties (anisotropic, topology-aware, lightweight, easy-to-optimise), and by using it as a building block for traditional deep generative architectures, we demonstrate state-of-the-art results on a variety of 3D shape datasets compared to the linear Morphable Model and other graph convolutional operators.* 

[Arxiv link](https://arxiv.org/abs/1905.02876)


# Repository Requirements

This code was written in Pytorch 1.1. We use tensorboardX for the visualisation of the training metrics. We recommend setting up a virtual environment using [Miniconda](https://docs.conda.io/en/latest/miniconda.html). To install Pytorch in a conda environment, simply run 

```
$ conda install pytorch torchvision -c pytorch
```

Then the rest of the requirements can be installed with 

```
$ pip install -r requirements.txt
```

### Mesh Decimation
For the mesh decimation code we use a function from the [COMA repository](https://github.com/anuragranj/coma) (the files **mesh_sampling.py** and **shape_data.py** - previously **facemesh.py** - were taken from the COMA repo and adapted to our needs). In order to decimate your template mesh, you will need the [MPI-Mesh](https://github.com/MPI-IS/mesh) package (a mesh library similar to Trimesh or Open3D).  This package requires Python 2. However once you have cached the generated downsampling and upsampling matrices, it is possible to run the rest of the code with Python 3 as well, if necessary.


# Data Organization

The following is the organization of the dataset directories expected by the code:

* data **root_dir**/
  * **dataset** name/ (eg DFAUST)
    * template
      * template.obj (all of the spiraling and downsampling code is run on the template only once)
      * downsample_method/
        * downsampling_matrices.pkl (created by the code the first time you run it)
    * preprocessed/
      * train.npy (number_meshes, number_vertices, 3) (no Faces because they all share topology)
      * test.npy 
      * points_train/ (created by data_generation.py)
      * points_val/ (created by data_generation.py)
      * points_test/ (created by data_generation.py)
      * paths_train.npy (created by data_generation.py)
      * paths_val.npy (created by data_generation.py)
      * paths_test.npy (created by data_generation.py)

# Usage

#### Data preprocessing 

In order to use a pytorch dataloader for training and testing, we split the data into seperate files by:

```
$ python data_generation.py --root_dir=/path/to/data_root_dir --dataset=DFAUST --num_valid=100
```

#### Training and Testing

For training and testing of the mesh autoencoder, we provide an ipython notebook, which you can run with 

```
$ jupyter notebook neural3dmm.ipynb
```

The first time you run the code, it will check if the downsampling matrices are cached (calculating the downsampling and upsampling matrices takes a few minutes), and then the spirals will be calculated on the template (**spiral_utils.py** file).

In the 2nd cell of the notebook one can specify their directories, hyperparameters (sizes of the filters, optimizer) etc. All this information is stored in a dictionary named _args_ that is used throughout the rest of the code. In order to run the notebook in train or test mode, simply set:

```
args['mode'] = 'train' or 'test'
```

#### Some important notes:
* The code has compatibility with both _mpi-mesh_ and _trimesh_ packages (it can be chosen by setting the _meshpackage_ variable in the first cell of the notebook).
* The reference points parameter needs exactly one vertex index per disconnected component of the mesh. So for DFAUST you only need one, but for COMA which has the eyes as diconnected components, you need a reference point on the head as well as one on each eye.
* **spiral_utils.py**: In order to get the spiral ordering for each neighborhood, the spiraling code works by walking along the triangulation exploiting the fact that the triangles are all listed in a consistent way (either clockwise or counter-clockwise). These are saved as lists (their length depends on the number of hops and number of neighbors), which are then truncated or padded with -1 (index to a dummy vertex) to match all the spiral lengths to a predefined value L (in our case L = mean spiral length + 2 standard deviations of the spiral lengths). These are used by the _SpiralConv_ function in **models.py**, which is the main module of our proposed method.

# Cite

Please consider citing our work if you find it useful:

```
@InProceedings{bouritsas2019neural,
    author = {Bouritsas, Giorgos and Bokhnyak, Sergiy and Ploumpis, Stylianos and Bronstein, Michael and Zafeiriou, Stefanos},
    title = {Neural 3D Morphable Models: Spiral Convolutional Networks for 3D Shape Representation Learning and Generation},
    booktitle   = {The IEEE International Conference on Computer Vision (ICCV)},
    year        = {2019}
}
```




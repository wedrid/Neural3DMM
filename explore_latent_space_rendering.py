import latent_to_rendering as ltr
import numpy as np

def explore_coordinate_directions(latents, model, shapedata_mean, shapedata_std, amplitude = 30):
    #finds barycenter of points and starts exploration
    baricentro = np.mean(latents, axis = 0)
    print(baricentro)
    
    for i in range(latents.shape[1]):
        id = np.eye(1, latents.shape[1], i)[0]
        ltr.decode_latent_segment(baricentro, baricentro+(id * amplitude), 3, model, shapedata_mean, shapedata_std, additional_path=f"basis_exploration/dir_{i}/")

def main():
    model, shapedata_mean, shapedata_std = ltr.init(dict_path="./TMP/dict_path.json", checkpoint="checkpoint290")
    print(shapedata_mean.shape)
    
    print("LATENT: ")
    latents = np.load("mylatents.npy")
    print(latents.shape)

    explore_coordinate_directions(latents, model, shapedata_mean, shapedata_std)
    #ltr.decode_and_render_latent(latents[0], model, shapedata_mean, shapedata_std)



    return 

if __name__ == '__main__':
    main()
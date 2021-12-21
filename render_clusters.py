import numpy as np
import latent_to_rendering
from pathlib import Path

#python3 render_clusters.py --dict ./TMP/dict_path.json --checkpoint_file checkpoint290

def main():
    labels = np.load('./clusters_labels/Kmeans.npy')
    latents = np.load('mylatents.npy')

    clusters = []

    for i in range(np.max(labels)+1):
        clusters.append([])
    
    i = 0
    for item in labels:
        if item >= 0 and len(clusters[item]) < 200:
            clusters[item].append(i)
        i+=1
    
    #da ottimizzare
    model, mean, std = latent_to_rendering.init()
    for i, cluster in enumerate(clusters): #clusters has 10 elements for each cluster, containing indices
        temp = []
        for index in cluster:
            temp.append(latents[index])
        temp = np.array(temp)
        print(temp.shape)
        directory = f"./cluster_renderings_kmeans/cluster_{i}/"
        Path(directory).mkdir(parents=True, exist_ok=True)
        latent_to_rendering.decode_and_save_latent(temp, model, mean, std, directory)

    

if __name__ == '__main__':
    lc = main()
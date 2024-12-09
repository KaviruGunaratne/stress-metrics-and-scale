# Import necessary libraries
import tqdm 
import warnings

import numpy as np

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import MDS, TSNE 
from umap import UMAP

from scipy.spatial.distance import pdist,squareform


class DimensionReducer():
    """
    Class for reducing the dimensionality of a dataset using various techniques.
    """
    def __init__(self, X, labels):
        """
        Initialize the DimensionReducer with a dataset and labels.
        The dataset is scaled to [0, 1] range.
        """
        self.X = MinMaxScaler().fit_transform(X)
        self.labels = labels
        self.D = squareform(pdist(self.X))

    def compute_MDS(self):
        """
        Compute MDS (Multidimensional Scaling) on the dataset.
        """
        Y = MDS(dissimilarity="precomputed").fit_transform(self.D)
        return Y

    def compute_TSNE(self):
        """
        Compute t-SNE (t-Distributed Stochastic Neighbor Embedding) on the dataset.
        """
        Y = TSNE(metric='precomputed', init='random').fit_transform(self.D)
        return Y

    def compute_UMAP(self):
        """
        Compute UMAP (Uniform Manifold Approximation and Projection) on the dataset.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            Y = UMAP(metric='precomputed',init='random').fit_transform(self.D)
            return Y

    def compute_random(self):
        """
        Generate a random 2D embedding of the dataset.
        """
        Y = np.random.uniform(0, 1, (self.X.shape[0], 2))
        return Y
    

def save_embeddings(data, name, i, folder):
    """
    Save the embeddings computed by various techniques into a specified folder.
    """
    DR = DimensionReducer(*data)
    computations = ['MDS', 'TSNE', 'UMAP', 'random']
    methods = [DR.compute_MDS, DR.compute_TSNE, DR.compute_UMAP, DR.compute_random]

    for comp, method in zip(computations, methods):
        result = method()
        np.save(f"{folder}/{name}_{i}_{comp.lower()}.npy", result)
        if folder == 'big_data_embeddings':
            np.save(f"{folder}/{name}_{i}.npy", data[0])


def sample_down(X,num_samples):
    """
    Sample down datasets > 5000 to 5000 samples.
    """
    return X[np.random.choice(X.shape[0], num_samples)]


if __name__ == "__main__":
    """
    Main function to compute and save embeddings for all datasets in the 'datasets' directory.
    """
    import os
    if not os.path.isdir("embeddings"):
        os.mkdir('embeddings')

    datasets = os.listdir('datasets')
    
    num_iter = 10
    size_limit = 5000
    with tqdm.tqdm(total=len(datasets) * num_iter) as pbar:

        for datasetStr in datasets:
            X = np.load(f"datasets/{datasetStr}")
            if X.shape[0] > size_limit:
                X = sample_down(X,size_limit)
                os.remove(f"datasets/{datasetStr}")
                np.save(f"datasets/{datasetStr}", X)
                print(f"I have sampled {datasetStr} down to {size_limit} elements.")

            DR = DimensionReducer(X,None)
            dname = datasetStr.replace(".npy", "")

            for i in range(num_iter):
                mds = DR.compute_MDS()
                np.save(f"embeddings/{dname}_MDS_{i}.npy",mds)

                tsne = DR.compute_TSNE()
                np.save(f"embeddings/{dname}_TSNE_{i}.npy",tsne)

                umap = DR.compute_UMAP()
                np.save(f"embeddings/{dname}_UMAP_{i}.npy",umap)

                random = DR.compute_random()
                np.save(f"embeddings/{dname}_RANDOM_{i}.npy",random)                                    
        
                pbar.update(1)
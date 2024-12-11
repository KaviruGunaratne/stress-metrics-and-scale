# Import necessary libraries
import numpy as np
import json
import os
import tqdm
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

from metrics import Metrics
from metrics import MACHINE_EPSILON


def compute_all_metrics():
    """
    Function to compute all metrics for each dataset in the 'datasets' directory.
    The results are saved in a JSON file.
    """
    results = dict()
    datasets = os.listdir('datasets')


    with tqdm.tqdm(total=len(datasets) * 10 * 4) as pbar:

        for datasetStr in datasets:
            datasetName = datasetStr.replace(".npy", "")
            if "fashion_mnist" in datasetStr:
                datasetName = "fashion_mnist"
            # Load and scale dataset
            X = np.load(f"datasets/{datasetName}.npy")

            print(f"Dataset: {datasetName}, size: {X.shape}")
            # Compute the metrics for each technique and dataset pair
            for i in range(10):
                datasetResults = dict()
                for alg in ["MDS", "TSNE", 'UMAP', "RANDOM"]:
                    Y = np.load(f"embeddings/{datasetName}_{alg}_{i}.npy")
                    Y *= 10

                    M = Metrics(X, Y)

                    # Compute and store each metric
                    datasetResults[f'{alg}_raw'] = M.compute_raw_stress()
                    datasetResults[f'{alg}_norm'] = M.compute_normalized_stress()
                    datasetResults[f'{alg}_scalenorm'] = M.compute_scale_normalized_stress()
                    datasetResults[f'{alg}_kruskal'] = M.compute_kruskal_stress()
                    datasetResults[f'{alg}_sheppard'] = M.compute_shepard_correlation()

                    pbar.update(1)

                datasetResults = {key: float(val)
                                  for key, val in datasetResults.items()}
                results[f'{datasetName}_{i}'] = datasetResults

            with open("out10x.json", 'w') as fdata:
                json.dump(results, fdata, indent=4)


def test_curve():
    import pylab as plt
    results = dict()

    with tqdm.tqdm(total=len(os.listdir('datasets')) * 4) as pbar:

        for datasetStr in os.listdir("datasets"):
            datasetName = datasetStr.replace(".npy", "")
            if "fashion_mnist" in datasetStr:
                datasetName = "fashion_mnist"
            X = np.load(f"datasets/{datasetName}.npy")

            print(f"Dataset: {datasetName}, size: {X.shape}")
            datasetResults = dict()
            fig, ax = plt.subplots()
            for alg in ["MDS", "TSNE", 'UMAP', "RANDOM"]:
                Y = np.load(f"embeddings/{datasetName}_{alg}_0.npy")

                M = Metrics(X, Y)

                rrange = np.linspace(0, 10, 100)
                stresses = [M.compute_normalized_stress(
                    alpha=a) for a in rrange]

                ax.plot(rrange, stresses, label=alg)

                # datasetResults[f'{alg}_norm'] = M.compute_normalized_stress()

                pbar.update(1)

            ax.legend()
            fig.savefig(f"test-figures/{datasetName}.png")
            plt.close(fig)

            # with open("out10x.json", 'w') as fdata:
            #     json.dump(results,fdata,indent=4)


def graph_kl(scales):

    target_dir = 'test-kl-figures'

    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    datasets = os.listdir('datasets')
    # datasets = ['orl.npy', 'har.npy', 'coil20.npy', 'cnae9.npy']
    algorithms = ["RANDOM", "MDS", "UMAP", "TSNE"]
    n_runs = 10
    

    with tqdm.tqdm(total=len(datasets) * len(algorithms) * n_runs) as pbar:

        for datasetStr in datasets:
            datasetName = datasetStr.replace(".npy", "")
            if "fashion_mnist" in datasetStr:
                datasetName = "fashion_mnist"
            X = np.load(f"datasets/{datasetName}.npy")

            for n in range(n_runs):
                pbar.set_postfix_str(f"Dataset={datasetName} (shape={X.shape}), Run={n}")

                fig = plt.figure(figsize=(20, 10))
                gs = GridSpec(2, 4, figure=fig)
                graph_ax = fig.add_subplot(gs[0, :])
                
                for i, alg in enumerate(algorithms):

                    Y = np.load(f"embeddings/{datasetName}_{alg}_{n}.npy")

                    kl_divergences = __compute_kl_divergences_in_chunks(X, Y, scales, perplexity=30)

                    graph_ax.plot(scales, kl_divergences, label=alg)
                    
                    alg_ax = fig.add_subplot(gs[1, i])
                    alg_ax.scatter(Y[:, 0], Y[:, 1], color='black', s=5)
                    alg_ax.set_title(f"{alg} Embedding")

                    pbar.update(1)

                graph_ax.set_title("KL Divergence w.r.t. Scale")
                graph_ax.set_xlabel("Scaling Factor")
                graph_ax.set_ylabel("KL Divergence")
                graph_ax.legend()

                fig.suptitle(f'Variation of KL Divergence with scale for {datasetName} Dataset', fontweight='bold')
                # plt.tight_layout()

                fig.savefig(f"{target_dir}/{datasetName}_{n}.png")
                plt.close(fig)


    # Add labels, title, and legend
    plt.legend()

    # Show grid
    plt.grid(True)

def __estimate_needed_memory(X, scales):
    n_samples, n_dims = X.shape
    n_scales = scales.shape[0]

    return max(MACHINE_EPSILON, (0.03 * n_samples - 5) * n_scales + 0.0007 * n_samples * n_dims + 0.018 * n_samples ** 2)

def __compute_kl_divergences_in_chunks(X, Y, scales, perplexity):
    from psutil import virtual_memory

    needed_memory = __estimate_needed_memory(X, scales)
    usable_memory = virtual_memory().available / (1024 ** 2) * 0.7

    kl_divergences = np.empty(0)

    scale_ranges = np.array_split(scales, (needed_memory // usable_memory) + 1)
    for scale_range in scale_ranges:

        M = Metrics(X, Y, scaling_factors=scale_range)
        kl_range = M.compute_kl_divergences(perplexity=perplexity)
        kl_divergences = np.append(kl_divergences, kl_range)
    
    return kl_divergences



if __name__ == "__main__":
    compute_all_metrics()
    # test_curve()
    graph_kl(scales=np.linspace(MACHINE_EPSILON, 20, 100))
# Import necessary libraries
import numpy as np
import json
import os
import tqdm

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
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    if not os.path.isdir("test-kl-figures"):
        os.mkdir('test-kl-figures')

    datasets = os.listdir('datasets')
    # datasets = ['orl.npy', 'har.npy', 'coil20.npy', 'cnae9.npy']
    


    with tqdm.tqdm(total=len(datasets) * 4) as pbar:

        for datasetStr in datasets:
            datasetName = datasetStr.replace(".npy", "")
            if "fashion_mnist" in datasetStr:
                datasetName = "fashion_mnist"
            X = np.load(f"datasets/{datasetName}.npy")

            print(f"Dataset: {datasetName}, size: {X.shape}")

            fig = plt.figure(figsize=(20, 10))
            gs = GridSpec(2, 4, figure=fig)
            graph_ax = fig.add_subplot(gs[0, :])
            
            for i, alg in enumerate(["RANDOM", "MDS", "UMAP", "TSNE"]):

                Y = np.load(f"embeddings/{datasetName}_{alg}_0.npy")

                M = Metrics(X, Y, scaling_factors=scales)
                kl_divergences = M.compute_kl_divergences(perplexity=30)

                graph_ax.plot(scales, kl_divergences, label=alg)
                
                alg_ax = fig.add_subplot(gs[1, i])
                alg_ax.scatter(Y[:, 0], Y[:, 1], color='black', s=2)
                alg_ax.set_title(f"{alg} Embedding")

                pbar.update(1)

            graph_ax.set_title("KL Divergence w.r.t. Scale")
            graph_ax.set_xlabel("Scaling Factor")
            graph_ax.set_ylabel("KL Divergence")
            graph_ax.legend()

            fig.suptitle(f'Variation of KL Divergence with scale for {datasetName} Dataset', fontweight='bold')
            # plt.tight_layout()

            fig.savefig(f"test-kl-figures/{datasetName}.png")
            plt.close(fig)


    # Add labels, title, and legend
    plt.legend()

    # Show grid
    plt.grid(True)


if __name__ == "__main__":
    compute_all_metrics()
    # test_curve()
    graph_kl(scales=np.linspace(MACHINE_EPSILON, 20, 100))
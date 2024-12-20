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


def graph_kl(scales, target_dir, min_kl_data_file, n_runs=10):
    """
    Plot KL Divergence vs. scale graphs for each group of embeddings in embeddings/
    Run log_min_kl first in order to draw the minimum points of the graphs.
    """
   

    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    datasets = os.listdir('datasets')
    # datasets = ['orl.npy', 'har.npy', 'coil20.npy', 'cnae9.npy']
    algorithms = ["RANDOM", "MDS", "UMAP", "TSNE"]

    # Load minimum point data
    min_kls = pd.read_csv(f'{target_dir}/{min_kl_data_file}', index_col=[0, 1, 2])
    

    with tqdm.tqdm(total=len(datasets) * len(algorithms) * n_runs) as pbar:

        for datasetStr in datasets:
            datasetName = datasetStr.replace(".npy", "")
            if "fashion_mnist" in datasetStr:
                datasetName = "fashion_mnist"
            X = np.load(f"datasets/{datasetName}.npy")

            for n in range(n_runs):
                pbar.set_postfix_str(f"Dataset={datasetName} (shape={X.shape}), Run={n}")

                fig = plt.figure(figsize=(20, 15))
                gs = GridSpec(3, 4, figure=fig)
                graph_ax = fig.add_subplot(gs[0:2, :])
                
                for i, alg in enumerate(algorithms):
                    Y = np.load(f"embeddings/{datasetName}_{alg}_{n}.npy")

                    kl_divergences = _compute_kl_divergences_in_chunks(X, Y, scales, perplexity=30)

                    # Plot kl vs. scale graph for alg
                    graph_ax.plot(scales, kl_divergences, label=alg)
                    # Plot minimum point
                    min_x = min_kls.loc[datasetName, f'Run {n}', 'x'][alg]
                    min_y = min_kls.loc[datasetName, f'Run {n}', 'y'][alg]
                    if min_x < scales[-1]:
                        graph_ax.scatter(min_x, min_y, marker='X', label=f"{alg} Minimum")
                    
                    # Plot embedding
                    alg_ax = fig.add_subplot(gs[2, i])
                    alg_ax.scatter(Y[:, 0], Y[:, 1], color='black', s=10, alpha=0.2)
                    alg_ax.set_title(f"{alg} Embedding")

                    pbar.update(1)

                graph_ax.set_title("KL Divergence w.r.t. Scale", fontdict={'fontsize': 22})
                graph_ax.set_xlabel("Scaling Factor", fontdict={'fontsize': 17})
                graph_ax.set_ylabel("KL Divergence", fontdict={'fontsize': 17})
                graph_ax.legend()
                graph_ax.grid(True)

                fig.subplots_adjust(hspace=0.5)
                fig.suptitle(f'Variation of KL Divergence with scale for {datasetName.capitalize()} Dataset', fontweight='bold')
                # plt.tight_layout()

                fig.savefig(f"{target_dir}/{datasetName}_{n}.png")
                plt.close(fig)



def __estimate_needed_memory(X, scales):
    n_samples, n_dims = X.shape
    n_scales = scales.shape[0]

    return max(MACHINE_EPSILON, (0.018 * n_samples - 3) * n_scales + 0.0002 * n_samples * n_dims + 0.013 * n_samples ** 2)

def _compute_kl_divergences_in_chunks(X, Y, scales, perplexity):
    from psutil import virtual_memory

    needed_memory = __estimate_needed_memory(X, scales)
    usable_memory = virtual_memory().available / (1024 ** 2) * 0.6

    kl_divergences = np.empty(0)
    scale_ranges = np.array_split(scales, (needed_memory // usable_memory) + 1)
    for scale_range in scale_ranges:

        M = Metrics(X, Y, scaling_factors=scale_range)
        kl_range = M.compute_kl_divergences(perplexity=perplexity)
        kl_divergences = np.append(kl_divergences, kl_range)
    
    return kl_divergences

def log_min_kl(perplexity, target_dir, target_csv_file, n_runs=10):
    """
    Calculate and log the scale at which KL Divergence is minimized and the corresponding KL Divergence for every single embedding.
    The data is stored as a CSV file in target_dir/target_csv_file.
    To load the CSV file into a Pandas.DataFrame, execute the following line: \\
    `pd.read_csv(f'{target_dir}/{target_csv_file}', index_col=[0, 1, 2])`
    """

    # Create target_dir if not exists
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    datasets = os.listdir('datasets')
    # datasets = ['epileptic.npy']
    algorithms = ["RANDOM", "MDS", "UMAP", "TSNE"]
    n_runs = 5

    # Load DataFrame if CSV file exists, else initialize new DataFrame
    if os.path.exists(f"{target_dir}/{target_csv_file}"):
        min_kls = pd.read_csv(f'{target_dir}/{target_csv_file}', index_col=[0, 1, 2])
    else:
        index = pd.MultiIndex.from_product(
            [[datasetStr.replace(".npy", "") for datasetStr in datasets],
            [f'Run {i}' for i in range(n_runs)],
            ['x', 'y']],
            names=["Dataset", "Run", "Coord"]
        )

        min_kls = pd.DataFrame(index=index, columns=algorithms)
    min_kls = min_kls.sort_index()
    

    # Record min KL 
    with tqdm.tqdm(total=(len(datasets) * n_runs * len(algorithms))) as pbar:

        for datasetStr in datasets:
            datasetName = datasetStr.replace(".npy", "")
            if "fashion_mnist" in datasetStr:
                datasetName = "fashion_mnist"

            X = np.load(f"datasets/{datasetName}.npy")

            for n in range(n_runs):
                # Skip if data has already been filled
                if min_kls.loc[datasetName, f"Run {n}"].notna().all(axis=None):
                    print(f"Entries for {datasetName}, Run {n} have already been computed. Skipped.")
                    pbar.update(4)
                    continue

                pbar.set_postfix_str(f"Dataset={datasetName}, Run={n}")

                for alg in algorithms:
                    try:
                        Y = np.load(f"embeddings/{datasetName}_{alg}_{n}.npy")
                    except FileNotFoundError:
                        print(f"embeddings/{datasetName}_{alg}_{n}.npy does not exist.")
                        print(f"Run {n} is skipped.")
                        break
                    coords = calculate_min_kl(X, Y, perplexity)
                    min_kls.loc[(datasetName, f'Run {n}', 'x'), alg] = coords[0]
                    min_kls.loc[(datasetName, f'Run {n}', 'y'), alg] = coords[1]

                    pbar.update(1)
                min_kls.to_csv(f"{target_dir}/{target_csv_file}")


    min_kls.to_csv(f"{target_dir}/{target_csv_file}")
    print(min_kls)


def calculate_min_kl(X, Y, perplexity):
    """
    Use minimize_scalar in Scipy.optimize to find the inimizing coordinates of KL Divergence w.r.t. scale.
    """
    def get_kl(scale):
        M = Metrics(X, Y * scale, scaling_factors=np.empty(0))
        return M.compute_kl_divergence(perplexity=perplexity)
    
    res = minimize_scalar(get_kl, bounds=(0, 300))
    return (res.x, res.fun)

def log_normalized_kl(perplexity, target_dir, target_csv_file, n_runs=10):
    # Create target_dir if not exists
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    datasets = os.listdir('datasets')
    algorithms = ["RANDOM", "MDS", "UMAP", "TSNE"]

    # Load DataFrame if CSV file exists, else initialize new DataFrame
    if os.path.exists(f"{target_dir}/{target_csv_file}"):
        zadu_kls = pd.read_csv(f'{target_dir}/{target_csv_file}', index_col=[0, 1, 2])
    else:
        index = pd.MultiIndex.from_product(
            [
                [datasetStr.replace(".npy", "") for datasetStr in datasets],
                [f'Run {i}' for i in range(n_runs)],
                ['x', 'y']
            ],
            names=["Dataset", "Run", "Coord"]
        )

        zadu_kls = pd.DataFrame(index=index, columns=algorithms)
        zadu_kls.to_csv(f"{target_dir}/{target_csv_file}")
        
    zadu_kls = zadu_kls.sort_index()
    

    # Record KL after normalizing 
    with tqdm.tqdm(total=(len(datasets) * (n_runs) * len(algorithms))) as pbar:

        for datasetStr in datasets:
            datasetName = datasetStr.replace(".npy", "")
            if "fashion_mnist" in datasetStr:
                datasetName = "fashion_mnist"

            X = np.load(f"datasets/{datasetName}.npy")
            dX = pairwise_distances(X)
            dX /= np.max(dX)

            for n in range(n_runs):
                # Skip if data has already been filled
                if zadu_kls.loc[datasetName, f"Run {n}"].notna().all(axis=None):
                    print(f"Entries for {datasetName}, Run {n} have already been computed. Skipped.")
                    pbar.update(4)
                    continue

                pbar.set_postfix_str(f"Dataset={datasetName}, Run={n}")

                for alg in algorithms:
                    try:
                        Y = np.load(f"embeddings/{datasetName}_{alg}_{n}.npy")
                        dY = pairwise_distances(Y)
                        
                        normalizing_factor = 1/ np.max(dY)
                        dY *= normalizing_factor
                    except FileNotFoundError:
                        print(f"embeddings/{datasetName}_{alg}_{n}.npy does not exist.")
                        print(f"Run {n} is skipped.")
                        break

                    M = Metrics(dX, dY, precomputed=True, setbatch=False)
                    zadu_kls.loc[(datasetName, f'Run {n}', 'x'), alg] = normalizing_factor
                    zadu_kls.loc[(datasetName, f'Run {n}', 'y'), alg] = M.compute_kl_divergence(perplexity=perplexity)
                    
                    pbar.update(1)
                zadu_kls.to_csv(f"{target_dir}/{target_csv_file}")


    zadu_kls.to_csv(f"{target_dir}/{target_csv_file}")
                    


if __name__ == "__main__":
    compute_all_metrics()
    # test_curve()
    target_dir = 'test-kl-figures'
    min_kl_csv_file = 'min_kls_new.csv'
    log_min_kl(perplexity=30, n_runs=7, target_dir=target_dir, target_csv_file=min_kl_csv_file)
    graph_kl(scales=np.linspace(0, 15, 250), target_dir=target_dir, min_kl_data_file=min_kl_csv_file, n_runs=10)
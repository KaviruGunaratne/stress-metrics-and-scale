# Import necessary libraries
import numpy as np
import json
import os
import tqdm
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from sklearn.metrics import pairwise_distances

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


def graph_kl(scales, target_dir, n_runs=10, drop_UMAP=False, plot_min_kl=True, min_kl_data_filepath=None, plot_normalized_kl=False, normalized_kl_data_filepath=None):
    """
    Plot KL Divergence vs. scale graphs for each group of embeddings in embeddings/
    If plotting minimum points (and points of KL Divergence where the embedding was normalized), log_min_kl (and log_normalized_kl) should be run first.


    Parameters
    ----------

    scales : 1D array
        The factors by which the embeddings are scaled before computing KL divergence

    target_dir : string
        The directory where the plotted graphs should be stored
    
    plot_min_kl : bool
        If True, the minimum points are plotted
        min_kl_data

    n_runs : int
        Number of sets of embeddings per dataset considered

    drop_UMAP : bool
        If True, only graphs of t-SNE, MDS, and Random embeddings are plotted
        If False, graphs of UMAP embeddings are also plotted

    min_kl_data_filepath : string
        Path of CSV file in target_dir from where minimum points are accessed

    plot_normalized_kl : bool
        If True, coordinates of embedding-normalized KL divergence are plotted

    normalized_kl_data_filepath : string
        Path of CSV file in target_dir from where coordinates of embedding-normalized KL divergences are plotted
    """

    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    if drop_UMAP:
        algorithms = ['RANDOM', 'MDS', 'TSNE']
    else:
        algorithms = ["RANDOM", "MDS", "UMAP", "TSNE"]

    # Load minimum point data
    if plot_min_kl:
        min_kls = pd.read_csv(min_kl_data_filepath, index_col=[0, 1, 2])

    # Load ZADU KL coordinates
    if plot_normalized_kl:
        normalized_kls = pd.read_csv(normalized_kl_data_filepath, index_col=[0, 1, 2])

    # Colors of graphs for each algorithm
    alg_color = {
        'TSNE' : 'darkblue',
        'UMAP' : 'purple',
        'MDS' : 'darkred',
        'RANDOM' : 'darkgreen',
    }

    alg_name = {
        'TSNE' : 't-SNE',
        'UMAP' : 'UMAP',
        'MDS' : 'MDS',
        'RANDOM' : 'Random',
    }
    

    with tqdm.tqdm(total=len(datasets) * len(algorithms) * n_runs) as pbar:

        for datasetStr in datasets:
            datasetName = datasetStr.replace(".npy", "")
            if "fashion_mnist" in datasetStr:
                datasetName = "fashion_mnist"
            X = np.load(f"datasets/{datasetName}.npy")
            labels = np.load(f"dataset_labels/{datasetName}.npy")
            
            # Larger scale for large datasets
            if datasetName in ['epileptic', 'spambase', 's-curve', 'swissroll']:
                orig_scales = scales
                scales = np.linspace(0, 250, 300)

            for n in range(n_runs):
                pbar.set_postfix_str(f"Dataset={datasetName} (shape={X.shape}), Run={n}")

                if drop_UMAP:
                    fig = plt.figure(figsize=(45, 45))
                    gs = GridSpec(3, 3, figure=fig)
                else:
                    fig = plt.figure(figsize=(60, 45))
                    gs = GridSpec(3, 4, figure=fig)
                graph_ax = fig.add_subplot(gs[0:2, :])
                
                for i, alg in enumerate(algorithms):
                    Y = np.load(f"embeddings/{datasetName}_{alg}_{n}.npy")

                    kl_divergences = _compute_kl_divergences_in_chunks(X, Y, scales, perplexity=30)

                    # Plot kl vs. scale graph for alg
                    graph_ax.plot(scales, kl_divergences, label=alg_name[alg], c=alg_color[alg])
                    
                    # Plot minimum point
                    if plot_min_kl:
                        min_x = min_kls.loc[datasetName, f'Run {n}', 'x'][alg]
                        min_y = min_kls.loc[datasetName, f'Run {n}', 'y'][alg]
                        if min_x < scales[-1]:
                            graph_ax.scatter(min_x, min_y, marker='o', label=f"Minimum", c=alg_color[alg])
                    
                    # Plot point where embedding was normalized
                    if plot_normalized_kl:
                        normalized_x = normalized_kls.loc[datasetName, f'Run {n}', 'x'][alg]
                        normalized_y = normalized_kls.loc[datasetName, f'Run {n}', 'y'][alg]
                        if normalized_x < scales[-1]:
                            graph_ax.scatter(normalized_x, normalized_y, marker='o', label=f"{alg} normalized", c=alg_color[alg])
                    
                    # Plot embedding
                    alg_ax = fig.add_subplot(gs[2, i])
                    alg_ax.scatter(Y[:, 0], Y[:, 1], s=30, alpha=0.7, c=labels, cmap='tab20', edgecolors='black')
                    alg_ax.set_title(f"{datasetName.capitalize} dataset embedded with {alg_name[alg]}", fontdict={'fontsize': 80})
                    alg_ax.set_xticks([])
                    alg_ax.set_yticks([])

                    pbar.update(1)

                graph_ax.set_title(datasetName.capitalize(), fontdict={'fontsize': 120})
                graph_ax.set_xlabel("Scale value", fontdict={'fontsize': 100})
                graph_ax.set_ylabel("KL Divergence", fontdict={'fontsize': 100})
                graph_ax.legend(fontsize=40)
                graph_ax.tick_params(axis='x', labelsize=80)
                graph_ax.tick_params(axis='y', labelsize=80)
                # graph_ax.grid(True)

                fig.subplots_adjust(hspace=0.5)
                fig.suptitle(f'Variation of KL Divergence with scale for {datasetName.capitalize()} Dataset', fontweight='bold', fontsize=120)
                # plt.tight_layout()

                fig.savefig(f"{target_dir}/{datasetName}_{n}.png")
                plt.close(fig)

            # Reset scale to original
            if datasetName in ['epileptic', 'spambase', 's-curve', 'swissroll']:
                scales = orig_scales


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


def calculate_min_kl(X, Y, perplexity, y_similarity='t'):
    """
    Use minimize_scalar in Scipy.optimize to find the inimizing coordinates of KL Divergence w.r.t. scale.
    """
    def get_kl(scale):
        M = Metrics(X, Y * scale, scaling_factors=np.empty(0))
        return M.compute_kl_divergence(perplexity=perplexity, y_similarity=y_similarity)
    
    res = minimize_scalar(get_kl, bounds=(0, 300))
    return (res.x, res.fun)
          
  
def fill_dataframe(df: pd.DataFrame, target_dir, target_csv_file, datasets, algorithms, n_runs : int, func):
    """
    Fills DataFrame df with values according to function func while looping through each embedding in embeddings/
    
    Parameters
    ----------

    target_dir : string
        Directory where the data is to be stored

    target_csv_file : string
        CSV file to write the data of the DataFrame df is written

    df : pd.DataFrame
        DataFrame to which the data is being written to

    n_runs : int
        Number of runs for which embeddings have been calculated per dataset
        
    func : function
        Function that decides how to fill df. \\
        Should return a tuple (index_labels, values). \\
        index_labels and values are equal length iterables where index_labels contains the index label
        at which the DataFrame is to be filled, and values contains the values that should be filled at the location specified by index_labels
        
    func_kwargs: dict
        Dictionary with keyword arguments of func
    """



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
                
                # Add current Dataset and run to progress bar
                pbar.set_postfix_str(f"Dataset={datasetName}, Run={n}")

                for alg in algorithms:
                    try:
                        Y = np.load(f"embeddings/{datasetName}_{alg}_{n}.npy")
                    except FileNotFoundError:
                        print(f"embeddings/{datasetName}_{alg}_{n}.npy does not exist.")
                        print(f"Run {n} is skipped.")
                        break

                    index_labels, values = func(X=X, Y=Y, n=n, datasetName=datasetName)
                    for index_label, value in zip(index_labels, values):
                        df.loc[index_label, alg] = value

                    pbar.update(1)
                df.to_csv(f"{target_dir}/{target_csv_file}")

        df.to_csv(f"{target_dir}/{target_csv_file}")


def log_min_kl(perplexity, target_dir, target_csv_file, n_runs=10):
    # Create target_dir if not exists
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    datasets = os.listdir('datasets')
    # datasets = ['iris.npy', 'wine.npy', 'orl.npy', 'auto-mpg.npy']
    algorithms = ["RANDOM", "MDS", "UMAP", "TSNE"]

    # Load DataFrame if CSV file exists, else initialize new DataFrame
    if os.path.exists(f"{target_dir}/{target_csv_file}"):
        min_kls = pd.read_csv(f'{target_dir}/{target_csv_file}', index_col=[0, 1, 2])
    else:
        index = pd.MultiIndex.from_product(
            [
                [datasetStr.replace(".npy", "") for datasetStr in datasets],
                [f'Run {i}' for i in range(n_runs)],
                ['x', 'y']
            ],
            names=["Dataset", "Run", "Coord"]
        )

        min_kls = pd.DataFrame(index=index, columns=algorithms)
        min_kls.to_csv(f"{target_dir}/{target_csv_file}")
    # Sort DataFrame index
    min_kls = min_kls.sort_index()

    # Define function to fill min_kls with fill_dataframe
    def get_min_kls(X, Y, n, datasetName, perplexity=perplexity):
        coords = calculate_min_kl(X, Y, perplexity)
        index_labels = [(datasetName, f'Run {n}', 'x'), (datasetName, f'Run {n}', 'y')]

        return index_labels, coords

    # Fill min_kls
    fill_dataframe(df=min_kls, target_dir=target_dir, target_csv_file=target_csv_file, datasets=datasets, algorithms=algorithms, n_runs=n_runs, func=get_min_kls)

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

        normalized_kls = pd.DataFrame(index=index, columns=algorithms)
        normalized_kls.to_csv(f"{target_dir}/{target_csv_file}")
        
    normalized_kls = normalized_kls.sort_index()

    # Define function to fill normalized_kls with fill_dataframe
    def get_normalized_kls(X, Y, n, datasetName, perplexity=perplexity):
        dX = pairwise_distances(X)
        dY = pairwise_distances(Y)
        normalizing_factor = 1 / np.max(dY)
        dY *= normalizing_factor

        M = Metrics(dX, dY, precomputed=True, setbatch=False)
        
        kl = M.compute_kl_divergence(perplexity=perplexity)

        coords = (normalizing_factor, kl)
        index_labels = [(datasetName, f'Run {n}', 'x'), (datasetName, f'Run {n}', 'y')]

        return index_labels, coords
        
    fill_dataframe(df=normalized_kls, target_dir=target_dir, target_csv_file=target_csv_file, datasets=datasets, algorithms=algorithms, n_runs=n_runs, func=get_normalized_kls)

def log_zadu_kls(perplexity, target_dir, target_csv_file, n_runs=10):
    # Create target_dir if not exists
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    datasets = os.listdir('datasets')
    algorithms = ["RANDOM", "MDS", "UMAP", "TSNE"]

    # Load DataFrame if CSV file exists, else initialize new DataFrame
    if os.path.exists(f"{target_dir}/{target_csv_file}"):
        zadu_kls = pd.read_csv(f'{target_dir}/{target_csv_file}', index_col=[0, 1])
    else:
        index = pd.MultiIndex.from_product(
            [
                [datasetStr.replace(".npy", "") for datasetStr in datasets],
                [f'Run {i}' for i in range(n_runs)]
            ],
            names=["Dataset", "Run"]
        )

        zadu_kls = pd.DataFrame(index=index, columns=algorithms)
        zadu_kls.to_csv(f"{target_dir}/{target_csv_file}")
        
    zadu_kls = zadu_kls.sort_index()

    # Define function to fill zadu_kls with fill_dataframe
    def get_zadu_kls(X, Y, n, datasetName, perplexity=perplexity):
        dX = pairwise_distances(X)
        dY = pairwise_distances(Y)
        dX /= np.max(dX)
        dY /= np.max(dY)

        M = Metrics(dX, dY, precomputed=True, setbatch=False)
        
        kl = M.compute_kl_divergence(perplexity=perplexity)

        index_labels = [(datasetName, f'Run {n}')]

        return index_labels, (kl,)
        
    fill_dataframe(df=zadu_kls, target_dir=target_dir, target_csv_file=target_csv_file, datasets=datasets, algorithms=algorithms, n_runs=n_runs, func=get_zadu_kls)

def log_kl_at_scale(perplexity, target_dir, target_csv_file, scales, n_runs=10):
    # Create target_dir if not exists
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    datasets = os.listdir('datasets')
    algorithms = ["RANDOM", "MDS", "UMAP", "TSNE"]

    # Load DataFrame if CSV file exists, else initialize new DataFrame
    if os.path.exists(f"{target_dir}/{target_csv_file}"):
        kls_at_scales = pd.read_csv(f'{target_dir}/{target_csv_file}', index_col=[0, 1, 2])
    else:
        index = pd.MultiIndex.from_product(
            [[datasetStr.replace(".npy", "") for datasetStr in datasets],
            [f'Run {i}' for i in range(n_runs)],
            scales],
            names=["Dataset", "Run", "Scale"]
        )

        kls_at_scales = pd.DataFrame(index=index, columns=algorithms)
    kls_at_scales = kls_at_scales.sort_index()

    def get_kls_at_scale(X, Y, n, datasetName, perplexity=perplexity):
        M = Metrics(X, Y, setbatch=True, precomputed=False, scaling_factors=scales)
        kls = M.compute_kl_divergences(perplexity=perplexity)

        index_labels = []
        for scale in scales:
            index_labels.append((datasetName, f'Run {n}', scale))

        return index_labels, kls
    
    fill_dataframe(kls_at_scales, target_dir, target_csv_file, datasets, algorithms, n_runs, func=get_kls_at_scale)

def log_kl_at_infty(perplexity, target_dir, target_csv_file, n_runs=10):
    # Create target_dir if not exists
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    datasets = os.listdir('datasets')
    algorithms = ["RANDOM", "MDS", "UMAP", "TSNE"]

    # Load DataFrame if CSV file exists, else initialize new DataFrame
    if os.path.exists(f"{target_dir}/{target_csv_file}"):
        kls_at_infty = pd.read_csv(f'{target_dir}/{target_csv_file}', index_col=[0, 1])
    else:
        index = pd.MultiIndex.from_product(
            [
                [datasetStr.replace(".npy", "") for datasetStr in datasets],
                [f'Run {i}' for i in range(n_runs)]
            ],
            names=["Dataset", "Run"]
        )

        kls_at_infty = pd.DataFrame(index=index, columns=algorithms)
        kls_at_infty.to_csv(f"{target_dir}/{target_csv_file}")
        
    kls_at_infty = kls_at_infty.sort_index()

    # Function to fill kls_at_infty using fill_dataframe
    def get_kl_at_infty(X, Y, n, datasetName, perplexity=perplexity):
        M = Metrics(X, Y, setbatch=False)

        kl = M.compute_kl_divergence_at_infty(perplexity=perplexity)
        index_labels = [(datasetName, f'Run {n}')]

        return index_labels, (kl,)
    
    # Fill kls_at_infty
    fill_dataframe(kls_at_infty, target_dir, target_csv_file, datasets, algorithms, n_runs, get_kl_at_infty)

def log_min_kl_normal(perplexity, target_dir, target_csv_file, n_runs=10):
    # Create target_dir if not exists
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    # datasets = os.listdir('datasets')
    datasets = ['iris.npy', 'wine.npy', 'orl.npy', 'auto-mpg.npy']
    algorithms = ["RANDOM", "MDS", "UMAP", "TSNE"]

    # Load DataFrame if CSV file exists, else initialize new DataFrame
    if os.path.exists(f"{target_dir}/{target_csv_file}"):
        min_kls = pd.read_csv(f'{target_dir}/{target_csv_file}', index_col=[0, 1, 2])
    else:
        index = pd.MultiIndex.from_product(
            [
                [datasetStr.replace(".npy", "") for datasetStr in datasets],
                [f'Run {i}' for i in range(n_runs)],
                ['x', 'y']
            ],
            names=["Dataset", "Run", "Coord"]
        )

        min_kls = pd.DataFrame(index=index, columns=algorithms)
        min_kls.to_csv(f"{target_dir}/{target_csv_file}")
    # Sort DataFrame index
    min_kls = min_kls.sort_index()

    # Define function to fill min_kls with fill_dataframe
    def get_min_kls(X, Y, n, datasetName, perplexity=perplexity):
        coords = calculate_min_kl(X, Y, perplexity, y_similarity='normal')
        index_labels = [(datasetName, f'Run {n}', 'x'), (datasetName, f'Run {n}', 'y')]

        return index_labels, coords

    # Fill min_kls
    fill_dataframe(df=min_kls, target_dir=target_dir, target_csv_file=target_csv_file, datasets=datasets, algorithms=algorithms, n_runs=n_runs, func=get_min_kls)



def draw_shepard_diagrams(perplexity, target_dir, n_runs=10):
    """
    Plots the Shepard diagram between the high-dimensional and low-dimensional probability values à la t-SNE corresponding to each embedding.
    The embeddings are considered at the default scale of 1.
    Saves the diagrams at target_dir.

    Parameters
    ----------
    perplexity : float
      Perplexity to calculate the conditional probability distribution corresponding to the SNE algorithm 

    target_dir : string
      The directory where the diagrams will be stored

    n_runs : int
      Number of groups of embeddings (from t-SNE, UMAP, MDS, and Random) for each dataset whose diagrams are drawn
    """

    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    datasets = os.listdir('datasets')
    algorithms = ["RANDOM", "MDS", "UMAP", "TSNE"]

    with tqdm.tqdm(total=len(datasets) * len(algorithms) * n_runs) as pbar:

        for datasetStr in datasets:
            datasetName = datasetStr.replace(".npy", "")
            if "fashion_mnist" in datasetStr:
                datasetName = "fashion_mnist"
            X = np.load(f"datasets/{datasetName}.npy")

            for n in range(n_runs):
                pbar.set_postfix_str(f"Dataset={datasetName} (shape={X.shape}), Run={n}")

                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16,16))

                for ax, alg in zip(axes.flatten(), algorithms):
                    Y = np.load(f"embeddings/{datasetName}_{alg}_{n}.npy")

                    M = Metrics(X, Y, setbatch=False, precomputed=False, scaling_factors=None)
                    conditional_P = M._conditional_probabilities(perplexity=perplexity)
                    P = ((conditional_P + conditional_P.T) / (2 * conditional_P.shape[0]))
                    Q = M._get_Q(is_batch=False)

                    ax.scatter(P, Q, s=5, alpha=1.)
                    ax.set_title(alg, fontweight='bold', fontsize='x-large')
                    ax.set_ylabel("Low-Dimensional Probabilities Q")
                    ax.set_xlabel("High-Dimensional Probabilities P")
                    ax.tick_params(axis='x', rotation=45)

                    pbar.update(1)

                fig.subplots_adjust(hspace=0.3, wspace=0.5)
                fig.suptitle(f'Shepard Diagrams of Probability Distributions for {datasetName.capitalize()} Dataset', fontweight='bold', fontsize='xx-large')
                # plt.tight_layout()

                fig.savefig(f"{target_dir}/Shepard_{datasetName}_{n}.png")
                plt.close(fig)

if __name__ == "__main__":
    compute_all_metrics()
    # test_curve()

    graph_dir = 'graphs'
    csv_dir = 'csv_files'

    min_kl_csv = 'min_kls.csv'
    zadu_kl_csv = 'zadu_kls.csv'
    y_normalized_kls_csv = 'y_normalized_kls.csv'
    kl_at_infty_csv = 'kl_at_infty.csv'
    scales_kl_csv = 'kl_at_1_and_10.csv'
    scales_to_log = np.array([1, 10])
    normal_min_kl_csv = 'normal_min_kls.csv'

    log_kl_at_infty(perplexity=30, target_dir=csv_dir, target_csv_file=kl_at_infty_csv, n_runs=10)
    log_zadu_kls(perplexity=30, target_dir=csv_dir, target_csv_file=zadu_kl_csv, n_runs=10)
    log_normalized_kls(perplexity=30, target_dir=csv_dir, target_csv_file=y_normalized_kls_csv, n_runs=10)
    log_min_kl(perplexity=30, target_dir=csv_dir, target_csv_file=min_kl_csv, n_runs=10)
    log_kl_at_scale(perplexity=30, target_dir=csv_dir, target_csv_file=scales_kl_csv, scales=scales_to_log, n_runs=10)
    # log_min_kl_normal(perplexity=30, target_dir=csv_dir, target_csv_file=normal_min_kl_csv, n_runs=10)


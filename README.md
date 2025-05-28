# stress-metrics-and-scale

Repository accompanying the paper, "Normalized Stress is Not Normalized: How to Interpret Stress Correctly." A web-based graphical interface
for exploring the behavior of normalized stress has also been made [available](https://kiransmelser.github.io/normalized-stress-is-not-normalized/).

## Contents

- `metrics.py`: Python module for computing various stress metrics between high-dimensional and low-dimensional data. Includes an implementation of scale-normalized stress.
- `compute_metrics.py`: Script for computing stress metrics on dimensionality reduction (DR) results.
- `populate_datasets.py`: Script to load and save all datasets used in the experiment.
- `compute_embeddings.py`: Script for computing and saving embeddings using DR techniques for all datasets.
- `kl_statistics.ipynb`: Jupyter notebook for analyzing results of different KL divergence metrics on the considered datasets and embeddings.
- `stress_statistics.ipynb`: Jupyter notebook for analyzing results of different stress metrics on the considered datasets and embeddings.
- `viz.ipynb`: Jupyter notebook for creating the figures seen in the paper.
- `figs/`: Folder containing the figures seen in the paper.

## Requirements

The code in this repository was written using Python version 3.12 and utilizes the following libraries:
- json
- pandas
- numpy
- scipy
- tqdm
- warnings
- sklearn
- umap
- os
- pylab
- zadu
- seaborn
- itertools
- matplotlib
- urllib.request

## Usage

1. Clone this repository: `git clone https://github.com/KiranSmelser/stress-metrics-and-scale`
2. Install the required libraries: `pip install -r requirements.txt`
3. Run the scripts and notebooks as needed. To analyze results of the different metrics, run in the following order:
    1. `populate_datasets.py`: All datasets will be loaded / generated.
    2. `compute_embeddings.py`: Embeddings of the datasets will be created using different DR techniques
    3. `compute_metrics.py`: Data for analyzing the behavior of the different metrics, as well as graphs will be generated
    4. Finally run `kl_statistics.ipynb` and `stress_statistics.ipynb` to obtain statistical results about the KL divergence
    and stress metrics respectively.

<!-- ## Citation

If you find this work useful, please consider citing our paper:
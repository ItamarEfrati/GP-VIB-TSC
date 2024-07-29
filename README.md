# Variational Information Bottleneck with Gaussian Processes for Time-Series Classification

## Abstract

Time series classification problems are prevalent across various domains, often characterized by intra-series relationships within features,
and inter-series relationships between the same features over time. Developing a generalized model capable of capturing these intricate properties poses a considerable challenge.
In this research paper, we introduce an innovative approach termed Gaussian Process Variational Information Bottleneck (GP-VIB). This model is designed as an end-to-end system,
with a primary focus on acquiring a concise representation of the initial sequence. It aims to retain essential information vital for accurate classification.
This stands in contrast to the GP-VAE model, which learns a latent space primarily geared towards information preservation for reconstruction, and necessitates a two-step process for classification.
Through our experiments, we illustrate that the proposed GP-VIB model outperforms existing methods on renowned benchmark datasets.

## Repository Structure

This repository contains the code for implementing and evaluating the GP-VIB model. It is organized as follows:

- `uea.py`: Script for running evaluation over the UEA multivariate archive.
- `hmnist.py`: Script for running evaluation over the HMNIST dataset.
- `configs/`: Directory containing the configuration files.
- `src/`: Directory containing the source code for the model and evaluation tasks.
- `results/`: Directory containing the results of the UEA 30 resample evaluation for each dataset
- `data/`: An empty Directory meant to place the data relevant for the datasets

## Getting Started

### Prerequisites

- Python 3.10+
  
### Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/your-repo/GP-VIB.git
cd GP-VIB
pip install -r requirements.txt
```

## Implementation Details

This repository uses [Hydra](https://hydra.cc/) for managing configuration files and [PyTorch Lightning](https://lightning.ai/) for implementing machine learning models.
Hydra simplifies the process of managing multiple configurations and allows for easy experimentation,
while PyTorch Lightning provides a high-level interface for PyTorch, making it easier to structure and train models.

## Running Evaluations

All results will be saved under the execution running directory in the logs folder

### Configurations
The configuration files are located in the configs/ directory. You can modify these files to change the parameters for different runs. The main configuration files are:

- uea.yaml: Configuration for running evaluations over the UEA multivariate archive.
- hmnist.yaml: Configuration for running evaluations over the HMNIST dataset.


### UEA Multivariate Archive
To run evaluations over the UEA multivariate archive datasets, use the `uea.py` script:
```
python uea.py
```

In order to run specific dataset for a specific number of run you can use the following command 
```
python uea.py dataset_name=EigenWorms num_runs=5
```


### HMNIST Dataset

To run evaluations over the HMNIST dataset, use the hmnist.py script:
```
python hmnist.py
```

or with a specific number of runs
```
python hmnist.py num_runs=3
```


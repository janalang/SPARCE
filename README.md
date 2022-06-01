# SPARCE

Implementation of the SPARCE framework which generates SPARse Counterfactual Explanations for multivariate time series. The architecture was introduced in: Generating Sparse Counterfactual Explanations For Multivariate Time Series. Jana Lang, Martin Giese, Winfried Ilg, Sebastian Otte.

## Abstract

Since neural networks play an increasingly important role in critical sectors, explaining network predictions has become a key research topic. Counterfactual explanations can help to understand why classifier models decide for particular class assignments and, moreover, how the respective input samples would have to be modified such that the class prediction changes. Previous approaches mainly focus on image and tabular data. In this work we propose SPARCE, a generative adversarial network (GAN) architecture that generates SPARse Counterfactual Explanations for multivariate time series. Our approach provides a custom sparsity layer and regularizes the counterfactual loss function in terms of similarity, sparsity, and smoothness of trajectories. We evaluate our approach on real-world human motion datasets as well as a synthetic time series interpretability benchmark. Although we make significantly sparser modifications than other approaches, we achieve comparable or better performance on all metrics. Moreover, we demonstrate that our approach predominantly modifies salient time steps and features, leaving non-salient inputs untouched.

## Prerequisites

Python
PyTorch
NumPy
Matplotlib
Argparse
Datetime
Pandas

## Folder Structure

Create the following folders:
- Data: store input data and labels for training, validation and testing
- Counterfactuals: save the generated counterfactuals along with query and target samples
- Models: save pretrained classification models
- Experiments: save output and log files for each experiment
- Figure: save generated figures

## Data

In the paper, we benchmark our approach on a proprietary clinical dataset, as well as a human motion dataset (https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset) and a synthetic time series interpretability benchmark (https://github.com/ayaabdelsalam91/TS-Interpretability-Benchmark). Input data should consist of multivariate time series truncated to equal length and split into subsets for training, validation and testing.

## Documentation

Sphinx documentation for all modules can be found under docs/build/index.html.











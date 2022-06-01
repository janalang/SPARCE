import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd
import random

"""Load and prepare data for counterfactual search.
"""

def get_feature_mapping(dataset):
    """Return feature mapping for datasets.

    Args:
        dataset (str): Name of dataset

    Returns:
        list: List of features in the dataset
    """
    feature_mapping = []
    if dataset == 'motionsense':
        feature_mapping = ['attitude.roll', 'attitude.pitch', 'attitude.yaw',
        'gravity.x', 'gravity.y', 'gravity.z', 'rotationRate.x',
        'rotationRate.y', 'rotationRate.z', 'userAcceleration.x',
        'userAcceleration.y', 'userAcceleration.z']
    return feature_mapping

def load_data(pathToData, fileFormat, replicate_labels_indicator=False):
    """Load data subsets for training, validation and testing.

    Args:
        pathToData (str): Path to data files
        fileFormat (str): Format of data files
        replicate_labels_indicator (bool, optional): Specifies if labels should be replicated for many-to-many models. Defaults to False.

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test): Data subsets for training, validation and testing
    """
    X_train, y_train = np.load(os.path.join(pathToData, 'X_train' + fileFormat)), np.load(os.path.join(pathToData, 'y_train'+ fileFormat))
    X_val, y_val = np.load(os.path.join(pathToData, 'X_val'+ fileFormat)), np.load(os.path.join(pathToData, 'y_val'+ fileFormat))
    X_test, y_test = np.load(os.path.join(pathToData, 'X_test'+ fileFormat)), np.load(os.path.join(pathToData, 'y_test'+ fileFormat))

    num_timesteps = X_train.shape[1]

    if replicate_labels_indicator: # for many-to-many classification
        y_train = replicate_labels(y_train, num_timesteps)
        y_val = replicate_labels(y_val, num_timesteps)
        y_test = replicate_labels(y_test, num_timesteps)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def prepare_counterfactual_data(args):
    """Prepare datasets for generation of counterfactual explanations. 
    Load data, exclude classes if applicable and split datasets into queries and targets.
    Create separate query and target dataloaders for training and testing.

    Args:
        args: Input arguments

    Returns:
        Query and target dataloaders for training and testing.
    """
    # set directories
    pathToProject = os.getcwd()
    dataset = args.dataset
    pathToData = os.path.join(pathToProject, 'data', dataset) 
    fileFormat = '.npy'
    print(f"Dataset: {dataset}")

    # load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(pathToData, fileFormat)

    # validation data is not used, i.e. combine training data and validation data to a larger training dataset
    X_train = np.vstack((X_train, X_val))
    y_train = np.vstack((y_train, y_val))

    print(f'Shape of training data: {X_train.shape}, {y_train.shape}')
    print(f'Shape of testing data: {X_test.shape}, {y_test.shape}')

    X_train, y_train = sort_data_by_labels(X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2])), y_train.reshape((y_train.shape[0], y_train.shape[1])))
    X_test, y_test = sort_data_by_labels(X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2])), y_test.reshape((y_test.shape[0], y_test.shape[1])))

    X_train_target_samples, y_train_target_samples, X_train_generator_input, y_train_generator_input = split_target_and_input(X_train, y_train, args.target_class)
    X_test_target_samples, y_test_target_samples, X_test_generator_input, y_test_generator_input = split_target_and_input(X_test, y_test, args.target_class)

    train_batchsize = args.batchsize
    test_batchsize = 1

    train_max_samples = min(X_train_target_samples.shape[0], X_train_generator_input.shape[0])
    train_max_batches = min(np.ceil(train_max_samples / train_batchsize), args.max_batches)

    test_max_samples = min(X_test_target_samples.shape[0], X_test_generator_input.shape[0])
    test_max_batches = np.ceil(test_max_samples / test_batchsize)

    X_train_target_samples, y_train_target_samples = take_max_samples(args.seed, X_train_target_samples, y_train_target_samples, train_max_samples)
    X_train_generator_input, y_train_generator_input = take_max_samples(args.seed, X_train_generator_input, y_train_generator_input, train_max_samples)

    X_test_target_samples, y_test_target_samples = take_max_samples(args.seed, X_test_target_samples, y_test_target_samples, test_max_samples)
    X_test_generator_input, y_test_generator_input = take_max_samples(args.seed, X_test_generator_input, y_test_generator_input, test_max_samples)
    
    print(f"Shape of training samples: {X_train_target_samples.shape}, {y_train_target_samples.shape} (target), {X_train_generator_input.shape}, {X_train_generator_input.shape} (different)")
    print(f"Shape of test samples: {X_test_target_samples.shape}, {y_test_target_samples.shape} (target), {X_test_generator_input.shape}, {X_test_generator_input.shape} (different)")

    X_train_real_samples, y_train_real_samples = torch.from_numpy(X_train_target_samples), torch.from_numpy(y_train_target_samples)
    X_train_generator_input, y_train_generator_input = torch.from_numpy(X_train_generator_input), torch.from_numpy(y_train_generator_input)

    X_test_real_samples, y_test_real_samples = torch.from_numpy(X_test_target_samples), torch.from_numpy(y_test_target_samples)

    X_test_generator_input, y_test_generator_input = torch.from_numpy(X_test_generator_input), torch.from_numpy(y_test_generator_input)

    # construct pytorch datasets
    train_ds_target = TensorDataset(X_train_real_samples, y_train_real_samples)
    train_ds_input = TensorDataset(X_train_generator_input, y_train_generator_input)

    test_ds_target = TensorDataset(X_test_real_samples, y_test_real_samples)
    test_ds_input = TensorDataset(X_test_generator_input, y_test_generator_input)

    # pass datasets to dataloaders (wraps an iterable over the dataset)
    train_dl_real_samples = DataLoader(train_ds_target, train_batchsize, shuffle=False)
    train_dl_generator_input = DataLoader(train_ds_input, train_batchsize, shuffle=False)

    test_dl_real_samples = DataLoader(test_ds_target, test_batchsize, shuffle=False)
    test_dl_generator_input = DataLoader(test_ds_input, test_batchsize, shuffle=False)

    return X_train_generator_input, train_dl_real_samples, train_dl_generator_input, test_dl_real_samples, test_dl_generator_input, train_max_samples, train_max_batches, test_max_samples, test_max_batches


def take_max_samples(seed, X, y, max_samples):
    """Shuffle X and y and take the first max samples.

    Args:
        seed: Random seed
        X: Data
        y: Labels
        max_samples (int): Maximum number of samples to extract
    """
    random.seed(seed)
    # shuffle arrays, then take first max samples
    idx_list = list(range(X.shape[0]))
    random.shuffle(idx_list)

    X = X[idx_list]
    y = y[idx_list]

    X = X[:max_samples]
    y = y[:max_samples]

    return X, y


def sort_data_by_labels(X, y):
    """Group input data by class labels.

    Args:
        X: Data
        y: Labels
    """

    # input: X and y as numpy arrays
    X_flat = X.reshape((X.shape[0], X.shape[1] * X.shape[2])) # collapse time and feature dimension
    df = pd.DataFrame(X_flat)
    y_class = np.argmax(y, axis=1)
    df['y'] = y_class
    df = df.sort_values(by='y')
    X_sorted = df.iloc[:,:-1].to_numpy().reshape((X.shape[0], X.shape[1], X.shape[2]))

    df = pd.DataFrame(y)
    df['y'] = y_class
    df = df.sort_values(by='y')
    y_sorted = df.iloc[:,:-1].to_numpy().reshape((y.shape[0], y.shape[1]))

    return X_sorted, y_sorted


def split_target_and_input(X, y, target_class):
    """Split input data and labels into queries and targets.

    Args:
        X: Data
        y: Labels
        target_class (int): Desired target class for counterfactuals
    """
    X_target_samples = X[np.argmax(y, axis=1) == target_class]
    y_target_samples = y[np.argmax(y, axis=1) == target_class]

    X_generator_input = X[np.argmax(y, axis=1) != target_class]
    y_generator_input = y[np.argmax(y, axis=1) != target_class]

    return X_target_samples, y_target_samples, X_generator_input, y_generator_input


def get_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batchsize):
    """Create training, validation and testing dataloaders for numpy arrays.

    Args:
        X_train (np.array): Training data
        y_train (np.array): Training labels
        X_val (np.array): Validation data
        y_val (np.array): Validation labels
        X_test (np.array): Testing data
        y_test (np.array): Testing labels
        batchsize (int): Desired size of batches

    Returns:
        train_dl, val_dl, test_dl: Dataloaders for training, validation and testing
    """

    # transform all numpy arrays to torch tensors
    X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
    X_val, y_val = torch.from_numpy(X_val), torch.from_numpy(y_val)
    X_test, y_test = torch.from_numpy(X_test), torch.from_numpy(y_test)

    # construct pytorch datasets
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)

    # pass datasets to dataloaders (wraps an iterable over the dataset)
    train_dl = DataLoader(train_ds, batchsize, shuffle=False)
    val_dl = DataLoader(val_ds, batchsize, shuffle=False)
    test_dl = DataLoader(test_ds, batchsize, shuffle=False)

    return train_dl, val_dl, test_dl


def replicate_labels(y, num_timesteps):
    """Replicate labels for many-to-many sequence classification.

    Args:
        y: Labels
        num_timesteps (int): Number of time steps
    """
    y_new = np.zeros((y.shape[0], num_timesteps, y.shape[1]))
    for trial in range(y.shape[0]):
        y_new[trial] = np.stack([y[trial] for i in range(num_timesteps)], axis=0) # replicate label num_timesteps times
    return y_new
        


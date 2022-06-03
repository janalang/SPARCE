import torch
import os
from datetime import date
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(modelname, dataset, model, num_epochs, batchsize, loss_fn, optimizer):
    """Save pretrained classification model as a .pt file as well as training parameters as a .csv file.

    Args:
        modelname (str): Name of pretrained model
        dataset (str): Name of dataset
        model: Pretrained torch model
        num_epochs (int): Number of training epochs
        batchsize (int): Number of samples per batch
        loss_fn: Loss function
        optimizer: Optimizer
    """
    version = 1
    cwd = os.getcwd()
    for file in os.listdir(os.path.join(cwd, 'models', dataset)):
        if file.startswith(modelname) and file.endswith('.pt'):
            version += 1

    modelname += f'_v{version}'

    print("Saving the pretrained model \n--------------------------------------")
    torch.save(model, os.path.join(cwd, 'models', dataset, modelname + '.pt'))

    parameters = [dataset, modelname, date.today(), num_epochs, batchsize, str(model), loss_fn, optimizer]
    parameter_df = pd.DataFrame([parameters], columns=['dataset', 'modelname', 'date', 'num_epochs', 'batchsize', 'architecture', 'loss_fn', 'optimizer'])
    print(parameter_df)
    parameter_df.to_csv(os.path.join(cwd, 'models', dataset, modelname + '.csv'), index=False)

def load_model(dataset, modelname, version=-1):
    """Load a pretrained torch classification model.

    Args:
        dataset (str): Name of dataset
        modelname (str): Name of pretrained model
        version (int, optional): Version of pretrained model. Defaults to -1.

    Returns:
        model: Pretrained torch classification model
    """
    cwd = os.getcwd()
    # if version is not specified, get most recent model version
    if version == -1:
        version = 0
        for file in os.listdir(os.path.join(cwd, 'models', dataset)):
            if file.startswith(modelname) and file.endswith('.pt'):
                version += 1

    modelname += f'_v{version}'

    pathToModel =  os.path.join(cwd, 'models', dataset, modelname + '.pt')
    model = torch.load(pathToModel)
    print(model)
    return model

def test_classifier(test_dl, model, loss_fn):
    """Test the trained classifier on unseen testing data.

    Args:
        test_dl (torch DataLoader): Data subset for testing
        model: Pretrained classification model
        loss_fn: Loss function
    """
    print("Testing the model...\n-------------------------------")
    test_size = len(test_dl.dataset)
    num_batches = len(test_dl)
    model.eval()
    test_loss, test_correct = 0, 0
    with torch.no_grad():
        for X, y in test_dl:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            y =  torch.argmax(y, dim=1)
            y = y.long()
            test_loss += loss_fn(pred, y).item()
            test_correct += (torch.argmax(pred, dim=1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    test_correct /= test_size
    print(f"Testing: \n Accuracy: {(100*test_correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def get_avg_loss(losses, loss_type):
    """Compute average loss over epochs

    Args:
        losses (list): List of loss dictionaries for each epoch
        loss_type (str): Name of loss term

    Returns:
        loss: Average loss over epochs
    """
    loss = 0
    for sample in range(len(losses)):
        loss += losses[sample][loss_type]
    loss /= len(losses)
    return loss



import numpy as np
import os
from dataloader import load_data, get_dataloaders
import torch
from classifier import LSTMClassifier, BidirectionalLSTMClassifier
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import Adam
from utils import test_classifier, save_model

# set directories
home = os.path.expanduser('~')
pathToProject = os.getcwd()
dataset = 'simulated'
pathToData = os.path.join(pathToProject, 'data', dataset) 
fileFormat = '.npy'
batchsize = 32
device = "cpu"

replicate_labels_indicator = False # use for many-to-many classification
bidirectional_indicator = True

print(f"Dataset: {dataset}")

# load data
(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(pathToData, fileFormat, replicate_labels_indicator=replicate_labels_indicator)

print(X_train.shape)
print(y_train.shape)

train_dl, val_dl, test_dl = get_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batchsize)

num_features = X_train.shape[2]
num_timesteps = X_train.shape[1]
if replicate_labels_indicator:
    output_dim = y_train.shape[2]
else:
    output_dim = y_train.shape[1] # number of classes, 3 or 6

hidden_dim = 32
layer_dim = 2

# instantiate model
if bidirectional_indicator:
    model = BidirectionalLSTMClassifier(num_features, hidden_dim, layer_dim, output_dim, replicate_labels_indicator)
else:
    model = LSTMClassifier(num_features, hidden_dim, layer_dim, output_dim, replicate_labels_indicator)

model = model.to(device)
print(model)

# define loss function and optimizer
num_epochs = 5
INIT_LR = 1e-3
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=INIT_LR)


def train(num_epochs, train_dl, val_dl, model, loss_fn, optimizer):
    print("Starting to train the model...\n-------------------------------")
    train_loss_per_epoch, train_acc_per_epoch = [], []
    val_loss_per_epoch, val_acc_per_epoch = [], []
    fig1, fig2 = plt.figure(), plt.figure()
    for t in range(num_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        size = len(train_dl.dataset)
        num_batches = len(train_dl)
        train_loss, correct = 0, 0
        model.train()
        for batch, (X, y) in enumerate(train_dl):
          
            X, y = X.to(device), y.to(device)

            # compute prediction error
            pred = model(X)
            y = torch.argmax(y, dim= -1) # -1 for last dimension
            y = y.long()

            if not replicate_labels_indicator:
                correct += (torch.argmax(pred, dim=-1) == y).type(torch.float).sum().item()
            else:
                correct += (torch.argmax(pred, dim=-1)[:,0] == y[:,0]).type(torch.float).sum().item()
                pred = torch.reshape(pred, (pred.shape[0], pred.shape[2], pred.shape[1])) # because crossentropy loss expects [N, C, d...]
            loss = loss_fn(pred, y)
            train_loss += loss.item()


            # backpropagate error
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= num_batches
        correct /= size
        print(f"Training: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")

        train_loss_per_epoch.append(train_loss)
        train_acc_per_epoch.append(100*correct)

        val_size = len(val_dl.dataset)
        num_batches_val = len(val_dl)
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(device), y.to(device)

                pred = model(X)
                y =  torch.argmax(y, dim=-1)
                y = y.long()

                if not replicate_labels_indicator:
                    val_correct += (torch.argmax(pred, dim=-1) == y).type(torch.float).sum().item()
                else:
                    val_correct += (torch.argmax(pred, dim=-1)[:,0] == y[:,0]).type(torch.float).sum().item()
                    pred = torch.reshape(pred, (pred.shape[0], pred.shape[2], pred.shape[1])) # because crossentropy loss expects [N, C, d...]


                val_loss += loss_fn(pred, y).item()

                loss = loss_fn(pred, y)
                train_loss += loss.item()

        val_loss /= num_batches_val
        val_correct /= val_size
        print(f"Validation: \n Accuracy: {(100*val_correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")

        val_loss_per_epoch.append(val_loss)
        val_acc_per_epoch.append(100*val_correct)

    plt.figure(fig1)
    plt.title(f"Training and validation loss over {num_epochs} epochs")
    plt.plot(np.linspace(1, num_epochs, num_epochs).astype(int), train_loss_per_epoch, label='training')
    plt.plot(np.linspace(1, num_epochs, num_epochs).astype(int), val_loss_per_epoch, label='validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')


    plt.figure(fig2)
    plt.title(f"Training and validation accuracy over {num_epochs} epochs")
    plt.plot(np.linspace(1, num_epochs, num_epochs).astype(int), train_acc_per_epoch, label='training')
    plt.plot(np.linspace(1, num_epochs, num_epochs).astype(int), val_acc_per_epoch, label='validation')
    plt.legend()
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Accuracy')

def test(test_dl, model, loss_fn):
    test_classifier(test_dl, model, loss_fn)


train(num_epochs, train_dl, val_dl, model, loss_fn, optimizer)

if bidirectional_indicator:
    modelname = 'bidirectional_lstm_classifier'
else:
    modelname = 'lstm_classifier'
    
save_model(modelname, dataset, model, num_epochs, batchsize, loss_fn, optimizer)

plt.show()
test(test_dl, model, loss_fn)
print("Done!")


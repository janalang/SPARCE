import torch
from torch import nn

"""
Recurrent sequence classification models for the classification of generated counterfactuals.
"""

device = "cpu"

class LSTMClassifier(nn.Module):
    """
    LSTM model for multivariate time series classification
    Args:
        nn (torch.nn): torch.nn Module
    """

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, replicate_labels_indicator):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.replicate_labels_indicator = replicate_labels_indicator
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)
        # self.act = nn.Softmax(dim=1) # dim=1 # don't use softmax since CrossEntropyLoss already combines LogSoftmax and NLLLoss
        self.batch_size = None
        self.hidden = None

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        x = x.float()
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        if not self.replicate_labels_indicator:
            out = out[:, -1, :] # drop the time dimension
        out = self.fc(out) 
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim) # initial hidden state for each element in the batch
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim) # initial cell state for each element in the batch
        return [t.to(device) for t in (h0, c0)]


class BidirectionalLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM model for multivariate time series classification
    Args:
        nn (torch.nn): torch.nn Module
    """

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, replicate_label_indicator):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.replicate_label_indicator = replicate_label_indicator
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)
        # self.act = nn.Softmax(dim=1) # dim=1 # don't use softmax since CrossEntropyLoss already combines LogSoftmax and NLLLoss in one single class.
        if output_dim == 1:
            self.act = nn.Sigmoid()
        self.batch_size = None
        self.hidden = None

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        # x: torch.Size([batch, time, features])
        x = x.float().to(device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # out: torch.Size([batch, time, hidden_dim])
        if not self.replicate_label_indicator:
            out = out[:, -1, :] # drop the time dimension (equals return_sequences = False)
        out = self.fc(out) 
        # out: torch.Size([batch, output_dim])
        if self.output_dim == 1:
            out = self.act(out)
        # out: torch.Size([batch, output_dim])
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(2 * self.layer_dim, x.size(0), self.hidden_dim) # initial hidden state for each element in the batch
        c0 = torch.zeros(2 * self.layer_dim, x.size(0), self.hidden_dim) # initial cell state for each element in the batch
        return [t.to(device) for t in (h0, c0)]
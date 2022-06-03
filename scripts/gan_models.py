import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_model_parameters(model, trainable_only=False):
    """Count number of parameters of a torch model.

    Args:
        model: torch model
        trainable_only (bool, optional): If true, only count trainable parameters. Defaults to False.

    Returns:
        num_parameters: Number of parameters of the model
    """
    if trainable_only == False:
        num_parameters = sum(p.numel() for p in model.parameters())
    else:
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_parameters


class ResidualGANLSTM(nn.Module):
    """Recurrent LSTM-based generator model that produces counterfactual modifications (residuals) for an input query.

    Args:
        nn (torch.nn): torch.nn Module
    """

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, freeze_indices=[]):
        super().__init__()
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.freeze_indices = freeze_indices


        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=0.4)
        self.fc1 = nn.Linear(hidden_dim, output_dim - len(self.freeze_indices))
        self.fc2 = nn.Linear(hidden_dim, output_dim - len(self.freeze_indices))

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        x = x.float()
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        fc1_out = self.fc1(lstm_out)
        fc2_out = self.fc2(lstm_out)

        act1_out = self.act1(fc1_out)
        act2_out = self.act2(fc2_out)
        act_out = act1_out - act2_out
        act_out = act_out.double()
        return act_out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim) 
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim) 
        return [t.to(device) for t in (h0, c0)]


class BidirectionalLSTM(nn.Module):
    """Recurrent bidirectional LSTM discriminator model for many-to-one sequence classification.

    Args:
        nn (torch.nn): torch.nn Module
    """

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=0.4, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)
        self.act = nn.Sigmoid()

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        x = x.float()
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        if self.output_dim == 1: # discriminator
            lstm_out = lstm_out[:, -1, :] # drop the time dimension (equals return_sequences = False)
        fc_out = self.fc(lstm_out)
        act_out = self.act(fc_out)
        return act_out

    def init_hidden(self, x):
        h0 = torch.zeros(2 * self.layer_dim, x.size(0), self.hidden_dim) 
        c0 = torch.zeros(2 * self.layer_dim, x.size(0), self.hidden_dim) 
        return [t.to(device) for t in (h0, c0)]

class BidirectionalResidualGANLSTM(nn.Module):
    """Recurrent bidirectional LSTM generator model that produces counterfactual modifications (residuals) for an input query.

    Args:
        nn (torch.nn): torch.nn Module
    """


    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, freeze_indices=[]):
        super().__init__()
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.freeze_indices = freeze_indices


        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=0.4, bidirectional=True)
        self.fc1 = nn.Linear(2 * hidden_dim, output_dim - len(self.freeze_indices))
        self.fc2 = nn.Linear(2 * hidden_dim, output_dim - len(self.freeze_indices))

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()


    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        x = x.float()
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

        fc1_out = self.fc1(lstm_out)
        fc2_out = self.fc2(lstm_out)

        act1_out = self.act1(fc1_out)
        act2_out = self.act2(fc2_out)
        act_out = act1_out - act2_out
        act_out = act_out.double()
        return act_out

    def init_hidden(self, x):
        h0 = torch.zeros(2 * self.layer_dim, x.size(0), self.hidden_dim) 
        c0 = torch.zeros(2 * self.layer_dim, x.size(0), self.hidden_dim) 
        return [t.to(device) for t in (h0, c0)]



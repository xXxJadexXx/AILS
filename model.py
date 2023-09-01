
import torch
import torch.nn as nn



class MoleculeRNN(nn.Module):

    def __init__(self, chars, n_hidden=10, n_layers=2, device=None):
        super(MoleculeRNN, self).__init__()

        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.chars = chars
        self.device = device

        self.lstm1 = nn.LSTM(chars, n_hidden, n_layers, batch_first=True)

        #self.lstm2 = nn.LSTM(n_hidden, n_hidden, n_layers,
                             #dropout=0.1, batch_first=True)

        self.fc = nn.Linear(n_hidden, chars)

    def forward(self, x, hidden):

        #times = x.shape[1]

        x, hidden1 = self.lstm1(x, hidden)

        #x, hidden2 = self.lstm2(x, hidden1)

        #x = torch.flatten(x, 1)

        x = self.fc(x)

        #x = x.view(-1, times, self.chars)

        return x, hidden1


    def init_hidden(self, x):

        hidden = [torch.zeros(self.n_layers, x.shape[0], self.n_hidden).to(self.device),
                  torch.zeros(self.n_layers, x.shape[0], self.n_hidden).to(self.device)]

        return hidden


















class Adversarial(nn.Module):

    def __init__(self):
        super(Adversarial, self).__init__()

        self.convs = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=120, out_channels=64, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.LeakyReLU(0.2, inplace=True),
        )

        self.lin = torch.nn.Sequential(
            torch.nn.Linear(512*6, 1),
            torch.nn.Sigmoid()
        )


    def forward(self, x):

        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.lin(x)


        return x



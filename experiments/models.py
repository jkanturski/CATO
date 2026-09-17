"""
models.py — LSTM, TCN, and hybrid TCN+XGBoost model definitions for
Poland CDS 5Y forecasting (Sections III.C/D/E of paper.tex).

Shared by train_lstm_tcn_ddp.py. Architectures mirror the reference
implementations already used on the CATO cluster:
  - LSTM:  /Users/rklopotek/CATO/ddp_lstm.py (SimpleLSTM)
  - TCN:   /Users/rklopotek/CATO/dl/model.py (CryptoTCN)
extended here from classification/synthetic-data to regression on real
CDS market data, with a configurable multi-horizon output head.
"""
import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_horizons=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_horizons)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride,
                                padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                                padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNForecaster(nn.Module):
    """Temporal Convolutional Network with an accessible penultimate
    embedding, used both standalone (Section III.D) and as the upstream
    feature extractor in the hybrid model (Section III.E)."""

    def __init__(self, input_size, num_channels=(64, 64, 64), kernel_size=3,
                 dropout=0.2, num_horizons=1):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size, stride=1,
                    dilation=dilation_size, padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)
        self.embedding_dim = num_channels[-1]
        self.linear = nn.Linear(self.embedding_dim, num_horizons)

    def forward(self, x, return_embedding=False):
        # x: (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        y = self.network(x)
        embedding = y[:, :, -1]  # (batch, embedding_dim), penultimate layer
        out = self.linear(embedding)
        if return_embedding:
            return out, embedding
        return out

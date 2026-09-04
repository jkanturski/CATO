import torch
import torch.nn as nn
import torch.nn.functional as F

class Chomp1d(nn.Module):
    """
    Removes the extra padding at the end of the convolution output 
    to enforce strict causal history (no looking into the future).
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    """
    A single residual block with dilated causal convolutions, 
    ReLU activation, and dropout.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        # 1x1 convolution downsample if channel dimensions change across blocks
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


class CryptoTCN(nn.Module):
    """
    Temporal Convolutional Network for parallelized time-series forecasting.
    """
    def __init__(self, input_size=15, num_channels=[64, 64, 64], kernel_size=3, dropout=0.2):
        super(CryptoTCN, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            # Exponentially increasing dilation (1, 2, 4, ...) to expand receptive field
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels, 
                    out_channels, 
                    kernel_size, 
                    stride=1, 
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size, 
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)
        
        # Final linear mapping layer for regression
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        """
        Input shape:  (batch_size, seq_len, input_size)
        Conv1d shape: (batch_size, input_size, seq_len)
        """
        x = x.transpose(1, 2)
        
        # Pass through the dilated convolutional residual blocks
        y = self.network(x)
        
        # Extract the prediction from the final time step in the sequence
        out = self.linear(y[:, :, -1])
        
        return out

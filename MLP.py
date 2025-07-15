import torch
import torch.nn as nn

# Defines a generic fully connected MLP architecture that can be used for the encoder/decoder networks.
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, activation):
        """
        Initializes the MLP model.
        :param input_size: Integer, size of the input layer.
        :param hidden_layers: List of integers, where each integer is the size of a hidden layer.
        :param output_size: Integer, size of the output layer.
        """
        super(MLP, self).__init__()
        layers = [nn.Linear(input_size, hidden_layers[0]), activation]  # Input layer
        
        # Creating hidden layers
        for i in range(1, len(hidden_layers)):
            layers += [nn.Linear(hidden_layers[i - 1], hidden_layers[i]), activation]
        
        # Registering the sequence of layers
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass of the MLP.
        :param x: Input tensor.
        :return: Output tensor.
        """
        return self.layers(x)
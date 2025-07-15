import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from MLP import MLP

class q_net(MLP):
    def __init__(self, input_size, hidden_layers, latent_dim, activation, logvar_range=None):
        """
        Initializes the Encoder model.
        :param input_size: Integer, size of the input layer.
        :param hidden_layers: List of integers, where each integer is the size of a hidden layer.
        :param output_size: Integer, size of the output layer.
        """
        super(q_net, self).__init__(input_size, hidden_layers, activation)
        self.mean_layer = nn.Linear(hidden_layers[len(hidden_layers)-1] , latent_dim)
        self.logvar_layer = nn.Linear(hidden_layers[len(hidden_layers)-1] , latent_dim)
        self.lvr = logvar_range

    def forward(self, x):
        x = super(q_net, self).forward(x) 
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        if self.lvr is not None:
            logvar = torch.clamp(logvar, min=self.lvr[0], max=self.lvr[1]) 
        
        return mean, logvar

class Decoder(MLP):
    def __init__(self, latent_dim, hidden_layers, input_size, activation):
        """
        Initializes the Encoder model.
        :param input_size: Integer, size of the input layer.
        :param hidden_layers: List of integers, where each integer is the size of a hidden layer.
        :param output_size: Integer, size of the output layer.
        """
        super(Decoder, self).__init__(latent_dim, hidden_layers, activation)
        self.out = nn.Linear(hidden_layers[len(hidden_layers)-1] , input_size)

    def forward(self, x):
        x = super(Decoder, self).forward(x)  
        return self.out(x)

class q_net_delta(MLP):
    def __init__(self, input_size, hidden_layers, latent_dim, activation):
        """
        Initializes the Encoder model.
        :param input_size: Integer, size of the input layer.
        :param hidden_layers: List of integers, where each integer is the size of a hidden layer.
        :param latent_dim: Integer, size of the latent dimension.
        :param activation: Activation function to use.
        """
        super(q_net_delta, self).__init__(input_size, hidden_layers, activation)
        self.mean_layer = nn.Linear(hidden_layers[-1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_layers[-1], latent_dim)

    def forward(self, x):
        x = super(q_net_delta, self).forward(x)  
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        
        # Clamp logvar to prevent it from becoming too large or too small
        logvar = torch.clamp(logvar, min=-5, max=3)  
        return mean, logvar

# Gaussian cross-entropy
def gauss_cross_entropy(mu1, var1, mu2, var2):
    """
    Computes the element-wise cross entropy
    Given q(z) ~ N(z| mu1, var1)
    returns E_q[ log N(z| mu2, var2) ]
    args:
        mu1:  mean of expectation 
        var1: var  of expectation 
        mu2:  mean of integrand 
        var2: var of integrand 

    returns:
        cross_entropy: 
    """
    term0 = torch.log(2*torch.tensor(torch.pi))
    term1 = torch.log(var2)
    term2 = (var1 + (mu1 - mu2) ** 2) / var2

    cross_entropy = -0.5 * (term0 + term1 + term2)

    return cross_entropy

def vae_loss(mu, var, mu_prior, var_prior, reduction='mean'):
    """
    Calculate the total loss for the VAE.
    
    Parameters:
        mu: the mean from the latent space (output of the encoder)
        var: the variance from the latent space (output of the encoder)
        mu_prior: the mean of the prior distribution
        var_prior: the variance of the prior distribution
        reduction: str, specifies the reduction method ('mean' or 'sum'). Default is 'mean'.
        
    Returns:
        The total VAE loss.
    """
    logvar = torch.log(var)
    logvar_prior = torch.log(var_prior)
    kl_div_elements = -0.5 * (1 + logvar - logvar_prior - ((var + (mu - mu_prior).pow(2)) / var_prior))
    
    if reduction == 'sum':
        kl_div = torch.sum(kl_div_elements)
    elif reduction == 'mean':
        kl_div = torch.mean(kl_div_elements)
    else:
        raise ValueError("Reduction must be either 'mean' or 'sum'.")
    
    return kl_div

def regularization_loss(x_hat, x_hat_phy):
    """
    Computes the regularization loss to minimize the difference between x_hat and x_hat_phy.

    Args:
        x_hat (Tensor): The output tensor of the VAE (reconstructed output).
        x_hat_phy (Tensor): The output tensor of the physical decoder.

    Returns:
        Tensor: The regularization loss.
    """
    return F.mse_loss(x_hat, x_hat_phy, reduction='mean')


class AnnealingStrategy:
    def __init__(self, strategy='linear', total_epochs=100, beta_min=0, beta_max=1, k=5, num_cycles=1):
        """
        Initialize the annealing strategy.
        
        Args:
            strategy (str): Annealing strategy ('linear', 'sigmoid', 'exponential', 'cyclic').
            total_epochs (int): Total number of epochs for annealing.
            beta_min (float): Minimum beta value.
            beta_max (float): Maximum beta value.
            k (float): Hyperparameter controlling the steepness (used for sigmoid and exponential).
            num_cycles (int): Number of cycles (used for cyclic annealing).
        """
        self.strategy = strategy
        self.total_epochs = total_epochs
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.k = k
        self.num_cycles = num_cycles
    
    def linear_annealing(self):
        """ Linear annealing from beta_min to beta_max over total_epochs. """
        return np.linspace(self.beta_min, self.beta_max, self.total_epochs)
    
    def sigmoid_annealing(self):
        """ Sigmoid annealing from beta_min to beta_max with adjustable steepness. """
        midpoint = self.total_epochs // 2
        beta_range = self.beta_max - self.beta_min
        return self.beta_min + beta_range / (1 + np.exp(-self.k * (np.arange(self.total_epochs) - midpoint)))
    
    def exponential_annealing(self):
        """ Exponential annealing from beta_min to beta_max with parameter k controlling the growth rate. """
        beta_range = self.beta_max - self.beta_min
        return self.beta_min + beta_range * (1 - np.exp(-self.k * np.arange(self.total_epochs)))
    
    def cyclic_annealing(self):
        """ Cyclic annealing from beta_min to beta_max that repeats over a number of cycles. """
        cycle_length = self.total_epochs // self.num_cycles
        beta = np.zeros(self.total_epochs)
        for cycle in range(self.num_cycles):
            start = cycle * cycle_length
            end = start + cycle_length
            beta[start:end] = np.linspace(self.beta_min, self.beta_max, cycle_length)
        return beta
    
    def get_beta_vector(self):
        """ Generate the beta vector based on the chosen strategy. """
        if self.strategy == 'linear':
            return self.linear_annealing()
        elif self.strategy == 'sigmoid':
            return self.sigmoid_annealing()
        elif self.strategy == 'exponential':
            return self.exponential_annealing()
        elif self.strategy == 'cyclic':
            return self.cyclic_annealing()
        else:
            raise ValueError(f"Unknown strategy {self.strategy}. Choose from 'linear', 'sigmoid', 'exponential', 'cyclic'.")

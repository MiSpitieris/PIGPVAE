import torch
import torch.nn as nn
import gpytorch
import pandas as pd
from gpytorch.distributions import MultivariateNormal

# Dynamic (in terms of input data updates) GP model


class DynamicExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, likelihood, kernel, mean_module=None):
        super(DynamicExactGPModel, self).__init__(
            train_inputs=None, train_targets=None, likelihood=likelihood
        )
        # Use the provided mean_module or default to ZeroMean if not provided
        self.mean_module = mean_module if mean_module is not None else gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# GP inference model
class GP_inference(nn.Module):
    def __init__(self, kernel, mean_module=None):
        super(GP_inference, self).__init__()
        self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.ones(1))
        self.gp_model = DynamicExactGPModel(self.likelihood, kernel, mean_module)

    def forward(self, X, y, q_var):
        # Ensure correct dimensions
        y = y.squeeze(-1) if y.dim() > 1 else y    # Shape [N]
        X = X.unsqueeze(-1) if X.dim() == 1 else X  # Shape [N, 1]
        q_var = q_var.squeeze(-1) if q_var.dim() > 1 else q_var  # Shape [N]

        # Set the training data dynamically
        self.gp_model.set_train_data(inputs=X, targets=y, strict=False)
        self.likelihood.noise = q_var

        # Compute the marginal log-likelihood
        self.gp_model.train()
        self.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)
        output = self.gp_model(X)
        ll = mll(output, y)

        # Compute predictive mean and variance without torch.no_grad()
        # Ensure that test caches are not detached to allow gradients to flow
        self.gp_model.eval()
        self.likelihood.eval()
        with gpytorch.settings.detach_test_caches(False):
            pred_dist = self.likelihood(self.gp_model(X))
            mu = pred_dist.mean.unsqueeze(1)            # Shape [N, 1]
            var = pred_dist.variance.unsqueeze(1)       # Shape [N, 1]

        return mu, var, ll
        
    def cond_mu_cov(self, X, X_c, y, q_var):
        """
        Compute mean and covariance at new temporal locations X_c.
        """
        # Ensure correct dimensions
        y = y.squeeze(-1) if y.dim() > 1 else y          # Shape [N]
        X = X.unsqueeze(-1) if X.dim() == 1 else X       # Shape [N, 1]
        X_c = X_c.unsqueeze(-1) if X_c.dim() == 1 else X_c  # Shape [N_c, 1]
        q_var = q_var.squeeze(-1) if q_var.dim() > 1 else q_var  # Shape [N]

        # Set the training data dynamically
        self.gp_model.set_train_data(inputs=X, targets=y, strict=False)
        self.likelihood.noise = q_var

        # **Keep the model in training mode to avoid GPInputWarning**
        # Compute predictive mean and covariance without torch.no_grad()
        with gpytorch.settings.detach_test_caches(False):
            pred_dist = self.likelihood(self.gp_model(X_c))
            mu = pred_dist.mean                          # Shape [N_c]
            cov = pred_dist.covariance_matrix            # Shape [N_c, N_c]

        return mu, cov

    def mean_values_per_time_df_multi(self, data, t_gen):
        """
        Compute mean values for each unique time point from a PyTorch tensor,
        converting it into a Pandas DataFrame for processing. This version supports
        multiple value columns and filters results based on a given set of time points.

        :param data: A 2D PyTorch tensor where the first column is time and 
                     the subsequent columns are the corresponding values.
        :param t_gen: A 1D PyTorch tensor or list containing specific time points to retain in the output.
        :return: A PyTorch tensor with specified time points and their mean values
                 for each value column.
        """
        # Convert the PyTorch tensor to a Pandas DataFrame
        num_columns = data.size(1)  # Get the number of columns in the tensor
        column_names = ['Time'] + [f'Value_{i}' for i in range(1, num_columns)]
        df = pd.DataFrame(data.cpu().numpy(), columns=column_names)

        # Group by 'Time' and calculate the mean for each group
        mean_df = df.groupby('Time').mean().reset_index()

        # Filter the DataFrame to only include the specified time points
        t_gen_df = pd.DataFrame(t_gen.cpu().numpy(), columns=['Time'])
        filtered_mean_df = pd.merge(t_gen_df, mean_df, on='Time', how='left')

        # Convert the resulting Pandas DataFrame back into a PyTorch tensor
        # Here we keep only the value columns, assuming the first column is 'Time'
        mean_tensor = torch.tensor(filtered_mean_df.values[:, 1:], dtype=data.dtype, device=data.device)

        return mean_tensor

def sample_MVN(mu, X, model, num_samples=1):
    """
    Generate samples from a multivariate normal distribution using GPyTorch,
    utilizing the learned GP kernel and parameters.

    Parameters:
    - mu: Mean vector of the distribution (tensor of shape [N] or [N, 1])
    - X: Input tensor (e.g., time points) of shape [N] or [N, 1]
    - model: The GP model with learned parameters
    - num_samples: Number of samples to generate

    Returns:
    - samples: Generated samples, tensor of shape [num_samples, N]
    """
    # Ensure X is of shape [N, 1]
    if X.dim() == 1:
        X = X.unsqueeze(-1)
    # if mu.dim() == 1:
        # mu = mu.unsqueeze(-1)
    # Check if mu is 'mean_module', else use provided mu
    if isinstance(mu, str) and mu == 'mean_module':
        mu = model.GP.gp_model.mean_module(X)
    elif mu is not None:
        # Ensure mu is of shape [N, 1] if it's provided
        if mu.dim() == 1:
            mu = mu.unsqueeze(-1)
        mu = mu.squeeze()  # Ensure mu is of shape [N]

    # Ensure mu is of shape [N]
    mu = mu.squeeze()
    
    # Get the GP model's kernel (with learned parameters)
    kernel = model.GP.gp_model.covar_module
    
    # Compute the covariance matrix (as a LazyTensor)
    covar_x = kernel(X)  
    
    # Create the MultivariateNormal distribution
    mvn_dist = MultivariateNormal(mu, covar_x)
    
    # Sample from the distribution
    samples = mvn_dist.rsample(torch.Size([num_samples]))  # Shape: [num_samples, N]
    
    return samples.T

def sample_MVN_cov(mu, cov, num_samples=1):
    
    # Create the MultivariateNormal distribution
    mvn_dist = MultivariateNormal(mu, cov)
    
    # Sample from the distribution
    samples = mvn_dist.rsample(torch.Size([num_samples]))  # Shape: [num_samples, N]
    
    return samples.T
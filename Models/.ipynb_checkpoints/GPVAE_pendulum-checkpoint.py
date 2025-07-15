import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gpytorch
import copy
from tqdm import tqdm

sys.path.append(os.path.abspath('..'))
from GP import *
from MLP import *
from VAE_utils import gauss_cross_entropy, vae_loss, regularization_loss, q_net, Decoder


class GPVAE(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_layers_encoder=[10],
        hidden_layers_decoder=[10, 10],
        latent_dim=1,
        output_size=1,
        GP_inf=None,
        activation_encoder=nn.Tanh(),
        activation_decoder=nn.PReLU(),
        device="cpu"
    ):
        super(GPVAE, self).__init__()

        # Define the q_net (encoder) and decoder
        self.q_net = q_net(input_size, hidden_layers_encoder, latent_dim, activation_encoder, logvar_range=(-8,-2))
        self.decoder = Decoder(latent_dim, hidden_layers_decoder, output_size, activation_decoder)
        
        # GP model for inference
        self.GP = GP_inf if GP_inf is not None else GP_inference(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        )
        self.device = device
        self.rec_loss = nn.MSELoss(reduction='mean')

    def encode(self, x, t):
        q_mu, q_logvar = self.q_net(x)
        q_var = torch.exp(q_logvar)
        mu_c, var_c, ll = self.GP(X=t, y=q_mu, q_var=q_var)
        return mu_c, var_c, ll, q_mu, q_var  

    def reparameterization(self, mean, var):
        std = torch.sqrt(var)
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, X):
        x, t = X[:, 0].unsqueeze(1), X[:, 1]  # x shape [N, 1], t shape [N]
        mu_c, var_c, ll, q_mu, q_var = self.encode(x, t)
        z = self.reparameterization(mu_c, var_c)
        x_hat = self.decode(z)
        return x_hat, mu_c, var_c, ll.reshape(1, 1, 1), q_mu, q_var


    def elbo(self, X, beta):
        x_hat, mu_c, var_c, ll, q_mu, q_var = self(X)
        gce = gauss_cross_entropy(mu_c, var_c, q_mu, q_var)
        ce = gce.mean()
        elbo_prior_KL = ll.mean() - ce
        elbo_recon = -self.rec_loss(x_hat, X[:, 0].unsqueeze(1))
        elbo = beta * elbo_prior_KL + elbo_recon
        return elbo, elbo_prior_KL, elbo_recon, -ce

    def train_step(self, X, opt, beta):
        opt.zero_grad()
        elb, KL, mse, ce = self.elbo(X, beta)
        mse = -mse  
        loss = -elb  
        loss.backward()  
        opt.step()  
        return {
            "loss": loss.item(),
            "KL": KL.item(),
            "mse": mse.item(),
            "ce": ce.item()
        }
        
    def val_step(self, X, opt, beta):
        self.eval()
        with torch.no_grad():
            elb, KL, mse, ce = self.elbo(X, beta)
            mse = -mse  
            loss = -elb   
            return {
                "loss": loss.item(),
                "KL": KL.item(),
                "mse": mse.item(),
                "ce": ce.item()
            } 

    def fit(self, train_loader, val_loader, opt, beta, num_epochs, t_uni=None, num_samples=50, ylim=(0,40), df=None, plot=False):
        # Lists to store metrics
        l_loss, KL, mse, ce = [], [], [], []
        val_loss, val_KL, val_mse, val_ce = [], [], [], []
        mean_q_mu_list, mean_q_var_list = [], []  # Lists for mean q_mu and q_var per epoch
        progress_interval = num_epochs // 20


        # check if beta is scalar or vector (for annealing)
        is_beta_scalar = np.isscalar(beta)
        if not is_beta_scalar and len(beta) < num_epochs:
            raise ValueError("The length of beta vector is shorter than the number of epochs.")


        for epoch in tqdm(range(num_epochs), desc="Training Progress"):
            
            current_beta = beta if is_beta_scalar else beta[epoch]
            # Training phase
            self.train()  # Set the model to training mode
            batch_losses, batch_KL, batch_mse, batch_ce = [], [], [], []
            q_mu_l, q_var_l = [], []  # Reset q_mu and q_var for each epoch

            for x_batch in train_loader:  # Iterate over each mini-batch
                # d_train = self.train_step(x_batch.squeeze(), opt, beta)  # Train step
                d_train = self.train_step(x_batch.squeeze(), opt, current_beta)
                batch_losses.append(d_train['loss'])
                batch_KL.append(d_train['KL'])
                batch_mse.append(d_train['mse'])
                batch_ce.append(d_train['ce'])

                # Compute q_mu and q_var using the model
                with torch.no_grad():
                    _, _, _, _, q_mu_i, q_var_i = self(x_batch.squeeze())  # Forward pass
                    q_mu_l.append(q_mu_i)
                    q_var_l.append(q_var_i)

            # Average the metrics over all mini-batches for the current epoch
            l_loss.append(np.mean(batch_losses))
            KL.append(np.mean(batch_KL))
            mse.append(np.mean(batch_mse))
            ce.append(np.mean(batch_ce))

            # Compute mean q_mu and q_var for the epoch
            mean_q_mu = sum(q_mu_l) / len(q_mu_l)
            mean_q_var = sum(q_var_l) / len(q_var_l)
            # mean_q_mu_list.append(mean_q_mu)
            # mean_q_var_list.append(mean_q_var)

            # Validation phase
            self.eval()  # Set the model to evaluation mode
            val_batch_losses, val_batch_KL, val_batch_mse, val_batch_ce = [], [], [], []
            with torch.no_grad():
                for val_batch in val_loader:
                    # d_val = self.val_step(val_batch.squeeze(), opt, beta)  # Validation step
                    d_val = self.val_step(val_batch.squeeze(), opt, current_beta)
                    val_batch_losses.append(d_val['loss'])
                    val_batch_KL.append(d_val['KL'])
                    val_batch_mse.append(d_val['mse'])
                    val_batch_ce.append(d_val['ce'])

            # Average the metrics over all validation mini-batches
            val_loss.append(np.mean(val_batch_losses))
            val_KL.append(np.mean(val_batch_KL))
            val_mse.append(np.mean(val_batch_mse))
            val_ce.append(np.mean(val_batch_ce))

            # Print progress at every 20%
            if (epoch + 1) % progress_interval == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch+1}:")
                print(f"  Training - Loss = {l_loss[-1]:.4f}, KL = {KL[-1]:.4f}, MSE = {mse[-1]:.4f}, CE = {ce[-1]:.4f}")
                print(f"  Validation - Loss = {val_loss[-1]:.4f}, KL = {val_KL[-1]:.4f}, MSE = {val_mse[-1]:.4f}, CE = {val_ce[-1]:.4f}")
                print(f"  Mean q_mu = {mean_q_mu.mean():.4f}, Mean q_var = {mean_q_var.mean():.4f}")
                if plot:
                    self.generate_and_plot(mu_s=mean_q_mu, t_uni=t_uni, num_samples=num_samples, ylim=ylim, df=df, plot=True)

        # Return metrics and mean q_mu, q_var over epochs
        return {
            'train_loss': l_loss,
            'train_KL': KL,
            'train_mse': mse,
            'train_ce': ce,
            'val_loss': val_loss,
            'val_KL': val_KL,
            'val_mse': val_mse,
            'val_ce': val_ce,
            'mean_q_mu': mean_q_mu_list,
            'mean_q_var': mean_q_var_list
        }

    # Method to generate and plot the data within GPVAE class
    def generate_and_plot(self, mu_s, t_uni, num_samples, ylim, df, plot=True, seed=None):
        """
        Generates data using the model's decoder and returns both the generated data and the plot.

        Args:
            mu_s: The mean values used for MVN sampling.
            t_uni: Time normalized to [0, 1] interval.
            num_samples: The number of samples to generate.
            ylim: The y-axis limit for the plots.
            df: Original data DataFrame with 'interval', 't', and 'theta' columns.
            plot: Whether to display the plots or not (default is True).

        Returns:
            generated_data: The generated data as a list.
            fig: The figure object of the generated plot.
        """
        if seed is not None:
            torch.manual_seed(seed)
        ns = num_samples
        generated_data = []
        fig = None

        for i in range(ns):
            with torch.no_grad():
                z = sample_MVN(mu_s, t_uni, self, num_samples=1)
                dec = self.decode(z)
                generated_data.append(dec)
                # axs[1].plot(t_uni, dec.cpu().numpy())

        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].set_ylim(ylim)
            axs[1].set_ylim(ylim)

            colors = plt.cm.jet(np.linspace(0, 1, len(df['interval'].unique())))
            for dec in generated_data:
                axs[1].plot(t_uni*10, dec.cpu().numpy())

            # Plot original data
            interval_indices = {interval: idx for idx, interval in enumerate(df['interval'].unique())}

            for i in df['interval'].unique():
                id = df['interval'] == i
                axs[0].plot(df[id]['t'].to_numpy(), df[id]['theta'].to_numpy(),
                            color=colors[interval_indices[i]])

            # Set titles and labels
            axs[1].set_title('Generated')
            axs[1].set_xlabel('time (min)')
            axs[0].set_title('Original')
            axs[0].set_xlabel('time (min)')
            axs[0].set_ylabel('theta')

            # Adjust layout
            plt.tight_layout()
            plt.show()

        return generated_data, fig

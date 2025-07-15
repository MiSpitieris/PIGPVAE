import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
import matplotlib.pyplot as plt
import pandas as pd
import copy
import random
from tqdm import tqdm

sys.path.append(os.path.abspath('..'))
from physics import NewtonsLaw
from GP import *
from MLP import *
from VAE_utils import gauss_cross_entropy, vae_loss, regularization_loss
from VAE_utils import q_net
from VAE_utils import q_net as q_net_phy
from VAE_utils import Decoder as Decoder_delta

class PIGPVAE(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_layers_encoder_phy=[10],
        hidden_layers_encoder_delta=[10],
        hidden_layers_decoder_delta=[10, 10],
        latent_dim=1,
        output_size=1,
        GP_inf=None,
        decoder_phy=NewtonsLaw(),
        activation_encoder_phy=nn.Tanh(),
        activation_encoder_delta=nn.ReLU(),
        activation_decoder_delta=nn.ReLU(),
        var_prior = torch.tensor(1.0),
        mu_prior = torch.tensor(.5),
        device="cpu",
        initial_alpha=0.5,
        reg_constraint=None
    ):
        super(PIGPVAE, self).__init__()

        # Define the q_net (encoder) and decoder
        self.q_net_phy = q_net_phy(input_size, hidden_layers_encoder_phy, latent_dim, activation_encoder_phy)
        self.q_net_delta = q_net(input_size+1, hidden_layers_encoder_delta, latent_dim, activation_encoder_delta, logvar_range=(-3, 3))
        self.decoder_delta = Decoder_delta(latent_dim+2+2, hidden_layers_decoder_delta, output_size, activation_decoder_delta)
        self.decode_phy = decoder_phy
        
        # GP model for inference
        self.GP = GP_inf if GP_inf is not None else GP_inference(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        )
        self.device = device
        self.rec_loss = nn.MSELoss(reduction='mean')
        self.mu_prior = mu_prior
        self.var_prior = var_prior
        # Define alpha as a trainable parameter
        self.raw_alpha = nn.Parameter(torch.log(torch.exp(torch.tensor(initial_alpha)) - 1))
        self.reg_constraint = reg_constraint
        
    def encode(self, x, t):
        q_mu_phy, q_logvar_phy = self.q_net_phy(x)
        q_var_phy = torch.exp(q_logvar_phy)
        k = self.reparameterization(q_mu_phy, q_var_phy)
        k = torch.clamp(k, min=0, max=10)
        
        Tk = torch.cat((x, k), dim=1)
        q_mu_delta, q_logvar_delta = self.q_net_delta(Tk)
        q_var_delta = torch.exp(q_logvar_delta)
        mu_c, var_c, ll = self.GP(X=t, y=q_mu_delta, q_var=q_var_delta)
        z_delta = self.reparameterization(mu_c, var_c)

        return mu_c, var_c, ll, q_mu_delta, q_var_delta, q_mu_phy, q_var_phy, k, z_delta
         
    def reparameterization(self, mean, var):
        std = torch.sqrt(var)
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        return z
    
    def decode_delta(self, x):
        return self.decoder_delta(x)
    
    def forward(self, X):
        T, Ts, t = X[:, 0].unsqueeze(1), X[:, 1].mean(), X[:, 2]
        T0 = T[0]
        mu_c, var_c, ll, q_mu_delta, q_var_delta, q_mu_phy, q_var_phy, k, z_delta = self.encode(T, t)
        
        x_hat_phy = self.decode_phy(T0, t, Ts, k.mean())
        x_hat_phy = x_hat_phy.unsqueeze(1)

        T0_broadcasted = torch.full_like(x_hat_phy, T0.item())
        Ts_broadcasted = torch.full_like(x_hat_phy, Ts.item())
        x_hat_phy__z_delta__k = torch.cat((x_hat_phy, z_delta, k, T0_broadcasted, Ts_broadcasted), dim=1)
        x_hat_delta = self.decode_delta(x_hat_phy__z_delta__k)
        x_hat = x_hat_phy + x_hat_delta
        
        return x_hat, x_hat_phy, x_hat_delta,  mu_c, var_c, ll.reshape(1, 1, 1), q_mu_delta, q_var_delta, q_mu_phy, q_var_phy 

    def elbo(self, X, beta_phy, beta_delta):
        # Convert beta_phy and beta_delta to torch tensors if they aren't already

        x_hat, x_hat_phy, x_hat_delta,  mu_c, var_c, ll, q_mu_delta, q_var_delta, q_mu_phy, q_var_phy = self(X)
        gce = gauss_cross_entropy(mu_c, var_c, q_mu_delta, q_var_delta)
        ce = gce.mean()
        elbo_prior_KL_delta = ll.mean() - ce
        KL_phy = -vae_loss(q_mu_phy, q_var_phy, self.mu_prior, self.var_prior)
        elbo_recon = -self.rec_loss(x_hat, X[:,0].unsqueeze(1))
        # reg_loss = -regularization_loss(x_hat_delta, torch.zeros_like(x_hat_delta))
        reg_loss = regularization_loss(x_hat, x_hat_phy)
        # Compute regularization loss and apply constraint
         # Apply  clipping if constraint is set
        if self.reg_constraint is not None:
            reg_loss = torch.clamp(reg_loss, max=self.reg_constraint)
        reg_loss = -reg_loss
        alpha = F.softplus(self.raw_alpha)
        elbo = elbo_recon + beta_phy * KL_phy + beta_delta * elbo_prior_KL_delta + alpha * reg_loss
        return elbo, elbo_prior_KL_delta, KL_phy, elbo_recon, ce, reg_loss
            
    def train_step(self, X, opt, beta_phy, beta_delta):
        opt.zero_grad()
        elb, KL_delta, KL_phy, mse, ce, reg_loss = self.elbo(X, beta_phy, beta_delta)
        mse = -mse
        loss = -elb
        loss.backward()
        opt.step()
        alpha = F.softplus(self.raw_alpha).item()
        
        return {
            "loss": loss.item(),
            "KL_delta": KL_delta.item(),
            "KL_phy": KL_phy.item(),
            "mse": mse.item(),
            "ce": -ce.item(),
            "reg_loss": reg_loss.item(),
            "alpha": alpha  # Track the trainable alpha
        }
        
    def val_step(self, X, beta_phy, beta_delta):
        self.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            elb, KL_delta, KL_phy, mse, ce, reg_loss = self.elbo(X, beta_phy, beta_delta)
            mse = -mse
            loss = -elb
            alpha = F.softplus(self.raw_alpha).item()
            return {
                "loss": loss.item(),
                "KL_delta": KL_delta.item(),
                "KL_phy": KL_phy.item(),
                "mse": mse.item(),
                "ce": -ce.sum().item(),
                "reg_loss": reg_loss.item(),
                "alpha": alpha  # Track the trainable alpha
        }

                    
    
    def fit(self, train_loader, val_loader, opt, num_epochs=1500, beta_phy=0.0001, beta_delta=0.0001, seed=0, 
            plot=False, df=None, ns=30, T0=None, Ts=None, t_uni=None, axs_ylim=(0, 35)):
        # Training configuration
        epochs = range(num_epochs)
        progress_interval = num_epochs // 20

        # Initialize lists to store training metrics
        l_loss, KL_delta, KL_phy, mse, ce, reg_loss = [], [], [], [], [], []
        mean_lengthscale_per_epoch, mean_outputscale_per_epoch = [], []
        alpha_values = []

        # Initialize lists to store validation metrics
        val_loss, val_KL_delta, val_KL_phy, val_mse, val_ce, val_reg_loss = [], [], [], [], [], []
        val_mean_lengthscale_per_epoch, val_mean_outputscale_per_epoch = [], []
        val_alpha_values = []

        # Seed for reproducibility
        torch.manual_seed(seed)

        # Training loop
        for epoch in tqdm(epochs, desc="Training Progress"):
            batch_losses, batch_KL_phy, batch_KL_delta, batch_mse, batch_ce, batch_reg_loss = [], [], [], [], [], []
            batch_lengthscales, batch_outputscales, batch_alpha = [], [], []
            q_mu_delta_l, q_var_delta_l, q_mu_phy_l, q_var_phy_l = [], [], [], []
            
            current_beta_phy = beta_phy[epoch] if isinstance(beta_phy, (list, np.ndarray)) else beta_phy
            current_beta_delta = beta_delta[epoch] if isinstance(beta_delta, (list, np.ndarray)) else beta_delta


            self.train()
            for x_batch in train_loader:
                # Perform a training step
                d_train = self.train_step(x_batch.squeeze(), opt, current_beta_phy, current_beta_delta)

                # Append training metrics
                batch_losses.append(d_train['loss'])
                batch_KL_phy.append(d_train['KL_phy'])
                batch_KL_delta.append(d_train['KL_delta'])
                batch_mse.append(d_train['mse'])
                batch_ce.append(d_train['ce'])
                batch_reg_loss.append(d_train['reg_loss'])
                batch_alpha.append(d_train['alpha'])

                # Extract and store kernel parameters
                with torch.no_grad():
                    lengthscale = self.GP.gp_model.covar_module.base_kernel.lengthscale
                    outputscale = self.GP.gp_model.covar_module.outputscale
                    epoch_mean_lengthscale = lengthscale.detach().cpu().mean().item()
                    epoch_mean_outputscale = outputscale.detach().cpu().mean().item()
                    batch_lengthscales.append(epoch_mean_lengthscale)
                    batch_outputscales.append(epoch_mean_outputscale)
                    _, _, _,  _, _, _, q_mu_delta_i, q_var_delta_i, q_mu_phy_i, q_var_phy_i = self(x_batch.squeeze())
                    q_mu_delta_l.append(q_mu_delta_i)
                    q_var_delta_l.append(q_var_delta_i)
                    q_mu_phy_l.append(q_mu_phy_i)
                    q_var_phy_l.append(q_var_phy_i)
                    
            mean_q_delta_mu = sum(q_mu_delta_l) / len(q_mu_delta_l)
            mean_q_delta_var = sum(q_var_delta_l) / len(q_var_delta_l)
            mean_q_mu_phy = sum(q_mu_phy_l) / len(q_mu_phy_l)
            mean_q_var_phy = sum(q_var_phy_l) / len(q_var_phy_l)

            # Validation loop
            self.eval()
            val_batch_losses, val_batch_KL_phy, val_batch_KL_delta, val_batch_mse, val_batch_ce, val_batch_reg_loss = [], [], [], [], [], []
            val_batch_lengthscales, val_batch_outputscales, val_batch_alpha = [], [], []

            with torch.no_grad():
                for x_batch_val in val_loader:
                    # Perform a validation step
                    d_val = self.val_step(x_batch_val.squeeze(), current_beta_phy, current_beta_delta)

                    # Append validation metrics
                    val_batch_losses.append(d_val['loss'])
                    val_batch_KL_phy.append(d_val['KL_phy'])
                    val_batch_KL_delta.append(d_val['KL_delta'])
                    val_batch_mse.append(d_val['mse'])
                    val_batch_ce.append(d_val['ce'])
                    val_batch_reg_loss.append(d_val['reg_loss'])
                    val_batch_alpha.append(d_val['alpha'])

                    # Extract and store kernel parameters for validation
                    lengthscale_val = self.GP.gp_model.covar_module.base_kernel.lengthscale
                    outputscale_val = self.GP.gp_model.covar_module.outputscale
                    epoch_mean_lengthscale_val = lengthscale_val.detach().cpu().mean().item()
                    epoch_mean_outputscale_val = outputscale_val.detach().cpu().mean().item()
                    val_batch_lengthscales.append(epoch_mean_lengthscale_val)
                    val_batch_outputscales.append(epoch_mean_outputscale_val)

            # Compute mean metrics for training
            l_loss.append(np.mean(batch_losses))
            KL_phy.append(-np.mean(batch_KL_phy))
            KL_delta.append(-np.mean(batch_KL_delta))
            mse.append(np.mean(batch_mse))
            ce.append(np.mean(batch_ce))
            reg_loss.append(-np.mean(batch_reg_loss))
            mean_lengthscale_per_epoch.append(np.mean(batch_lengthscales))
            mean_outputscale_per_epoch.append(np.mean(batch_outputscales))
            alpha_values.append(np.mean(batch_alpha))

            # Compute mean metrics for validation
            val_loss.append(np.mean(val_batch_losses))
            val_KL_phy.append(-np.mean(val_batch_KL_phy))
            val_KL_delta.append(-np.mean(val_batch_KL_delta))
            val_mse.append(np.mean(val_batch_mse))
            val_ce.append(np.mean(val_batch_ce))
            val_reg_loss.append(-np.mean(val_batch_reg_loss))
            val_mean_lengthscale_per_epoch.append(np.mean(val_batch_lengthscales))
            val_mean_outputscale_per_epoch.append(np.mean(val_batch_outputscales))
            val_alpha_values.append(np.mean(val_batch_alpha))

            # Print progress at specified intervals
            if (epoch + 1) % progress_interval == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch + 1}: "
                      f"Train Loss = {l_loss[-1]:.4f}, KL_phy = {KL_phy[-1]:.4f}, KL_delta = {KL_delta[-1]:.4f}, "
                      f"MSE = {mse[-1]:.4f}, CE = {ce[-1]:.4f}, Reg_Loss = {reg_loss[-1]:.4f}, Alpha = {alpha_values[-1]:.4f}")
                print(f"Validation Loss = {val_loss[-1]:.4f}, Val_KL_phy = {val_KL_phy[-1]:.4f}, Val_KL_delta = {val_KL_delta[-1]:.4f}, "
                      f"Val_MSE = {val_mse[-1]:.4f}, Val_CE = {val_ce[-1]:.4f}, Val_Reg_Loss = {val_reg_loss[-1]:.4f}, "
                      f"Val_Alpha = {val_alpha_values[-1]:.4f}")
                if plot:
                    unique_intervals = df['interval'].unique()
                    num_unique_intervals = len(unique_intervals)
                    mu_s = mean_q_delta_mu
                    # self.generate_and_plot(mu_s, mean_q_mu_phy, mean_q_var_phy, df, num_unique_intervals, ns, T0=None, Ts=None, t_uni=None, axs_ylim=(0, 35), plot=True)
                    self.generate_and_plot(mu_s, mean_q_mu_phy, mean_q_var_phy, df, num_unique_intervals, ns, T0=T0, Ts=Ts, t_uni=t_uni, axs_ylim=axs_ylim, plot=True)
                    


        # Return all metrics
        return {
            'epochs': list(epochs),
            'train_loss': l_loss,
            'val_loss': val_loss,
            'train_mse': mse,
            'val_mse': val_mse,
            'train_KL_phy': KL_phy,
            'val_KL_phy': val_KL_phy,
            'train_KL_delta': KL_delta,
            'val_KL_delta': val_KL_delta,
            'train_reg_loss': reg_loss,
            'val_reg_loss': val_reg_loss,
            'mean_lengthscale_per_epoch': mean_lengthscale_per_epoch,
            'mean_outputscale_per_epoch': mean_outputscale_per_epoch,
            'alpha_values': alpha_values
        }
        
    def generate_and_plot(self, mu_s, mean_q_mu_phy, mean_q_var_phy, df, num_unique_intervals, ns, T0=None, Ts=None, t_uni=None, axs_ylim=(0, 35), plot=True, seed=None):
        """
        Method to generate and plot both the generated and original data from the model.
        This method integrates the plot_generated_vs_original_data function into the class.
        
        Args:
            df: DataFrame containing the original data with intervals and temperature readings.
            num_unique_intervals: Number of unique intervals in df.
            ns: Number of samples to generate for the generated data.
            T0: Tensor for initial temperatures (optional, will be generated if not provided).
            Ts: Tensor for secondary temperatures (optional, will be generated if not provided).
            t_uni: Uniform time variable (optional, should be provided based on your setup).
            axs_ylim: Tuple defining the y-axis limits for the plots.
            plot: Boolean indicating whether to display the plot (True) or not (False).

        Returns:
            k_values: The list of `k` values used in the generated data.
            generated_data: A list containing the decoded generated data (combination of phy and delta).
            fig, axs: The matplotlib figure and axes objects (if plot=True). If plot=False, returns None for fig and axs.
        """
        
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
        # Initialize figure and axes for plotting
        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].set_ylim(axs_ylim)
            axs[1].set_ylim(axs_ylim)
        else:
            fig, axs = None, None

        # If T0 and Ts are not provided, generate them based on ns
        if T0 is None:
            min_T, max_T = 5, 35
            T0 = min_T + (max_T - min_T) * torch.rand(ns)
        
        if Ts is None:
            # Generate difference and Ts such that Ts > T0
            difference = 1 + (15 - 1) * torch.rand(ns)
            Ts = T0 - difference

        k_values = []  # To store the k values for each sample
        generated_data = []  # To store the generated decoded data

        # Set color map based on the number of unique intervals
        if num_unique_intervals <= 20:
            cmap = plt.get_cmap('tab20')
        else:
            cmap = plt.get_cmap('hsv')
        
        # Generate a list of colors based on the number of unique intervals
        colors = [cmap(i / num_unique_intervals) for i in range(num_unique_intervals)]
        interval_to_color = {interval: colors[idx] for idx, interval in enumerate(df['interval'].unique())}

        # Plot each unique interval from the original data (df) on axs[0]
        if plot:
            for interval in df['interval'].unique():
                id_mask = df['interval'] == interval
                axs[0].plot(
                    df[id_mask]['time_within_interval'].to_numpy(),
                    df[id_mask]['B.RTD1'].to_numpy(),
                    color=interval_to_color[interval],
                    label=f'Interval {interval}'
                )

            # Add legend if the number of intervals is manageable
            if num_unique_intervals <= 20:
                axs[0].legend()

        # Generate and plot data for each sample on axs[1]
        for i in range(ns):
            with torch.no_grad():
                # Sample z_delta and z_phy 
                z_delta = sample_MVN(mu_s, t_uni, self, num_samples=1)  
                z_phy = self.reparameterization(mean_q_mu_phy, mean_q_var_phy)  

                # Compute k and store its value
                k = z_phy.mean()
                k_values.append(k.item())

                # Decode phy and delta components
                dec_phy = self.decode_phy(T0[i], t_uni, Ts[i], k)
                dec_phy = dec_phy.unsqueeze(1)

                # Prepare tensors for concatenation
                T0_broadcasted = torch.full_like(dec_phy, T0[i])
                Ts_broadcasted = torch.full_like(dec_phy, Ts[i])
                z_delta__cond = torch.cat((dec_phy, z_delta, z_phy, T0_broadcasted, Ts_broadcasted), dim=1)

                # Decode delta and combine with phy to get the final decoded value
                dec_delta = self.decode_delta(z_delta__cond)
                dec = dec_phy + dec_delta
                generated_data.append(dec)  # Store the generated data

                # Plot the generated data on axs[1] if plot is True
                if plot:
                    axs[1].plot(t_uni * 240, dec)

        # If plot is enabled, add titles, labels, and finalize the layout
        if plot:
            axs[1].set_title('Generated')
            axs[1].set_xlabel('Time (min)')
            axs[0].set_title('Original')
            axs[0].set_xlabel('Time (min)')
            axs[0].set_ylabel('Temperature (°C)')

            # Finalize the layout
            # Adjust layout
            plt.tight_layout()
            plt.show()

        # Return the generated data and the plot objects (if plot=True). Otherwise, return None for fig and axs.
        return k_values, generated_data, fig, axs
        
    def generate_and_plot2(self, mu_s, mean_q_mu_phy, mean_q_var_phy, df, num_unique_intervals, ns, T0=None, Ts=None, t_uni=None, axs_ylim=(0, 35), plot=True, seed=None):
        """
        Method to generate and plot both the generated and original data from the model.
        This method integrates the plot_generated_vs_original_data function into the class.
        
        Args:
            df: DataFrame containing the original data with intervals and temperature readings.
            num_unique_intervals: Number of unique intervals in df.
            ns: Number of samples to generate for the generated data.
            T0: Tensor for initial temperatures (optional, will be generated if not provided).
            Ts: Tensor for secondary temperatures (optional, will be generated if not provided).
            t_uni: Uniform time variable (optional, should be provided based on your setup).
            axs_ylim: Tuple defining the y-axis limits for the plots.
            plot: Boolean indicating whether to display the plot (True) or not (False).

        Returns:
            k_values: The list of `k` values used in the generated data.
            generated_data: A list containing the decoded generated data (combination of phy and delta).
            fig, axs: The matplotlib figure and axes objects (if plot=True). If plot=False, returns None for fig and axs.
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            
        # Initialize figure and axes for plotting
        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].set_ylim(axs_ylim)
            axs[1].set_ylim(axs_ylim)
        else:
            fig, axs = None, None

        # If T0 and Ts are not provided, generate them based on ns
        if T0 is None:
            min_T, max_T = 5, 35
            T0 = min_T + (max_T - min_T) * torch.rand(ns)
        
        if Ts is None:
            # Generate difference and Ts such that Ts > T0
            difference = 1 + (15 - 1) * torch.rand(ns)
            Ts = T0 - difference

        k_values = []  # To store the k values for each sample
        generated_data = []  # To store the generated decoded data

        # Set color map based on the number of unique intervals
        if num_unique_intervals <= 20:
            cmap = plt.get_cmap('tab20')
        else:
            cmap = plt.get_cmap('hsv')
        
        # Generate a list of colors based on the number of unique intervals
        colors = [cmap(i / num_unique_intervals) for i in range(num_unique_intervals)]
        interval_to_color = {interval: colors[idx] for idx, interval in enumerate(df['interval'].unique())}

        # Plot each unique interval from the original data (df) on axs[0]
        if plot:
            for interval in df['interval'].unique():
                id_mask = df['interval'] == interval
                axs[0].plot(
                    df[id_mask]['time_within_interval'].to_numpy(),
                    df[id_mask]['B.RTD1'].to_numpy(),
                    color=interval_to_color[interval],
                    label=f'Interval {interval}'
                )

            # Add legend if the number of intervals is manageable
            if num_unique_intervals <= 20:
                axs[0].legend()

        # Generate and plot data for each sample on axs[1]
        for i in range(ns):
            with torch.no_grad():
                # Sample z_delta 
                z_delta = sample_MVN(mu_s, t_uni, self, num_samples=1)  # Assuming sample_MVN is defined

                # Sample k using the posterior and create z_phy 
                mu = mean_q_mu_phy.mean()
                var = mean_q_var_phy.mean()
                std = torch.sqrt(var)
                normal_dist = torch.distributions.Normal(mu, std)
                sample = -1.0
                while sample <= 0:
                    sample = normal_dist.sample()#.item()
                z_phy = torch.full(mean_q_mu_phy.shape, sample)

                # z_phy = self.reparameterization(mean_q_mu_phy, mean_q_var_phy)  # Assuming these are defined

                # Compute k and store its value
                k = sample
                k_values.append(k.item())

                # Decode phy and delta components
                dec_phy = self.decode_phy(T0[i], t_uni, Ts[i], k)
                dec_phy = dec_phy.unsqueeze(1)

                # Prepare tensors for concatenation
                T0_broadcasted = torch.full_like(dec_phy, T0[i])
                Ts_broadcasted = torch.full_like(dec_phy, Ts[i])
                z_delta__cond = torch.cat((dec_phy, z_delta, z_phy, T0_broadcasted, Ts_broadcasted), dim=1)

                # Decode delta and combine with phy to get the final decoded value
                dec_delta = self.decode_delta(z_delta__cond)
                dec = dec_phy + dec_delta
                generated_data.append(dec)  # Store the generated data

                # Plot the generated data on axs[1] if plot is True
                if plot:
                    axs[1].plot(t_uni * 240, dec)

        # If plot is enabled, add titles, labels, and finalize the layout
        if plot:
            axs[1].set_title('Generated')
            axs[1].set_xlabel('Time (min)')
            axs[0].set_title('Original')
            axs[0].set_xlabel('Time (min)')
            axs[0].set_ylabel('Temperature (°C)')

            # Finalize the layout
            # Adjust layout
            plt.tight_layout()
            plt.show()

        # Return the generated data and the plot objects (if plot=True). Otherwise, return None for fig and axs.
        return k_values, generated_data, fig, axs
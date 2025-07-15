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
from physics import SimplePendulumSolver
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
        decoder_phy=SimplePendulumSolver(),
        activation_encoder_phy=nn.Tanh(),
        activation_encoder_delta=nn.ReLU(),
        activation_decoder_delta=nn.ReLU(),
        var_prior=torch.tensor(1.0),
        mu_prior=torch.tensor(0.5),
        device="cpu",
        initial_alpha=0.5,
        trainable_alpha=True,
        reg_constraint=None
    ):
        super(PIGPVAE, self).__init__()

        # Define the q_net (encoder) and decoder
        self.q_net_phy = q_net_phy(input_size, hidden_layers_encoder_phy, latent_dim, activation_encoder_phy)
        self.q_net_delta = q_net(
            input_size + 1,
            hidden_layers_encoder_delta,
            latent_dim,
            activation_encoder_delta,
            # logvar_range=(-8, 4)
            logvar_range=(-8, 0)
        )
        self.decoder_delta = Decoder_delta(latent_dim + 2 + 1 + 1, hidden_layers_decoder_delta, output_size, activation_decoder_delta)
        self.decode_phy = decoder_phy

        # GP model for inference
        self.GP = GP_inf if GP_inf is not None else GP_inference(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        )
        self.device = device
        self.rec_loss = nn.MSELoss(reduction='mean')
        self.mu_prior = mu_prior
        self.var_prior = var_prior

        # Define alpha as a trainable parameter or fixed based on the input
        self.trainable_alpha = trainable_alpha
        if self.trainable_alpha:
            self.raw_alpha = nn.Parameter(torch.log(torch.exp(torch.tensor(initial_alpha)) - 1))
        else:
            self.alpha = torch.tensor(initial_alpha, requires_grad=False)

        self.reg_constraint = reg_constraint

    def encode(self, x, t):
        q_mu_phy, q_logvar_phy = self.q_net_phy(x)
        q_var_phy = torch.exp(q_logvar_phy)
        k = self.reparameterization(q_mu_phy, q_var_phy)
        k = torch.exp(k)

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
        T, t = X[:, 0].unsqueeze(1), X[:, 1]
        T0 = T[0]
        y0 = torch.tensor([T0, 0.0])
        mu_c, var_c, ll, q_mu_delta, q_var_delta, q_mu_phy, q_var_phy, k, z_delta = self.encode(T, t)

        x_hat_phy = self.decode_phy(y0, t, k.mean())
        x_hat_phy = x_hat_phy.unsqueeze(1)

        T0_broadcasted = torch.full_like(x_hat_phy, T0.item())
        t_scaled = t.unsqueeze(1)/t.max()
        x_hat_phy__z_delta__k = torch.cat((x_hat_phy, z_delta, k, T0_broadcasted, t_scaled), dim=1)
        x_hat_delta = self.decode_delta(x_hat_phy__z_delta__k)
        x_hat = x_hat_phy + x_hat_delta

        return (
            x_hat,
            x_hat_phy,
            x_hat_delta,
            mu_c,
            var_c,
            ll.reshape(1, 1, 1),
            q_mu_delta,
            q_var_delta,
            q_mu_phy,
            q_var_phy
        )

    def elbo(self, X, beta_phy, beta_delta):
        x_hat, x_hat_phy, x_hat_delta, mu_c, var_c, ll, q_mu_delta, q_var_delta, q_mu_phy, q_var_phy = self(X)
        gce = gauss_cross_entropy(mu_c, var_c, q_mu_delta, q_var_delta)
        ce = gce.mean()
        elbo_prior_KL_delta = ll.mean() - ce
        KL_phy = -vae_loss(q_mu_phy, q_var_phy, self.mu_prior, self.var_prior)
        elbo_recon = -self.rec_loss(x_hat, X[:, 0].unsqueeze(1))
        reg_loss_val = regularization_loss(x_hat, x_hat_phy)

        # Apply clipping if a constraint is set
        if self.reg_constraint is not None:
            reg_loss_val = torch.clamp(reg_loss_val, max=self.reg_constraint)
        reg_loss_val = -reg_loss_val

        # Use trainable or fixed alpha based on the trainable_alpha flag
        if self.trainable_alpha:
            alpha = F.softplus(self.raw_alpha)
        else:
            alpha = self.alpha

        elbo_val = elbo_recon + beta_phy * KL_phy + beta_delta * elbo_prior_KL_delta + alpha * reg_loss_val
        return elbo_val, elbo_prior_KL_delta, KL_phy, elbo_recon, ce, reg_loss_val

    def train_step(self, X, opt, beta_phy, beta_delta):
        opt.zero_grad()
        elb, KL_delta, KL_phy, mse, ce, reg_loss_val = self.elbo(X, beta_phy, beta_delta)
        mse = -mse
        loss = -elb
        loss.backward()
        opt.step()

        if self.trainable_alpha:
            alpha = F.softplus(self.raw_alpha).item()
        else:
            alpha = self.alpha.item()

        return {
            "loss": loss.item(),
            "KL_delta": KL_delta.item(),
            "KL_phy": KL_phy.item(),
            "mse": mse.item(),
            "ce": -ce.item(),
            "reg_loss": reg_loss_val.item(),
            "alpha": alpha  # Track the alpha value
        }

    def val_step(self, X, beta_phy, beta_delta):
        self.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            elb, KL_delta, KL_phy, mse, ce, reg_loss_val = self.elbo(X, beta_phy, beta_delta)
            mse = -mse
            loss = -elb
            if self.trainable_alpha:
                alpha = F.softplus(self.raw_alpha).item()
            else:
                alpha = self.alpha.item()

            return {
                "loss": loss.item(),
                "KL_delta": KL_delta.item(),
                "KL_phy": KL_phy.item(),
                "mse": mse.item(),
                "ce": -ce.sum().item(),
                "reg_loss": reg_loss_val.item(),
                "alpha": alpha  # Track the alpha value
            }

    def fit(self, train_loader, val_loader, opt, num_epochs=1500, beta_phy=0.0001, beta_delta=0.0001, seed=0,
            plot=False, df=None, ns=30, T0=None, t_uni=None, axs_ylim=(0, 35)):
        # Training configuration
        epochs = range(num_epochs)
        progress_interval = num_epochs // 20

        # Initialize lists to store training metrics
        l_loss, KL_delta_list, KL_phy_list = [], [], []
        mse_list, ce_list, reg_loss_list = [], [], []
        alpha_values = []

        # Initialize lists to store validation metrics
        val_loss, val_KL_delta_list, val_KL_phy_list = [], [], []
        val_mse_list, val_ce_list, val_reg_loss_list = [], [], []
        val_alpha_values = []

        # Seed for reproducibility
        torch.manual_seed(seed)

        # Training loop
        for epoch in tqdm(epochs, desc="Training Progress"):
            batch_losses, batch_KL_phy, batch_KL_delta = [], [], []
            batch_mse, batch_ce, batch_reg_loss = [], [], []
            batch_lengthscales, batch_outputscales, batch_alpha = [], [], []
            q_mu_delta_l, q_var_delta_l, q_mu_phy_l, q_var_phy_l = [], [], [], []

            if isinstance(beta_phy, (list, np.ndarray)):
                current_beta_phy = beta_phy[epoch]
            else:
                current_beta_phy = beta_phy

            if isinstance(beta_delta, (list, np.ndarray)):
                current_beta_delta = beta_delta[epoch]
            else:
                current_beta_delta = beta_delta

            self.train()
            for x_batch in train_loader:
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

                    _, _, _, _, _, _, q_mu_delta_i, q_var_delta_i, q_mu_phy_i, q_var_phy_i = self(x_batch.squeeze())
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
            val_batch_losses, val_batch_KL_phy, val_batch_KL_delta = [], [], []
            val_batch_mse, val_batch_ce, val_batch_reg_loss = [], [], []
            val_batch_lengthscales, val_batch_outputscales, val_batch_alpha = [], [], []

            with torch.no_grad():
                for x_batch_val in val_loader:
                    d_val = self.val_step(x_batch_val.squeeze(), current_beta_phy, current_beta_delta)

                    # Append validation metrics
                    val_batch_losses.append(d_val['loss'])
                    val_batch_KL_phy.append(d_val['KL_phy'])
                    val_batch_KL_delta.append(d_val['KL_delta'])
                    val_batch_mse.append(d_val['mse'])
                    val_batch_ce.append(d_val['ce'])
                    val_batch_reg_loss.append(d_val['reg_loss'])
                    val_batch_alpha.append(d_val['alpha'])


            # Compute mean metrics for training
            l_loss.append(np.mean(batch_losses))
            KL_phy_list.append(-np.mean(batch_KL_phy))
            KL_delta_list.append(-np.mean(batch_KL_delta))
            mse_list.append(np.mean(batch_mse))
            ce_list.append(np.mean(batch_ce))
            reg_loss_list.append(-np.mean(batch_reg_loss))
            alpha_values.append(np.mean(batch_alpha))

            # Compute mean metrics for validation
            val_loss.append(np.mean(val_batch_losses))
            val_KL_phy_list.append(-np.mean(val_batch_KL_phy))
            val_KL_delta_list.append(-np.mean(val_batch_KL_delta))
            val_mse_list.append(np.mean(val_batch_mse))
            val_ce_list.append(np.mean(val_batch_ce))
            val_reg_loss_list.append(-np.mean(val_batch_reg_loss))
            val_alpha_values.append(np.mean(val_batch_alpha))

            # Print progress at specified intervals
            if (epoch + 1) % progress_interval == 0 or epoch == num_epochs - 1:
                print(
                    f"Epoch {epoch + 1}: "
                    f"Train Loss = {l_loss[-1]:.4f}, KL_phy = {KL_phy_list[-1]:.4f}, KL_delta = {KL_delta_list[-1]:.4f}, "
                    f"MSE = {mse_list[-1]:.4f}, CE = {ce_list[-1]:.4f}, Reg_Loss = {reg_loss_list[-1]:.4f}, "
                    f"Alpha = {alpha_values[-1]:.4f}"
                )
                print(
                    f"Validation Loss = {val_loss[-1]:.4f}, Val_KL_phy = {val_KL_phy_list[-1]:.4f}, "
                    f"Val_KL_delta = {val_KL_delta_list[-1]:.4f}, Val_MSE = {val_mse_list[-1]:.4f}, "
                    f"Val_CE = {val_ce_list[-1]:.4f}, Val_Reg_Loss = {val_reg_loss_list[-1]:.4f}, "
                    f"Val_Alpha = {val_alpha_values[-1]:.4f}"
                )
                if plot:
                    unique_intervals = df['interval'].unique()
                    num_unique_intervals = len(unique_intervals)
                    mu_s = mean_q_delta_mu
                    self.generate_and_plot(
                        mu_s,
                        mean_q_mu_phy,
                        mean_q_var_phy,
                        df,
                        num_unique_intervals,
                        ns,
                        T0=T0,
                        t_uni=t_uni,
                        axs_ylim=axs_ylim,
                        plot=True
                    )

        # Return all metrics
        return {
            'epochs': list(epochs),
            'train_loss': l_loss,
            'val_loss': val_loss,
            'train_mse': mse_list,
            'val_mse': val_mse_list,
            'train_KL_phy': KL_phy_list,
            'val_KL_phy': val_KL_phy_list,
            'train_KL_delta': KL_delta_list,
            'val_KL_delta': val_KL_delta_list,
            'train_reg_loss': reg_loss_list,
            'val_reg_loss': val_reg_loss_list,
            'alpha_values': alpha_values
        }

    def generate_and_plot(
        self, mu_s, mean_q_mu_phy, mean_q_var_phy, df, num_unique_intervals, ns,
        T0=None, t_uni=None, axs_ylim=(-1.5, 1.5), plot=True, seed=None
    ):
        """
        Method to generate and plot both the generated and original data from the model,
        showing 4 subplots side by side:
            1) Original data
            2) dec (combined)
            3) dec_phy
            4) dec_delta

        Args:
            df: DataFrame containing the original data with intervals and temperature readings.
            num_unique_intervals: Number of unique intervals in df.
            ns: Number of samples to generate for the generated data.
            T0: Tensor for initial temperatures (optional, will be generated if not provided).
            t_uni: Uniform time variable (optional).
            axs_ylim: Tuple defining the y-axis limits for the plots.
            plot: Boolean indicating whether to display the plot (True) or not (False).
            seed: Random seed for reproducibility (optional).

        Returns:
            k_values: The list of `k` values used in the generated data.
            generated_data: A list containing the decoded generated data (combination of phy and delta).
            fig, axs: The matplotlib figure and axes objects (if plot=True). 
                      If plot=False, returns None for fig and axs.
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        # Initialize figure and axes for plotting
        if plot:
            # 4 subplots in 1 row, side by side
            fig, axs = plt.subplots(1, 4, figsize=(24, 6))
            for ax in axs:
                ax.set_ylim(axs_ylim)
        else:
            fig, axs = None, None

        # If T0 is not provided, generate them based on ns
        if T0 is None:
            min_T, max_T = -1, 1
            T0 = min_T + (max_T - min_T) * torch.rand(ns)

        k_values = []        # To store the k values for each sample
        generated_data = []  # To store the final decoded data (dec)

        # Prepare color map for original data
        if num_unique_intervals <= 20:
            cmap = plt.get_cmap('tab20')
        else:
            cmap = plt.get_cmap('hsv')

        colors = [cmap(i / num_unique_intervals) for i in range(num_unique_intervals)]
        interval_to_color = {
            interval: colors[idx]
            for idx, interval in enumerate(df['interval'].unique())
        }

        # 1) Plot each unique interval (original data) on axs[0]
        if plot:
            for interval in df['interval'].unique():
                id_mask = df['interval'] == interval
                axs[0].plot(
                    df[id_mask]['t'].to_numpy(),
                    df[id_mask]['theta'].to_numpy(),
                    color=interval_to_color[interval],
                    label=f'Interval {interval}'
                )

        # Generate and plot data for each sample
        for i in range(ns):
            with torch.no_grad():
                # Sample z_delta and z_phy
                z_delta = sample_MVN(mu_s, t_uni, self, num_samples=1)
                z_phy = self.reparameterization(mean_q_mu_phy, mean_q_var_phy)

                # Compute k and store its value
                k = z_phy.mean()
                k = torch.exp(k)
                k_values.append(k.item())

                # Decode phy
                dec_phy = self.decode_phy(
                    torch.tensor([T0[i], 0.0]),
                    t_uni,
                    k
                )
                dec_phy = dec_phy.unsqueeze(1)  # shape [len(t_uni), 1]

                # Prepare condition for delta
                T0_broadcasted = torch.full_like(dec_phy, T0[i])
                t_scaled = t_uni.unsqueeze(1)/t_uni.max()
                z_delta__cond = torch.cat((dec_phy, z_delta, z_phy, T0_broadcasted,t_scaled), dim=1)

                # Decode delta
                dec_delta = self.decode_delta(z_delta__cond)

                # Combine
                dec = dec_phy + dec_delta
                generated_data.append(dec)

                # Plot dec (combined), dec_phy, and dec_delta on respective axes
                if plot:
                    label_dec = "dec" if i == 0 else None
                    label_dec_phy = "dec_phy" if i == 0 else None
                    label_dec_delta = "dec_delta" if i == 0 else None

                    axs[1].plot(t_uni, dec.squeeze(), label=label_dec)
                    axs[2].plot(t_uni, dec_phy.squeeze(), label=label_dec_phy)
                    axs[3].plot(t_uni, dec_delta.squeeze(), label=label_dec_delta)

        # If plotting is enabled, add titles, labels, legends
        if plot:
            axs[0].set_title('Original Data')
            axs[0].set_xlabel('Time')
            axs[0].set_ylabel('Theta')

            axs[1].set_title('Combined (dec = phy + delta)')
            axs[2].set_title('Physics-based (dec_phy)')
            axs[3].set_title('Delta (dec_delta)')

            axs[1].set_xlabel('Time')
            axs[2].set_xlabel('Time')
            axs[3].set_xlabel('Time')

            axs[1].legend()
            axs[2].legend()
            axs[3].legend()

            plt.tight_layout()
            plt.show()

        return k_values, generated_data, fig, axs

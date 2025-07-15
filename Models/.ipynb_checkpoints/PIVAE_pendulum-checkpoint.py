import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import copy
from tqdm import tqdm

sys.path.append(os.path.abspath('..'))
from physics import SimplePendulumSolver
from VAE_utils import q_net
from VAE_utils import vae_loss as KL_div

class PIVAE(nn.Module):
    def __init__(self, 
                 input_size=1, 
                 hidden_layers=[10], 
                 latent_dim=1, 
                 output_size=1, 
                 decoder=SimplePendulumSolver(), 
                 activation=nn.LeakyReLU(),
                 mu_prior=torch.tensor(0.0),
                 var_prior=torch.tensor(1.0),
                 device="cpu"):
        super(PIVAE, self).__init__()

        self.q_net = q_net(input_size, hidden_layers, latent_dim, activation)
        self.decode = decoder
        self.rec_loss = nn.MSELoss(reduction='mean')
        self.mu_prior = mu_prior
        self.var_prior = var_prior
        
        
    def encode(self, x):
        q_mu, q_logvar = self.q_net(x)
        q_var = torch.exp(q_logvar)
        return q_mu, q_var
        
    def reparameterization(self, mean, var):
        std = torch.sqrt(var)
        epsilon = torch.randn_like(var)#.to(device)      
        z = mean + std*epsilon
        return z      

    def forward(self, X):
        T, t = X[:, 0].unsqueeze(1), X[:, 1]
        y0 = torch.tensor([T[0], 0.0])
        q_mu, q_var = self.encode(T)
        z = self.reparameterization(q_mu, q_var)
        k = torch.exp(z)
        x_hat = self.decode(y0, t, k.mean())

        return x_hat, q_mu, q_var
        
    def elbo(self, X, beta_phy):
        x_hat, q_mu, q_var = self(X)
        KL_phy = -KL_div(q_mu, q_var, self.mu_prior, self.var_prior)
        elbo_recon = -self.rec_loss(x_hat, X[:, 0])
        elbo = elbo_recon + beta_phy * KL_phy 
        return elbo, KL_phy, elbo_recon
        
    def train_step(self, X, opt, beta_phy):
        opt.zero_grad()
        elb, kl_div, recon_loss = self.elbo(X, beta_phy)
        total_loss=-elb
        mse=-recon_loss
        total_loss.backward()
        opt.step()
        return {
            "total_loss": total_loss.item(),
            "KL": -kl_div.item(),
            "mse": mse.item()
        }
        
    def fit(self, dataloader, opt, beta_phy, num_epochs):
        l_loss, KL, mse = [], [], []
        epoch_progress = max(1, num_epochs // 5)

        for epoch in range(num_epochs):
            batch_total_loss, batch_KL, batch_mse = [], [], []

            for x_batch in dataloader:
                x_batch = x_batch.squeeze(0)  # Remove batch dimension if necessary
                d_train = self.train_step(x_batch, opt, beta_phy)
                batch_total_loss.append(d_train['total_loss'])
                batch_KL.append(d_train['KL'])
                batch_mse.append(d_train['mse'])
            
            # Log epoch metrics
            l_loss.append(np.mean(batch_total_loss))
            KL.append(np.mean(batch_KL))
            mse.append(np.mean(batch_mse))

            if (epoch + 1) % epoch_progress == 0 or epoch == num_epochs - 1:
                print(f"Progress: Epoch {epoch+1}/{num_epochs} | Loss={l_loss[-1]:.4f}, KL={KL[-1]:.4f}, MSE={mse[-1]:.4f}")
               

        return l_loss, KL, mse
        
    def reconstruct(self, dataloader):
        reconstructed_data = []
        original_data = []
        with torch.no_grad():  # No gradient calculation for inference
            for x_batch in dataloader:
                x_batch = x_batch.squeeze(0)  # Adjust dimensions if necessary
                x_hat, _, _ = self(x_batch)  # Forward pass
                reconstructed_data.append(x_hat.cpu().numpy())  # Store reconstructed data
                original_data.append(x_batch[:, 0].unsqueeze(1).cpu().numpy())  # Store original data (assuming first column is target)
        reconstructed_data = np.vstack(reconstructed_data)
        original_data = np.hstack(original_data)
    
        return original_data.T, reconstructed_data
        
    def generate(self, mu_post, var_post, y0, t):
        gen_data, k_samples = [], []
        for i in range(len(y0)):
            z = self.reparameterization(mu_post, var_post)
            k = torch.exp(z)
            gen = self.decode(torch.tensor([y0[i], 0.0]), t, k.mean())
            
            k_samples.append(k)
            gen_data.append(gen.unsqueeze(1))
            
        return gen_data, k_samples
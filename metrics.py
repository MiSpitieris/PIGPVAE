import torch
import numpy as np
from scipy.stats import wasserstein_distance
from itertools import product
import tsgm

class SyntheticDataMetrics:
    def __init__(self):
        # maximum mean discrepancy (mmd)
        self.mmd_metric = tsgm.metrics.MMDMetric()

    
    # marginal distribution metrcicode  from https://arxiv.org/pdf/2111.01207
    @staticmethod
    def compute_marginal_metric_from_tensors(real_data, synthetic_data):
        """
        Compute the Marginal Distribution Metric using Wasserstein distance.
        """
        assert len(real_data) == len(synthetic_data), "Real and synthetic data lists must have the same length."
        
        total_distance = 0.0
        count = 0

        for real_tensor, synthetic_tensor in zip(real_data, synthetic_data):
            assert real_tensor.shape == synthetic_tensor.shape, "Tensor shapes must match."
            
            real_samples = real_tensor.squeeze(-1).numpy()
            synthetic_samples = synthetic_tensor.squeeze(-1).numpy()
            
            distance = wasserstein_distance(real_samples, synthetic_samples)
            total_distance += distance
            count += 1
        
        return total_distance / count

    @staticmethod
    def compute_correlation(X, Y):
        """Computes Pearson correlation between two tensors X and Y."""
        X_mean, Y_mean = X.mean(), Y.mean()
        numerator = ((X - X_mean) * (Y - Y_mean)).sum()
        denominator = torch.sqrt(((X - X_mean) ** 2).sum() * ((Y - Y_mean) ** 2).sum())
        return numerator / (denominator + 1e-8)
        
    # Correlation metric from https://arxiv.org/pdf/2111.01207
    @staticmethod
    def correlation_metric(original, synthetic):
        """
        Computes the correlation metric as the sum of absolute differences
        between correlations of original and synthetic data.
        """
        T = len(original)
        d = original[0].size(1)
        
        total_diff = 0.0
        for s, t in product(range(T), repeat=2):
            for i, j in product(range(d), repeat=2):
                rho_original = SyntheticDataMetrics.compute_correlation(original[s][:, i], original[t][:, j])
                rho_synthetic = SyntheticDataMetrics.compute_correlation(synthetic[s][:, i], synthetic[t][:, j])
                total_diff += abs(rho_original - rho_synthetic)
        
        return total_diff.item()
    # Marginal Distribution Difference from https://arxiv.org/abs/2309.03755
    @staticmethod
    def compute_mdd(original_list, generated_list, bins=20):
        """
        Computes the Marginal Distribution Difference (MDD) between original and generated time series.
        """
        if isinstance(original_list, torch.Tensor):
            original_list = [original_list]
        if isinstance(generated_list, torch.Tensor):
            generated_list = [generated_list]

        original_data = torch.cat([tensor for tensor in original_list]).flatten().numpy()
        generated_data = torch.cat([tensor for tensor in generated_list]).flatten().numpy()

        hist_orig, bin_edges = np.histogram(original_data, bins=bins, density=True)
        hist_gen, _ = np.histogram(generated_data, bins=bin_edges, density=True)

        hist_orig = hist_orig / np.sum(hist_orig)
        hist_gen = hist_gen / np.sum(hist_gen)

        return np.mean(np.abs(hist_orig - hist_gen))
    # maximum mean discrepancy (mmd)
    def compute_mmd(self, real_data, synthetic_data):
        """
        Computes the Maximum Mean Discrepancy (MMD) between real and synthetic data.
        """
        return self.mmd_metric(real_data, synthetic_data)

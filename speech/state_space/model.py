import numpy as np
from torch import nn
import torch

class FactorAnalysis(nn.Module):
    
    def __init__(self, noise_prob, noise_cov : dict(), factors):
        super(FactorAnalysis, self).__init__()
        self.noise_prob = noise_prob
        self.noise_cov = noise_cov
        self.factors = torch.from_numpy(factors)

    def epsilon(self):
        eps = np.stack([self.noise_prob * np.sqrt(n**2) for model,n in self.noise_cov.items()])
        return torch.from_numpy(eps)

    def forward(self, z):
        return torch.bmm(self.factors, z) + self.epsilon()
    
def save_onnx(famodel, factors, noise_prob, noise_cov, filename="factor_analysis.onnx"):

    print("Saving onnx file for factor analysis")

    famodel = FactorAnalysis(noise_prob, noise_cov, np.array(list(factors.values())))
    # latent model for 60 states and 24 actors
    latent = np.random.normal(0, 1.0, (60,24,24))
    latent = torch.from_numpy(latent)
    
    d = famodel(latent)

    torch.onnx.export(famodel, latent, filename, 
        verbose=True, input_names=['latent'], output_names=['data'])
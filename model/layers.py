import torch
import torch.nn as nn

class RMSNorm(nn.Module):

    def __init__(self, d_model, eps = 0.00001):
        super().__init__()
        self.eps = eps 
        self.alpha = nn.Parameter(torch.ones(d_model))

    def forward(self,x):
        rms = (x*x).mean(dim=-1, keepdim=True) + self.eps
        return (x / torch.sqrt(rms)) * self.alpha
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


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff_glu=None):
        super().__init__()
        d_ff_glu = d_ff_glu or int((8/3) * d_model)
        self.w_in = nn.Linear(d_model, 2 * d_ff_glu)
        self.w_out = nn.Linear(d_model, d_ff_glu, d_model)

    def forward(self, x):
        a,b = self.w_in(x).chunk(2, dim=-1)
        gated = nn.SiLU(x) * b
        return self.w_out(gated)
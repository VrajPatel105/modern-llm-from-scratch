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
        self.silu = nn.SiLU()
        self.w_in = nn.Linear(d_model, 2 * d_ff_glu)
        self.w_out = nn.Linear(d_ff_glu, d_model)

    def forward(self, x):
        a, b = self.w_in(x).chunk(2, dim=-1)
        gated = self.silu(a) * b
        return self.w_out(gated)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: int = 10000):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Step 1: frequencies, one per dimension-pair -> shape (head_dim/2,)
        i = torch.arange(0, head_dim, 2).float()
        theta = base ** (-i / head_dim)

        # Step 2: precompute angles for every position up to max_seq_len
        positions = torch.arange(0, max_seq_len).float()
        angles = torch.outer(positions, theta)  # shape (max_seq_len, head_dim/2)

        # cache cos/sin tables so we don't recompute them every forward call
        self.register_buffer('cos_cached', torch.cos(angles))  # (max_seq_len, head_dim/2)
        self.register_buffer('sin_cached', torch.sin(angles))  # (max_seq_len, head_dim/2)

    def _apply_rotation(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_heads, seq_len, head_dim)
        # split into even/odd dimension pairs
        x_even = x[..., 0::2]  # (batch, num_heads, seq_len, head_dim/2)
        x_odd  = x[..., 1::2]  # (batch, num_heads, seq_len, head_dim/2)

        # cos/sin: (seq_len, head_dim/2) -> broadcast to (1, 1, seq_len, head_dim/2)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos

        # interleave back into original even/odd positions
        x_out = torch.empty_like(x)
        x_out[..., 0::2] = x_rot_even
        x_out[..., 1::2] = x_rot_odd
        return x_out

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        # q, k: (batch, num_heads, seq_len, head_dim)
        cos = self.cos_cached[positions]  # (seq_len, head_dim/2)
        sin = self.sin_cached[positions]  # (seq_len, head_dim/2)

        q_rotated = self._apply_rotation(q, cos, sin)
        k_rotated = self._apply_rotation(k, cos, sin)
        return q_rotated, k_rotated
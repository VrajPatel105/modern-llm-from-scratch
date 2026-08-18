import torch
import torch.nn.functional as F

from model.layers import RMSNorm  # Rename to RMSNorm if possible

torch.manual_seed(0)

x = torch.randn(2, 5, 64)
dim = x.shape[-1]
eps = 1e-6

layer_norm = RMSNorm(dim, eps=eps)

torch_output = F.rms_norm(
    x,
    normalized_shape=(dim,),
    weight=layer_norm.alpha,
    eps=eps,
)

custom_output = layer_norm(x)

torch.testing.assert_close(
    custom_output,
    torch_output,
    rtol=1e-5,
    atol=1e-6,
)

print("Custom RMSNorm matches PyTorch.")

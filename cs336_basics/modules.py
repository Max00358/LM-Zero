import torch
from torch import nn

class Linear(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, std=0.02) # truncated normal distribution w/ std 0.02
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, std=0.02) # truncated normal distribution w/ std 0.02
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]

class RMSNorm(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        eps: float = 1e-5, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32) # prevent float16 overflow

        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        rms_norm = (x / rms) * self.weight
        return rms_norm.to(in_type)

# position-wise feedforward network
class SwiGLU(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        d_ff: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # nn.Linear(in, out) creates weight matrix of shape (out, in)
        self.w1 = nn.Parameter(
            torch.empty(d_ff, d_model, device=device, dtype=dtype)
        )
        self.w2 = nn.Parameter(
            torch.empty(d_model, d_ff, device=device, dtype=dtype)
        )
        self.w3 = nn.Parameter(
            torch.empty(d_ff, d_model, device=device, dtype=dtype)
        )

        # init weights
        nn.init.trunc_normal_(self.w1, std=0.02)
        nn.init.trunc_normal_(self.w2, std=0.02)
        nn.init.trunc_normal_(self.w3, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        # w1: (d_ff, d_model)
        w1x = x @ self.w1.T
        silu = torch.sigmoid(w1x) * w1x
        w3x = x @ self.w3.T
        
        return (silu * w3x) @ self.w2.T

class RoPE(nn.Module):
    def __init__(
        self, 
        theta: float, 
        d_k: int, 
        max_seq_len: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        k = torch.arange(0, d_k, 2, device=device, dtype=dtype)
        inverse_k = 1.0 / (theta ** (k / d_k))

        positions = torch.arange(0, max_seq_len, device=device, dtype=dtype) # index
        angles = torch.outer(positions, inverse_k)

        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        self.register_buffer("cos_angles", cos_angles)
        self.register_buffer("sin_angles", sin_angles)
    
    def forward(
        self, 
        x: torch.Tensor, 
        token_positions: torch.Tensor
    ) -> torch.Tensor:
        cos = self.cos_angles[token_positions]
        sin = self.sin_angles[token_positions]
        
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # apply rotation on each pair (x_even, x_odd)
        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_even * sin + x_odd * cos
        x_rot = torch.stack([x_even_rot, x_odd_rot], dim=-1).flatten(-2, -1)

        return x_rot
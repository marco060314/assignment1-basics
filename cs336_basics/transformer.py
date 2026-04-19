import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features 
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(
            out_features,
            in_features,
            device=device,
            dtype=dtype
        ))
        nn.init.trunc_normal(self.weight)

    def forward(self, x):
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device = device,
                dtype = dtype
            )
        )

        torch.nn.init.trunc_normal(self.weight)

    def forward(self, token_ids):
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(
                d_model,
                device = device,
                dtype = dtype
            )
        )



    def forward(self, x):
        start_type = x.dtype
        x = x.to(torch.float32)

        rms = torch.mean((x ** 2), dim = -1, keepdim = True)
        norm = x / (torch.sqrt(rms + self.eps))
        out = norm * self.weight
        return out.to(start_type)


class SwiGLU(nn.Module):
    def __init__(self, d_model, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        df_ff = round((8/3 * d_model)/ 64) * 64

        self.w1 = Linear(
            in_features=d_model,
            out_features=d_ff,
            device=device,
            dtype=dtype,
        )

        self.w3 = Linear(
            in_features=d_model,
            out_features=d_ff,
            device=device,
            dtype=dtype,
        )

        self.w2 = Linear(
            in_features=d_ff,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        a = self.w1(x)
        b = self.w3(x)
        silu_a = a * torch.sigmoid(a)
        gated = silu_a * b
        out = self.w2(gated)
        return out

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        assert d_k % 2 == 0
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        pair_indices = torch.arange(d_k // 2, device=device, dtype=torch.float32)
        angle_rates = angle_rates = theta ** (-(2 * pair_indices) / d_k)

        angles = positions * angle_rates
        #precompute sin and cos tables
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # store as buffers, not parameters
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    
    def forward(self, x, token_positions):
        """
        x: (..., seq_len, d_k)
        token_positions: (..., seq_len)
        returns: same shape as x
        """

        # split into even and odd coordinates
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # use token_positions to gather the correct cos/sin rows
        # result should broadcast to x_even/x_odd shape
        cos = self.cos[token_positions]
        sin = self.sin[token_positons]

        # rotate each pair
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd  = x_even * sin + x_odd * cos

        # stitch them back together to shape (..., seq_len, d_k)
        out[..., 0::2] = rotated_even
        out[..., 1::2] = rotated_odd
        return out

    




        
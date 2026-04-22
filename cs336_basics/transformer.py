import math
import torch
import torch.nn as nn
from einops import rearrange


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )
        nn.init.trunc_normal_(self.weight)

    def forward(self, x):
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype,
            )
        )

        nn.init.trunc_normal_(self.weight)

    def forward(self, token_ids):
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, x):
        start_type = x.dtype
        x = x.to(torch.float32)

        rms = torch.mean(x ** 2, dim=-1, keepdim=True)
        norm = x / torch.sqrt(rms + self.eps)
        out = norm * self.weight

        return out.to(start_type)


class SwiGLU(nn.Module):
    def __init__(self, d_model, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        d_ff = round((8 / 3 * d_model) / 64) * 64

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
        angle_rates = theta ** (-(2 * pair_indices) / d_k)

        angles = positions[:, None] * angle_rates[None, :]

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x, token_positions):
        """
        x: (batch, num_heads, seq_len, d_k)
        token_positions: (batch, seq_len)
        """
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        cos = self.cos[token_positions].unsqueeze(1)  # (batch, 1, seq_len, d_k/2)
        sin = self.sin[token_positions].unsqueeze(1)  # (batch, 1, seq_len, d_k/2)

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[..., 0::2] = rotated_even
        out[..., 1::2] = rotated_odd
        return out


def softmax(x, dim):
    max_vals = torch.max(x, dim=dim, keepdim=True).values
    shifted = x - max_vals
    exp_vals = torch.exp(shifted)
    denom = torch.sum(exp_vals, dim=dim, keepdim=True)
    return exp_vals / denom


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]

    scores = Q @ K.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = torch.where(
            mask,
            scores,
            torch.tensor(-1e9, device=scores.device, dtype=scores.dtype),
        )

    attn_probs = softmax(scores, dim=-1)
    out = attn_probs @ V
    return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, theta=10000.0, device=None, dtype=None):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = RotaryPositionalEmbedding(
            theta=theta,
            d_k=self.d_k,
            max_seq_len=max_seq_len,
            device=device,
        )

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        returns: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, seq_len)

        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )

        attn_out = scaled_dot_product_attention(q, k, v, mask=mask)

        attn_out = rearrange(attn_out, "b h s d -> b s (h d)")

        out = self.out_proj(attn_out)
        return out


def cross_entropy(inputs, targets):

    max_vals = torch.max(inputs, dim=-1, keepdim=True).values
    shifted = inputs - max_vals
    logsumexp = torch.log(torch.sum(torch.exp(shifted), dim=-1))
    target_logits = shifted.gather(
        dim=-1,
        index=targets.unsqueeze(-1)
    ).squeeze(-1)
    loss = -target_logits + logsumexp
    return loss.mean()

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                m = state["m"]
                v = state["v"]

                state["step"] += 1
                t = state["step"]

                # update first and second moments
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # bias correction
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # decoupled weight decay
                p.mul_(1 - lr * weight_decay)

                # parameter update
                p.addcdiv_(m_hat, torch.sqrt(v_hat) + eps, value=-lr)

        return loss


def get_adamw_cls():
    return AdamW




def run_get_lr_cosine_schedule(
    it,
    max_learning_rate,
    min_learning_rate,
    warmup_iters,
    cosine_cycle_iters,
):
    """
    it = current iteration t
    alpha_max = max_learning_rate
    alpha_min = min_learning_rate
    T_w = warmup_iters
    T_c = cosine_cycle_iters
    """

    # 1. warmup phase
    if it < warmup_iters:
        return max_learning_rate * (it + 1) / warmup_iters

    # 2. after cosine schedule ends
    if it >= cosine_cycle_iters:
        return min_learning_rate

    # 3. cosine decay phase
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)

    cosine_term = 0.5 * (1 + math.cos(math.pi * progress))

    lr = min_learning_rate + cosine_term * (
        max_learning_rate - min_learning_rate
    )

    return lr

def run_gradient_clipping(parameters, max_l2_norm):
    """
    parameters: iterable of nn.Parameter
    max_l2_norm: maximum allowed global L2 norm

    modifies parameter.grad in-place
    """

    params = [p for p in parameters if p.grad is not None]

    if len(params) == 0:
        return

    # compute total gradient norm
    total_norm_sq = torch.zeros(
        (),
        device=params[0].grad.device,
        dtype=params[0].grad.dtype
    )

    for p in params:
        total_norm_sq += torch.sum(p.grad ** 2)

    total_norm = torch.sqrt(total_norm_sq)

    # clip only if too large
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)

        for p in params:
            p.grad.mul_(scale)


def run_get_batch(dataset, batch_size, context_length, device):
    """
    dataset: 1D numpy array of token ids
    returns:
        x: (batch_size, context_length)
        y: (batch_size, context_length)
    """

    # random starting indices
    starts = np.random.randint(
        0,
        len(dataset) - context_length,
        size=batch_size
    )

    # input sequences
    x = np.stack([
        dataset[i : i + context_length]
        for i in starts
    ])

    # next-token targets
    y = np.stack([
        dataset[i + 1 : i + context_length + 1]
        for i in starts
    ])

    # move to torch tensors on requested device
    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)

    return x, y

def run_save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def run_load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["iteration"]
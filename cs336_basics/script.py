import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# change this import to match your filename
from transformer import (
    Embedding,
    Linear,
    RMSNorm,
    SwiGLU,
    MultiHeadSelfAttention,
    cross_entropy,
    AdamW,
    run_get_lr_cosine_schedule,
    run_gradient_clipping,
    run_get_batch,
    run_save_checkpoint,
)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta=10000.0, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, device=device, dtype=dtype)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta=10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.context_length = context_length

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, idx):
        x = self.token_embeddings(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, context_length, device, eval_iters):
    model.eval()
    out = {}

    for split_name, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(eval_iters):
            x, y = run_get_batch(data, batch_size, context_length, device)
            logits = model(x)
            B, T, V = logits.shape
            loss = cross_entropy(
                logits.reshape(B * T, V),
                y.reshape(B * T),
            )
            losses.append(loss.item())
        out[split_name] = sum(losses) / len(losses)

    model.train()
    return out


def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--dtype_np", type=str, default="int32")

    # model
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # optimization
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # logging / eval / checkpoint
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--checkpoint_interval", type=int, default=500)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")

    # device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    torch.manual_seed(1337)

    np_dtype = getattr(np, args.dtype_np)
    train_data = np.memmap(args.train_path, dtype=np_dtype, mode="r")
    val_data = np.memmap(args.val_path, dtype=np_dtype, mode="r")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
        dtype=torch.float32,
    ).to(args.device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    print(f"device: {args.device}")
    print(f"train tokens: {len(train_data)}")
    print(f"val tokens: {len(val_data)}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    model.train()

    for it in range(args.max_iters):
        lr_t = run_get_lr_cosine_schedule(
            it=it,
            max_learning_rate=args.lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.max_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr_t

        x, y = run_get_batch(
            dataset=train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device,
        )

        logits = model(x)
        B, T, V = logits.shape
        loss = cross_entropy(
            logits.reshape(B * T, V),
            y.reshape(B * T),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        run_gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        if it % args.log_interval == 0:
            print(f"iter {it:6d} | train loss {loss.item():.4f} | lr {lr_t:.6e}")

        if it % args.eval_interval == 0:
            losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device,
                eval_iters=args.eval_iters,
            )
            print(
                f"[eval] iter {it:6d} | "
                f"train {losses['train']:.4f} | "
                f"val {losses['val']:.4f}"
            )

        if it > 0 and it % args.checkpoint_interval == 0:
            ckpt_path = Path(args.checkpoint_path)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            run_save_checkpoint(model, optimizer, it, ckpt_path)
            print(f"saved checkpoint to {ckpt_path}")

    final_ckpt = Path(args.checkpoint_path)
    final_ckpt.parent.mkdir(parents=True, exist_ok=True)
    run_save_checkpoint(model, optimizer, args.max_iters, final_ckpt)
    print(f"saved final checkpoint to {final_ckpt}")


if __name__ == "__main__":
    main()
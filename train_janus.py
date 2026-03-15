#!/usr/bin/env python3
"""
Janus char-level training — PyTorch + CUDA on A100.
Three-way hybrid attention: QKV + RRPRAM + Janus self-resonance.

Usage:
  python3 train_janus.py --data leo_train.txt --steps 15000
  python3 train_janus.py --arch metajanus --data leo_train.txt --steps 15000
"""

import argparse
import math
import struct
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 256


# ═══════════════════════════════════════════════════════════
# Janus: QKV + RRPRAM + Janus self-resonance (3-way gate)
# Matches janus.c: DIM=288, HEADS=6, HEAD_DIM=48, BLOCKS=6
# ═══════════════════════════════════════════════════════════

class JanusAttention(nn.Module):
    """Three-way hybrid: QKV + RRPRAM + Janus self-resonance"""
    def __init__(self, E, H, D, T):
        super().__init__()
        self.H, self.D, self.T = H, D, T
        # QKV path
        self.wq = nn.Linear(E, H * D, bias=False)
        self.wk = nn.Linear(E, H * D, bias=False)
        self.wv = nn.Linear(E, H * D, bias=False)
        # RRPRAM path (separate value projection)
        self.wr = nn.Parameter(torch.randn(H, E, T) * (2.0 / E) ** 0.5)
        self.wvr = nn.Linear(E, H * D, bias=False)
        # Janus self-resonance (shared across heads)
        self.wj = nn.Linear(E, E, bias=False)
        # 3-way gate per head: [qkv, rrpram, janus]
        self.gate = nn.Parameter(torch.zeros(H, 3))
        # Output
        self.wo = nn.Linear(H * D, E, bias=False)

    def forward(self, x):
        B, T, E = x.shape
        H, D = self.H, self.D
        scale = 1.0 / (D ** 0.5)

        # === QKV attention ===
        q = self.wq(x).view(B, T, H, D).transpose(1, 2)  # [B,H,T,D]
        k = self.wk(x).view(B, T, H, D).transpose(1, 2)
        v = self.wv(x).view(B, T, H, D).transpose(1, 2)
        qkv_attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        qkv_attn = qkv_attn.masked_fill(mask, float('-inf'))
        qkv_attn = F.softmax(qkv_attn, dim=-1)
        qkv_out = torch.matmul(qkv_attn, v)  # [B,H,T,D]

        # === RRPRAM attention ===
        rrp_v = self.wvr(x).view(B, T, H, D).transpose(1, 2)  # [B,H,T,D]
        rrp_attn = torch.einsum('bte,het->bht', x, self.wr[:, :, :T]).unsqueeze(2)
        rrp_attn = rrp_attn.expand(B, H, T, T).clone() * scale
        rrp_attn = rrp_attn.masked_fill(mask, float('-inf'))
        rrp_attn = F.softmax(rrp_attn, dim=-1)
        rrp_out = torch.matmul(rrp_attn, rrp_v)  # [B,H,T,D]

        # === Janus self-resonance ===
        echo = self.wj(x)  # [B,T,E] — project through Janus weights
        # echo_back = wj^T @ echo → self-resonance scores
        echo_back = F.linear(echo, self.wj.weight.T)  # [B,T,E]
        # Score = dot(x, echo_back) per position
        scores = (x * echo_back).sum(dim=-1)  # [B,T]
        scores = scores / (E ** 0.5)
        # Attention = outer product of scores (mutual resonance)
        j_attn = scores.unsqueeze(-1) * scores.unsqueeze(-2)  # [B,T,T]
        j_attn = j_attn.masked_fill(mask, float('-inf'))
        j_attn = F.softmax(j_attn, dim=-1)  # [B,T,T]
        # Janus value: use echo reshaped per head
        j_val = echo.view(B, T, H, D).transpose(1, 2)  # [B,H,T,D]
        j_attn_h = j_attn.unsqueeze(1).expand(B, H, T, T)
        j_out = torch.matmul(j_attn_h, j_val)  # [B,H,T,D]

        # === 3-way gate ===
        gate = F.softmax(self.gate, dim=-1)  # [H, 3]
        ga = gate[:, 0].view(1, H, 1, 1)
        gb = gate[:, 1].view(1, H, 1, 1)
        gc = gate[:, 2].view(1, H, 1, 1)
        out = ga * qkv_out + gb * rrp_out + gc * j_out

        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.wo(out)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class JanusBlock(nn.Module):
    def __init__(self, E, H, D, T, M):
        super().__init__()
        self.rms1 = RMSNorm(E)
        self.attn = JanusAttention(E, H, D, T)
        self.rms2 = RMSNorm(E)
        self.w_gate = nn.Linear(E, M, bias=False)
        self.w_up = nn.Linear(E, M, bias=False)
        self.w_down = nn.Linear(M, E, bias=False)

    def forward(self, x):
        x = x + self.attn(self.rms1(x))
        h = self.rms2(x)
        x = x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))
        return x


class Janus(nn.Module):
    def __init__(self, E=288, H=6, D=48, T=256, B=6, M=768):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, E)
        self.pos_emb = nn.Embedding(T, E)
        self.blocks = nn.ModuleList([JanusBlock(E, H, D, T, M) for _ in range(B)])
        self.rms_f = RMSNorm(E)
        self.head = nn.Linear(E, VOCAB, bias=False)
        self.T = T

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        x = self.rms_f(x)
        return self.head(x)


# ═══════════════════════════════════════════════════════════
# MetaJanus: pure Janus self-resonance (no QKV, no RRPRAM)
# Matches metajanus.c: DIM=384, HEADS=6, HEAD_DIM=64, BLOCKS=6
# ═══════════════════════════════════════════════════════════

class MetaJanusAttention(nn.Module):
    """Pure Janus self-resonance: echo → mutual resonance → value"""
    def __init__(self, E, H, D, T):
        super().__init__()
        self.H, self.D, self.T = H, D, T
        self.wj = nn.Linear(E, E, bias=False)      # echo projection
        self.wj_v = nn.Linear(E, E, bias=False)     # value projection
        self.wo = nn.Linear(H * D, E, bias=False)

    def forward(self, x):
        B, T, E = x.shape
        H, D = self.H, self.D

        echo = self.wj(x)  # [B,T,E]
        echo_back = F.linear(echo, self.wj.weight.T)  # [B,T,E]
        scores = (x * echo_back).sum(dim=-1) / (E ** 0.5)  # [B,T]

        j_attn = scores.unsqueeze(-1) * scores.unsqueeze(-2)  # [B,T,T]
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        j_attn = j_attn.masked_fill(mask, float('-inf'))
        j_attn = F.softmax(j_attn, dim=-1)

        val = self.wj_v(x).view(B, T, H, D).transpose(1, 2)  # [B,H,T,D]
        j_attn_h = j_attn.unsqueeze(1).expand(B, H, T, T)
        out = torch.matmul(j_attn_h, val)  # [B,H,T,D]
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.wo(out)


class MetaJanusBlock(nn.Module):
    def __init__(self, E, H, D, T, M):
        super().__init__()
        self.rms1 = RMSNorm(E)
        self.attn = MetaJanusAttention(E, H, D, T)
        self.rms2 = RMSNorm(E)
        self.w_gate = nn.Linear(E, M, bias=False)
        self.w_up = nn.Linear(E, M, bias=False)
        self.w_down = nn.Linear(M, E, bias=False)

    def forward(self, x):
        x = x + self.attn(self.rms1(x))
        h = self.rms2(x)
        x = x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))
        return x


class MetaJanus(nn.Module):
    def __init__(self, E=384, H=6, D=64, T=256, B=6, M=1024):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, E)
        self.pos_emb = nn.Embedding(T, E)
        self.blocks = nn.ModuleList([MetaJanusBlock(E, H, D, T, M) for _ in range(B)])
        self.rms_f = RMSNorm(E)
        self.head = nn.Linear(E, VOCAB, bias=False)
        self.T = T

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        x = self.rms_f(x)
        return self.head(x)


# ═══════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════

def load_data(path):
    with open(path, 'rb') as f:
        raw = f.read()
    return torch.tensor(list(raw), dtype=torch.long)


def get_batch(data, T, batch_size, device):
    ix = torch.randint(0, len(data) - T - 1, (batch_size,))
    x = torch.stack([data[i:i+T] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+T+1] for i in ix]).to(device)
    return x, y


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(model, arch, path):
    """Save as binary: param count header + float32 weights.
    Linear weights transposed for C compatibility."""
    n = count_params(model)
    with open(path, 'wb') as f:
        f.write(struct.pack('i', n))
        for name, p in model.named_parameters():
            data = p.detach().cpu().float()
            if data.dim() == 2 and 'emb' not in name:
                data = data.T.contiguous()
            f.write(data.numpy().tobytes())
    print(f"[{arch}] saved {path} ({n:,} params)")


def cfg_from_depth(depth):
    T = 64 if depth >= 8 else 32
    E = depth * 32
    H = 4 if depth >= 4 else 2
    D = E // H
    B = depth
    M = E * 2
    return dict(T=T, E=E, H=H, D=D, B=B, M=M)


class MetaJanusRRPRAM(nn.Module):
    """MetaJanus + RRPRAM: Janus self-resonance + positional patterns, no QKV"""
    def __init__(self, E=384, H=4, D=96, T=64, B=12, M=768):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, E)
        self.pos_emb = nn.Embedding(T, E)
        self.blocks = nn.ModuleList([MetaJanusRRPRAMBlock(E, H, D, T, M) for _ in range(B)])
        self.rms_f = RMSNorm(E)
        self.head = nn.Linear(E, VOCAB, bias=False)
        self.T = T

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        x = self.rms_f(x)
        return self.head(x)


class MetaJanusRRPRAMBlock(nn.Module):
    def __init__(self, E, H, D, T, M):
        super().__init__()
        self.rms1 = RMSNorm(E)
        self.attn = MetaJanusRRPRAMAttention(E, H, D, T)
        self.rms2 = RMSNorm(E)
        self.w_gate = nn.Linear(E, M, bias=False)
        self.w_up = nn.Linear(E, M, bias=False)
        self.w_down = nn.Linear(M, E, bias=False)

    def forward(self, x):
        x = x + self.attn(self.rms1(x))
        h = self.rms2(x)
        x = x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))
        return x


class MetaJanusRRPRAMAttention(nn.Module):
    """Janus self-resonance + RRPRAM, 2-way gate, no QKV"""
    def __init__(self, E, H, D, T):
        super().__init__()
        self.H, self.D, self.T = H, D, T
        # RRPRAM
        self.wr = nn.Parameter(torch.randn(H, E, T) * (2.0 / E) ** 0.5)
        self.wvr = nn.Linear(E, H * D, bias=False)
        # Janus
        self.wj = nn.Linear(E, E, bias=False)
        # 2-way gate
        self.gate = nn.Parameter(torch.zeros(H, 2))
        self.wo = nn.Linear(H * D, E, bias=False)

    def forward(self, x):
        B, T, E = x.shape
        H, D = self.H, self.D
        scale = 1.0 / (D ** 0.5)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        # RRPRAM
        rv = self.wvr(x).view(B, T, H, D).transpose(1, 2)
        r_attn = torch.einsum('bte,het->bht', x, self.wr[:, :, :T]).unsqueeze(2)
        r_attn = (r_attn.expand(B, H, T, T).clone() * scale).masked_fill(mask, float('-inf'))
        rrp_out = torch.matmul(F.softmax(r_attn, dim=-1), rv)

        # Janus
        echo = self.wj(x)
        echo_back = F.linear(echo, self.wj.weight.T)
        scores = (x * echo_back).sum(dim=-1) / (E ** 0.5)
        j_attn = (scores.unsqueeze(-1) * scores.unsqueeze(-2)).masked_fill(mask, float('-inf'))
        j_attn = F.softmax(j_attn, dim=-1).unsqueeze(1).expand(B, H, T, T)
        j_val = echo.view(B, T, H, D).transpose(1, 2)
        jan_out = torch.matmul(j_attn, j_val)

        # 2-way gate
        g = F.softmax(self.gate, dim=-1)
        out = g[:, 0].view(1, H, 1, 1) * rrp_out + g[:, 1].view(1, H, 1, 1) * jan_out
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, H * D))


def train(arch, data_path, steps, save_path, lr=3e-4, batch_size=32, depth=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if arch == 'janus' and depth is None:
        model = Janus(E=288, H=6, D=48, T=256, B=6, M=768).to(device)
        T = 256
    elif arch == 'janus' and depth:
        c = cfg_from_depth(depth)
        model = Janus(E=c['E'], H=c['H'], D=c['D'], T=c['T'], B=c['B'], M=c['M']).to(device)
        T = c['T']
    elif arch == 'metajanus' and depth is None:
        model = MetaJanus(E=384, H=6, D=64, T=256, B=6, M=1024).to(device)
        T = 256
    elif arch == 'metajanus' and depth:
        c = cfg_from_depth(depth)
        model = MetaJanus(E=c['E'], H=c['H'], D=c['D'], T=c['T'], B=c['B'], M=c['M']).to(device)
        T = c['T']
    elif arch == 'metajanus_rrpram' and depth is None:
        model = MetaJanusRRPRAM(E=384, H=4, D=96, T=64, B=12, M=768).to(device)
        T = 64
    elif arch == 'metajanus_rrpram' and depth:
        c = cfg_from_depth(depth)
        model = MetaJanusRRPRAM(E=c['E'], H=c['H'], D=c['D'], T=c['T'], B=c['B'], M=c['M']).to(device)
        T = c['T']
    else:
        raise ValueError(f"unknown arch: {arch}")

    n_params = count_params(model)
    print(f"\n{'='*60}")
    print(f"  {arch.upper()} — char-level, T={T}")
    print(f"  params: {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"  device={device}, lr={lr}, batch={batch_size}, steps={steps}")
    print(f"{'='*60}")

    data = load_data(data_path)
    print(f"  data: {len(data)} bytes ({len(data)/1024:.1f}KB)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999),
                                   weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    model.train()
    t0 = time.time()
    best_loss = float('inf')

    for step in range(1, steps + 1):
        x, y = get_batch(data, T, batch_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

        if step % 100 == 0 or step == 1:
            dt = time.time() - t0
            sps = step / dt
            clr = scheduler.get_last_lr()[0]
            print(f"  step {step:5d}/{steps}  loss={loss.item():.4f}  "
                  f"best={best_loss:.4f}  lr={clr:.2e}  "
                  f"{sps:.1f} steps/s", flush=True)

        if step % 2000 == 0 and save_path:
            ckpt = save_path.replace('.bin', f'_step{step}.bin')
            save_checkpoint(model, arch, ckpt)

    if save_path:
        save_checkpoint(model, arch, save_path)

    dt = time.time() - t0
    print(f"\n  [{arch}] DONE: {steps} steps in {dt:.1f}s ({steps/dt:.1f} steps/s)")
    print(f"  [{arch}] final loss={loss.item():.4f}  best={best_loss:.4f}")

    # Print gate values for Janus
    if arch == 'janus':
        for i, blk in enumerate(model.blocks):
            g = F.softmax(blk.attn.gate, dim=-1).detach().cpu()
            qkv = [f'{v:.3f}' for v in g[:, 0]]
            rrp = [f'{v:.3f}' for v in g[:, 1]]
            jan = [f'{v:.3f}' for v in g[:, 2]]
            print(f"  [{arch}] block {i} gate: QKV=[{','.join(qkv)}] "
                  f"RRPRAM=[{','.join(rrp)}] Janus=[{','.join(jan)}]")

    # Generate sample
    print(f"\n  [{arch}] --- sample (temp=0.8) ---")
    model.eval()
    with torch.no_grad():
        seed = b"Q: who are you\nA: "
        ctx = torch.tensor([list(seed)], dtype=torch.long, device=device)
        out = list(seed)
        for _ in range(300):
            if ctx.shape[1] > T:
                ctx = ctx[:, -T:]
            logits = model(ctx)
            logits = logits[0, -1, :] / 0.8
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            out.append(next_id)
            if next_id == 10 and len(out) > len(seed) + 50:
                break
            ctx = torch.cat([ctx, torch.tensor([[next_id]], device=device)], dim=1)
        text = bytes(out).decode('utf-8', errors='replace')
        print(f"  {text}")

    return best_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='janus',
                        choices=['janus', 'metajanus', 'metajanus_rrpram'])
    parser.add_argument('--data', required=True)
    parser.add_argument('--depth', type=int, default=None,
                        help='Scale model with cfg_from_depth (default: original sizes)')
    parser.add_argument('--steps', type=int, default=15000)
    parser.add_argument('--save', default=None)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()

    if args.save is None:
        d = f'_d{args.depth}' if args.depth else ''
        args.save = f"{args.arch}{d}.bin"

    train(args.arch, args.data, args.steps, args.save,
          lr=args.lr, batch_size=args.batch, depth=args.depth)

"""LLM-bridge baseline: the ITFormer / Time-MQA paradigm, reproduced (Phase I.3).

A **frozen pretrained language model** reads the question as natural language; a
small **time-series encoder** embeds the signal and is projected into the LM's
representation space as *bridge tokens*; a **trainable cross-modal transformer**
fuses them and a head predicts yes/no. Only the bridge + TS encoder + head are
trained (the LM is frozen), mirroring ITFormer's "<1% trainable bridge over a
frozen LLM" and Time-MQA's LLM-driven QA.

This is an honest reimplementation of the *paradigm*, not ITFormer itself: the
frozen LM is a small sentence encoder (all-MiniLM-L6-v2, ~22M params, loaded from
the local HF cache) so the baseline runs on commodity hardware. The comparison is
structural --- like every end-to-end system it sees *all* channels and can exploit
a spurious shortcut, and it produces no by-construction explanation --- not an
attempt to match a 7B LLM's raw accuracy. We label results "an ITFormer-style
bridge", never "ITFormer".

``transformers`` is imported lazily so this module only needs it when actually used.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from executor.grammar import Always, And, Eventually, Not, Or, Predicate, Until

_FAM = {"high": "high", "low": "low", "rising": "rising", "falling": "falling",
        "anomalous": "anomalous"}


def _chan(names, c: int) -> str:
    return names[c] if names and c < len(names) else f"channel {c}"


def program_to_text(node, names) -> str:
    """Render an STL program node as a natural-language clause (for the frozen LM)."""
    if isinstance(node, Predicate):
        return f"{_chan(names, node.channel)} is {_FAM.get(node.name, node.name)}"
    if isinstance(node, Not):
        return f"it is not the case that {program_to_text(node.child, names)}"
    if isinstance(node, And):
        return f"({program_to_text(node.left, names)}) and ({program_to_text(node.right, names)})"
    if isinstance(node, Or):
        return f"({program_to_text(node.left, names)}) or ({program_to_text(node.right, names)})"
    if isinstance(node, Eventually):
        return f"at some time in [{node.a},{node.b}], {program_to_text(node.child, names)}"
    if isinstance(node, Always):
        return f"throughout [{node.a},{node.b}], {program_to_text(node.child, names)}"
    if isinstance(node, Until):
        return (f"({program_to_text(node.left, names)}) until "
                f"({program_to_text(node.right, names)}) within [{node.a},{node.b}]")
    raise TypeError(type(node).__name__)


def instance_text(inst) -> str:
    names = (getattr(inst, "provenance", None) or {}).get("channel_names") or []
    return "Is it true that " + program_to_text(inst.phi_star, names) + "?"


_LM_CACHE: dict = {}
_TXT_CACHE: dict = {}


def load_frozen_lm(name: str, device):
    """Load (and cache) a frozen pretrained LM + tokenizer from the local HF cache."""
    if name not in _LM_CACHE:
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        lm = AutoModel.from_pretrained(name).to(device).eval()
        for p in lm.parameters():
            p.requires_grad_(False)
        _LM_CACHE[name] = (tok, lm)
    return _LM_CACHE[name]


@torch.no_grad()
def encode_texts_frozen(lm, tok, texts, device, batch: int = 64, max_len: int = 48):
    """Run the frozen LM ONCE over ``texts`` -> sentence embeddings [N,d] (masked mean
    pool, the standard MiniLM usage). Cached by the text list (LM frozen, questions
    templated), so it is computed once and reused across seeds."""
    key = hash(tuple(texts))
    if key in _TXT_CACHE:
        return _TXT_CACHE[key]
    embs = []
    for s in range(0, len(texts), batch):
        enc = tok(texts[s:s + batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len)
        h = lm(input_ids=enc["input_ids"].to(device),
               attention_mask=enc["attention_mask"].to(device)).last_hidden_state  # [b,l,d]
        m = enc["attention_mask"].to(device).unsqueeze(-1).float()                  # [b,l,1]
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1e-6)                          # [b,d]
        embs.append(pooled.cpu())
    out = torch.cat(embs, 0).to(device)
    _TXT_CACHE[key] = out
    return out


class LLMBridge(nn.Module):
    """TS encoder (signal) + trainable cross-modal bridge over precomputed frozen-LM
    sentence embeddings -> yes/no. The LM runs once outside (``encode_texts_frozen``)."""

    def __init__(self, n_channels: int, d_lm: int = 384, d_bridge: int = 128,
                 ts_hidden: int = 64, n_ts_tokens: int = 8, bridge_layers: int = 2,
                 bridge_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.K = n_ts_tokens
        self.ts = nn.Sequential(
            nn.Conv1d(n_channels, ts_hidden, 5, padding=2), nn.ReLU(),
            nn.Conv1d(ts_hidden, ts_hidden, 5, padding=2), nn.ReLU())
        self.ts_proj = nn.Linear(ts_hidden, d_bridge)
        self.txt_proj = nn.Linear(d_lm, d_bridge)
        self.modal = nn.Embedding(2, d_bridge)        # 0 = TS token, 1 = text token
        layer = nn.TransformerEncoderLayer(d_bridge, bridge_heads, d_bridge * 2, dropout,
                                           batch_first=True)
        self.bridge = nn.TransformerEncoder(layer, bridge_layers)
        self.cls = nn.Parameter(torch.randn(1, 1, d_bridge) * 0.02)
        self.head = nn.Sequential(nn.LayerNorm(d_bridge), nn.Linear(d_bridge, 1))

    def forward(self, X: torch.Tensor, txt_emb: torch.Tensor):
        B = X.shape[0]
        z = self.ts(X.transpose(1, 2))                            # [B, ts_hidden, T]
        z = F.adaptive_avg_pool1d(z, self.K).transpose(1, 2)      # [B, K, ts_hidden]
        ts = self.ts_proj(z) + self.modal.weight[0]               # [B, K, db]
        txt = (self.txt_proj(txt_emb) + self.modal.weight[1]).unsqueeze(1)  # [B, 1, db]
        cls = self.cls.expand(B, -1, -1)
        seq = torch.cat([cls, ts, txt], dim=1)                    # [B, 1+K+1, db]
        out = self.bridge(seq)
        return self.head(out[:, 0]).squeeze(-1)                   # [B] logit


def train_llm_bridge(instances, n_channels: int, lm, tok, device, *, epochs: int = 60,
                     lr: float = 1e-3, batch_size: int = 64, seed: int = 0,
                     verbose: bool = False) -> LLMBridge:
    torch.manual_seed(seed)
    model = LLMBridge(n_channels).to(device)
    X = torch.stack([i.X for i in instances]).to(device)
    y = torch.tensor([float(i.answer_star) for i in instances], device=device)
    txt = encode_texts_frozen(lm, tok, [instance_text(i) for i in instances], device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    n = X.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            opt.zero_grad()
            loss = bce(model(X[idx], txt[idx]), y[idx])
            loss.backward()
            opt.step()
        if verbose and (epoch % 15 == 0 or epoch == epochs - 1):
            print(f"  [llm-bridge] epoch {epoch:3d} bce={float(loss):.4f}")
    return model


@torch.no_grad()
def eval_llm_bridge(model: LLMBridge, instances, lm, tok, device, batch_size: int = 512) -> dict:
    if not instances:
        return {"n": 0, "answer_accuracy": None, "by_depth": {}}
    from utils.oracle_metrics import group_accuracy
    model.eval()
    X = torch.stack([i.X for i in instances]).to(device)
    txt = encode_texts_frozen(lm, tok, [instance_text(i) for i in instances], device)
    preds = []
    for s in range(0, X.shape[0], batch_size):
        logit = model(X[s:s + batch_size], txt[s:s + batch_size])
        preds.extend((logit > 0).cpu().tolist())
    gold = [bool(i.answer_star) for i in instances]
    depth = [i.depth for i in instances]
    acc = sum(p == g for p, g in zip(preds, gold)) / len(gold)
    by_depth = {int(k): v["accuracy"] for k, v in group_accuracy(depth, preds, gold).items()}
    return {"n": len(gold), "answer_accuracy": acc, "by_depth": by_depth}

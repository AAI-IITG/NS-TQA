"""Faithful LLM-bridge baseline (WO-2A): a FROZEN instruct LLM answers our QA.

The conference bridge used a tiny frozen sentence encoder (all-MiniLM). This is the
stronger, ITFormer-faithful version a reviewer will ask for: a real **frozen instruct
LLM** (Qwen2.5-1.5B/7B-Instruct) reads the templated question as natural language, a
small **time-series encoder** turns the signal into ``K`` *time tokens*, a trainable
**projection** maps those into the LLM's embedding space, and the answer is scored by
comparing the LLM's next-token logits for `` yes`` vs `` no`` (no free generation --
exact and cheap). Only the encoder + projection (+ optional LoRA on the LLM) train;
the LLM weights are frozen. Honest label: "ITFormer-style bridge", never "ITFormer".

Design (soft-prefix injection, robust to prompt length):
    inputs_embeds = [ time_tokens (K,H) ] ++ [ left-padded chat-prompt embeddings ]
    logits        = LLM(inputs_embeds).logits[:, -1]          # answer position
    margin        = logits[:, yes_id] - logits[:, no_id]      # >0 => "yes"
The time tokens sit as a soft prefix before the prompt; the causal LLM attends to them
when scoring the final answer token. Nothing on the answer path is privileged: the
bridge sees the SAME train split and answer labels as the other baselines.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .llm_bridge import instance_text            # reuse the NL rendering of the program

# Qwen2.5 single-token answer ids (shared across 1.5B/7B); overridden from the tokenizer.
YES_ID, NO_ID = 9834, 902

_SYS = ("You answer yes/no questions about a machine's health from its sensor signal. "
        "Reply with a single word: yes or no.")


def build_prompt(question: str) -> list:
    """Qwen chat-formatted message list; the signal enters as a soft prefix, so the
    text only carries the system role + the question."""
    return [{"role": "system", "content": _SYS},
            {"role": "user", "content": "Signal summary provided. " + question}]


class TSEncoder(nn.Module):
    """Conv1d patcher over [B,C,T] -> K time tokens projected to the LLM hidden size."""

    def __init__(self, n_channels: int, d_llm: int, hidden: int = 128, n_tokens: int = 8):
        super().__init__()
        self.K = n_tokens
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU())
        self.proj = nn.Sequential(nn.Linear(hidden, d_llm), nn.LayerNorm(d_llm))

    def forward(self, X: torch.Tensor) -> torch.Tensor:      # X:[B,T,C] -> [B,K,H]
        z = self.net(X.transpose(1, 2))                      # [B,hidden,T]
        z = F.adaptive_avg_pool1d(z, self.K).transpose(1, 2)  # [B,K,hidden]
        return self.proj(z)


class InstructBridge(nn.Module):
    """Frozen instruct LLM + trainable TS encoder/projection; scores yes vs no logits."""

    def __init__(self, llm, tokenizer, n_channels: int, n_tokens: int = 8,
                 ts_hidden: int = 128, yes_id: int = YES_ID, no_id: int = NO_ID):
        super().__init__()
        self.llm = llm                                       # frozen (set by caller)
        self.tok = tokenizer
        self.d = llm.config.hidden_size
        self.encoder = TSEncoder(n_channels, self.d, ts_hidden, n_tokens)
        self.yes_id, self.no_id = yes_id, no_id
        self._embed = llm.get_input_embeddings()
        # match projected time-token norm to real token-embedding norm, else the
        # frozen LLM largely ignores the soft prefix (overfit-sanity was failing).
        with torch.no_grad():
            self.emb_norm = float(self._embed.weight.norm(dim=1).mean())

    def _prompt_ids(self, questions, device):
        texts = [self.tok.apply_chat_template(build_prompt(q), tokenize=False,
                                              add_generation_prompt=True) for q in questions]
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=96, add_special_tokens=False)
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    def forward(self, X, questions):
        device = X.device
        ts = self.encoder(X)                                 # [B,K,H] trainable
        ts = F.normalize(ts, dim=-1) * self.emb_norm         # match real-token embedding scale
        ids, mask = self._prompt_ids(questions, device)      # left-padded
        with torch.no_grad():
            txt_emb = self._embed(ids)                       # [B,L,H] (frozen embedding)
        inp = torch.cat([ts.to(txt_emb.dtype), txt_emb], dim=1)   # soft prefix [B,K+L,H]
        attn = torch.cat([torch.ones(ts.shape[:2], device=device, dtype=mask.dtype), mask], dim=1)
        out = self.llm(inputs_embeds=inp, attention_mask=attn)
        last = out.logits[:, -1, :].float()                  # [B,V] (left-pad => -1 is answer pos)
        return last[:, self.yes_id] - last[:, self.no_id]    # [B] margin (fp32), >0 => yes


def set_tokenizer_padding(tok):
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def train_instruct_bridge(instances, n_channels, llm, tok, device, *, epochs=30,
                          lr=1e-3, batch_size=8, grad_accum=1, n_tokens=8,
                          use_lora=False, seed=0, verbose=False):
    """Train the encoder/projection (LLM frozen) on answer labels via BCE on the
    yes/no margin. Backprop flows through the frozen LLM to the projection."""
    torch.manual_seed(seed)
    for p in llm.parameters():
        p.requires_grad_(False)
    model = InstructBridge(llm, tok, n_channels, n_tokens=n_tokens,
                           yes_id=_yes(tok), no_id=_no(tok)).to(device)
    if use_lora:
        _apply_lora(model)
    train_params = [p for p in model.parameters() if p.requires_grad]
    if verbose:
        n_tr = sum(p.numel() for p in train_params)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"  [instruct-bridge] trainable {n_tr/1e6:.2f}M / {n_all/1e9:.2f}B "
              f"({100*n_tr/n_all:.2f}%){' +LoRA' if use_lora else ' (LLM frozen)'}", flush=True)
    opt = torch.optim.AdamW(train_params, lr=lr)
    bce = nn.BCEWithLogitsLoss()
    X = torch.stack([i.X for i in instances])
    y = torch.tensor([float(i.answer_star) for i in instances])
    q = [instance_text(i) for i in instances]
    n = len(instances)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        opt.zero_grad()
        last = 0.0
        for step, s in enumerate(range(0, n, batch_size)):
            idx = perm[s:s + batch_size]
            xb = X[idx].to(device)
            yb = y[idx].to(device)
            qb = [q[j] for j in idx.tolist()]
            margin = model(xb, qb)
            loss = bce(margin, yb) / grad_accum
            loss.backward()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(train_params, 1.0)   # the soft-prefix grads are noisy
                opt.step()
                opt.zero_grad()
            last = float(loss) * grad_accum
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  [instruct-bridge] epoch {epoch:3d} bce={last:.4f}", flush=True)
    return model


@torch.no_grad()
def eval_instruct_bridge(model, instances, device, batch_size=16) -> dict:
    if not instances:
        return {"n": 0, "answer_accuracy": None, "by_depth": {}}
    from utils.oracle_metrics import group_accuracy
    model.eval()
    X = torch.stack([i.X for i in instances])
    q = [instance_text(i) for i in instances]
    preds = []
    for s in range(0, len(instances), batch_size):
        margin = model(X[s:s + batch_size].to(device), q[s:s + batch_size])
        preds.extend((margin > 0).cpu().tolist())
    gold = [bool(i.answer_star) for i in instances]
    depth = [i.depth for i in instances]
    acc = sum(p == g for p, g in zip(preds, gold)) / len(gold)
    by_depth = {int(k): v["accuracy"] for k, v in group_accuracy(depth, preds, gold).items()}
    return {"n": len(gold), "answer_accuracy": acc, "by_depth": by_depth}


def _yes(tok):
    ids = tok(" yes", add_special_tokens=False).input_ids
    return ids[0] if ids else YES_ID


def _no(tok):
    ids = tok(" no", add_special_tokens=False).input_ids
    return ids[0] if ids else NO_ID


def _apply_lora(model):
    """Wrap the frozen LLM with LoRA adapters (small % of params become trainable).
    The work order permits this as a config flag; a purely frozen LLM steered by a few
    soft tokens cannot fit the task (overfit sanity plateaued ~0.83)."""
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                     lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model.llm = get_peft_model(model.llm, cfg)     # must ASSIGN the wrapped model back


def load_frozen_instruct_lm(path: str, device, dtype="fp16"):
    """Load a frozen causal instruct LLM from a local path (offline)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = set_tokenizer_padding(AutoTokenizer.from_pretrained(path))
    td = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    lm = AutoModelForCausalLM.from_pretrained(path, torch_dtype=td).to(device).eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    return lm, tok

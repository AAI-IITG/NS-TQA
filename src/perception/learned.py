"""Learned temporal perception: signal window -> predicate truths (the de-renderer).

This is the ONLY load-bearing neural component on the faithful answer path. It
maps a raw window ``X:[T,C]`` to a symbolic state ``mu_hat:[T,P]`` (P = 4C) of
predicate truths in [0,1], in the SAME family-major column order as
``perception.grounding`` (high, low, rising, falling). The deterministic STL
executor then runs the program over ``mu_hat`` -- so a wrong perception can
produce a wrong answer (a genuine prediction), while ``executor`` itself stays
parameter-free.

Relation to the necessity experiment
------------------------------------
``per_channel=True`` (default) applies a shared temporal encoder to each channel
independently, so the predicted truths of a causal channel cannot depend on the
spurious channel. The symbolic path is therefore *structurally* unable to use
the spurious shortcut, not merely empirically reluctant to. ``per_channel=False``
is a cross-channel variant kept for real datasets where inter-channel context
may help perception (used as an ablation, not the necessity headline).

Supervision
-----------
Trained with per-predicate binary cross-entropy against the planted/privileged
``mu_star`` -- the analogue of training NS-VQA's attribute net against
ground-truth object attributes, NOT against answer correctness. Keeping the
target at the predicate level (not the answer) is what prevents the perception
net from quietly becoming an end-to-end answer model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from benchmark.baseline import select_device
from executor.hard_logic import evaluate
from perception.grounding import PRED_FAMILIES, predicate_index

N_FAMILIES = len(PRED_FAMILIES)  # 4: high, low, rising, falling


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

class LearnedPerception(nn.Module):
    """X:[B,T,C] -> mu_hat:[B,T,4C] in [0,1], family-major columns.

    A small same-padding temporal CNN gives each timestep a local receptive
    field large enough to read level (high/low) and trend (rising/falling).
    """

    def __init__(
        self,
        n_channels: int,
        hidden: int = 32,
        kernel: int = 5,
        n_layers: int = 2,
        per_channel: bool = True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.per_channel = per_channel
        pad = kernel // 2
        in_ch = 1 if per_channel else n_channels
        out_ch = N_FAMILIES if per_channel else N_FAMILIES * n_channels

        layers: list[nn.Module] = []
        prev = in_ch
        for _ in range(n_layers):
            layers += [nn.Conv1d(prev, hidden, kernel, padding=pad), nn.ReLU()]
            prev = hidden
        layers += [nn.Conv1d(prev, out_ch, 1)]  # 1x1 conv to predicate logits
        self.net = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X:[B,T,C] -> mu_hat:[B,T,4C] (sigmoid), family-major."""
        B, T, C = X.shape
        if self.per_channel:
            # process each channel independently with shared weights
            z = X.permute(0, 2, 1).reshape(B * C, 1, T)      # [B*C, 1, T]
            logits = self.net(z)                              # [B*C, 4, T]
            logits = logits.reshape(B, C, N_FAMILIES, T)      # [B, C, 4, T]
            # assemble family-major: column fi*C + c
            blocks = [logits[:, :, fi, :].transpose(1, 2) for fi in range(N_FAMILIES)]
            mu = torch.cat(blocks, dim=2)                     # [B, T, 4C]
        else:
            z = X.permute(0, 2, 1)                            # [B, C, T]
            logits = self.net(z)                              # [B, 4C, T]
            mu = logits.permute(0, 2, 1)                      # [B, T, 4C]
        return torch.sigmoid(mu)


# --------------------------------------------------------------------------- #
# data plumbing
# --------------------------------------------------------------------------- #

def _attr(instance: Any, key: str, default: Any = None) -> Any:
    if isinstance(instance, dict):
        return instance.get(key, default)
    return getattr(instance, key, default)


def stack_xy(instances: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack (X, mu_star) over a list of instances -> ([B,T,C], [B,T,P])."""
    X = torch.stack([_attr(i, "X") for i in instances])
    mu = torch.stack([_attr(i, "mu_star") for i in instances])
    return X, mu


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

@dataclass
class PerceptionTrainResult:
    model: LearnedPerception
    history: list[dict]
    device: str


def train_perception(
    instances: list,
    n_channels: int,
    hidden: int = 32,
    kernel: int = 5,
    n_layers: int = 2,
    per_channel: bool = True,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device_pref: str = "cpu",
    seed: int = 0,
    log_every: int = 10,
    verbose: bool = True,
) -> PerceptionTrainResult:
    """Train perception with per-predicate BCE against mu_star.

    Targets are predicate truths, NOT answers: the executor downstream is the
    only thing that turns truths into an answer, so the net never learns to
    answer directly.
    """
    torch.manual_seed(seed)
    device = select_device(device_pref)

    model = LearnedPerception(n_channels, hidden, kernel, n_layers, per_channel).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCELoss()

    X_all, mu_all = stack_xy(instances)
    X_all, mu_all = X_all.to(device), mu_all.to(device)
    n = X_all.shape[0]

    history: list[dict] = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            xb, yb = X_all[idx], mu_all[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = bce(pred, yb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        epoch_loss = total / n
        history.append({"epoch": epoch, "bce": epoch_loss})
        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  [perception] epoch {epoch:3d} bce={epoch_loss:.4f}")

    return PerceptionTrainResult(model=model, history=history, device=str(device))


def train_perception_answer_only(
    instances: list,
    n_channels: int,
    hidden: int = 32,
    kernel: int = 5,
    n_layers: int = 2,
    per_channel: bool = True,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    beta: float = 8.0,
    answer_scale: float = 5.0,
    device_pref: str = "cpu",
    seed: int = 0,
    log_every: int = 10,
    verbose: bool = True,
) -> PerceptionTrainResult:
    """Train perception from ANSWER labels only (no ``mu_star`` supervision).

    The ablation that isolates the architecture from dense privileged supervision:
    perception is learned by backpropagating the answer loss through the
    *differentiable* (soft) STL executor, so the only signal is whether the
    program's answer is right. Inference still uses the HARD executor. If NS-TQA
    keeps its edge here, its advantage is the neuro-symbolic *structure*, not the
    per-predicate supervision the headline NS-TQA enjoys (pre-empts the "unfair
    mu* supervision vs answer-only baselines" critique).

    Per-instance programs differ, so the executor is looped per instance while the
    perception CNN stays batched. ``answer_scale`` maps soft robustness (~[-1,1])
    to a logit; ``beta`` is the softmin/softmax sharpness.
    """
    from executor.soft_logic import soft_robustness

    torch.manual_seed(seed)
    device = select_device(device_pref)
    pidx = predicate_index(n_channels)

    model = LearnedPerception(n_channels, hidden, kernel, n_layers, per_channel).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()

    X_all = torch.stack([_attr(i, "X") for i in instances]).to(device)
    phis = [_attr(i, "phi_star") for i in instances]
    y_all = torch.tensor([float(_attr(i, "answer_star")) for i in instances], device=device)
    n = X_all.shape[0]

    history: list[dict] = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total, correct = 0.0, 0
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            xb = X_all[idx]
            opt.zero_grad()
            mu_b = model(xb)                                   # [B, T, P] differentiable
            logits = []
            for j, ii in enumerate(idx.tolist()):
                rho0 = soft_robustness(mu_b[j], phis[ii], pidx, beta)[0]  # anchor robustness
                logits.append(answer_scale * rho0)
            logits = torch.stack(logits)
            loss = bce(logits, y_all[idx])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
            correct += int(((logits.detach() > 0) == (y_all[idx] > 0.5)).sum())
        epoch_loss = total / n
        history.append({"epoch": epoch, "bce": epoch_loss, "soft_acc": correct / n})
        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  [perception/answer-only] epoch {epoch:3d} "
                  f"bce={epoch_loss:.4f} soft_acc={correct / n:.3f}")

    return PerceptionTrainResult(model=model, history=history, device=str(device))


# --------------------------------------------------------------------------- #
# inference + evaluation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def predict_mu(model: LearnedPerception, X: torch.Tensor) -> torch.Tensor:
    """Predict mu_hat:[T,P] for a single window X:[T,C] (eval mode, detached)."""
    model.eval()
    device = next(model.parameters()).device
    mu = model(X.unsqueeze(0).to(device))[0]
    return mu.detach().cpu()


@torch.no_grad()
def predicate_metrics(model: LearnedPerception, instances: list) -> dict:
    """Per-family and macro F1 of predicted predicate truths (threshold 0.5).

    This is the perception-quality report -- the NS-VQA 'attribute accuracy'
    analogue. It should be high and roughly EQUAL across regimes, since the
    causal channels are identical between indist and shift.
    """
    model.eval()
    device = next(model.parameters()).device
    X_all, mu_all = stack_xy(instances)
    pred = (model(X_all.to(device)) >= 0.5).float().cpu()
    gold = (mu_all >= 0.5).float()
    C = model.n_channels

    out: dict[str, Any] = {}
    f1s = []
    for fi, fam in enumerate(PRED_FAMILIES):
        sl = slice(fi * C, (fi + 1) * C)
        p, g = pred[:, :, sl], gold[:, :, sl]
        tp = float((p * g).sum())
        fp = float((p * (1 - g)).sum())
        fn = float(((1 - p) * g).sum())
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        out[fam] = {"precision": precision, "recall": recall, "f1": f1}
        f1s.append(f1)
    out["macro_f1"] = sum(f1s) / len(f1s)
    out["elementwise_acc"] = float((pred == gold).float().mean())
    return out


@torch.no_grad()
def answer_via_executor(
    model: LearnedPerception, instances: list, n_channels: int
) -> dict:
    """Run the FAITHFUL path: learned perception -> hard executor -> answer.

    Returns per-instance predictions and answer accuracy. This is the symbolic
    path's real score (bounded by perception error), NOT the oracle.
    """
    pidx = predicate_index(n_channels)
    records = []
    correct = 0
    for inst in instances:
        X = _attr(inst, "X")
        phi = _attr(inst, "phi_star")
        gold = bool(_attr(inst, "answer_star"))
        mu_hat = predict_mu(model, X)
        rho, tr = evaluate(mu_hat, phi, pidx, anchor=0)
        pred = bool(rho > 0.0)
        correct += int(pred == gold)
        records.append({
            "pred_answer": pred, "gold_answer": gold,
            "pred_rho": float(rho), "critical_t": tr.critical_t,
            "depth": _attr(inst, "depth"),
        })
    return {"answer_accuracy": correct / max(1, len(instances)), "records": records}
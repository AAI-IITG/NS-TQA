"""Learned `anomalous` predicate: privileged healthy-baseline target + neural head.

This is the predicate that gives perception a *substantive* job (cf.
``markdowns/anomaly_predicate_design.md``). Unlike ``high/low/rising/falling`` ---
whose privileged target is a quantile threshold a fixed hand-rule can reproduce ---
the ``anomalous`` target is **deviation from a healthy baseline**, so:

  * a fixed-threshold STL-only baseline (which has no notion of a per-channel
    healthy reference) cannot reproduce it, but
  * a learned per-channel head, trained on the target, can.

This isolates the value of *learned* perception on real signals (where the
threshold-defined families made it look unnecessary, since STL-only is near-oracle
there). The grammar/executor are unchanged: ``anomalous`` is just another atomic
per-channel predicate with leaf robustness ``2*mu - 1``.

Target construction (privileged, train-only; never uses RUL / answer / program):
  1. healthy reference R = first ``healthy_frac`` of each TRAIN unit's life;
  2. per-channel healthy mean ``m_c`` and std ``s_c`` from R;
  3. standardized deviation ``dev_c(t) = |(smooth(x)_c(t) - m_c)/s_c|``;
  4. threshold ``theta_c = quantile_q(dev over R)`` ("further from baseline than the
     machine ever was while healthy");
  5. soft truth ``mu*_anom(c,t) = sigmoid(a_anom * (dev_c(t) - theta_c))``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from benchmark.adapters import window_series
from benchmark.baseline import select_device
from perception.grounding import smooth


@dataclass
class HealthyBaseline:
    """Per-channel healthy reference stats + anomaly threshold (train-only fit)."""

    m: torch.Tensor          # [C] healthy mean
    s: torch.Tensor          # [C] healthy std
    theta: torch.Tensor      # [C] anomaly threshold (quantile of healthy deviation)
    a_anom: float = 4.0
    smooth_k: int = 5

    @staticmethod
    def fit(healthy_windows: torch.Tensor, q: float = 0.99,
            a_anom: float = 4.0, smooth_k: int = 5) -> "HealthyBaseline":
        """Fit from healthy windows ``[N,T,C]`` (drawn from early life of train units)."""
        N, T, C = healthy_windows.shape
        z = torch.stack([smooth(w, smooth_k) for w in healthy_windows])  # [N,T,C]
        flat = z.reshape(-1, C)
        m = flat.mean(0)
        s = flat.std(0).clamp_min(1e-6)
        dev = ((flat - m) / s).abs()
        theta = torch.quantile(dev, q, dim=0)
        return HealthyBaseline(m=m, s=s, theta=theta, a_anom=a_anom, smooth_k=smooth_k)

    def deviation(self, x: torch.Tensor) -> torch.Tensor:
        """Standardized |deviation from healthy| for window x:[T,C] -> [T,C]."""
        z = smooth(x, self.smooth_k)
        return ((z - self.m) / self.s).abs()

    def target(self, x: torch.Tensor) -> torch.Tensor:
        """Privileged soft anomaly truth mu*_anom:[T,C] for window x:[T,C]."""
        return torch.sigmoid(self.a_anom * (self.deviation(x) - self.theta))


def healthy_windows(series_list, healthy_frac: float, T: int, stride: int,
                    max_per_unit: int = 50) -> torch.Tensor:
    """Stack windows from the first ``healthy_frac`` of each unit's life -> [N,T,C]."""
    out = []
    for s in series_list:
        cut = max(T, int(round(healthy_frac * s.L)))
        head = s.values[:cut]
        if head.shape[0] < T:
            continue
        for w in window_series(_slice_series(s, head), T, stride, max_per_unit):
            out.append(w.X)
    if not out:
        raise ValueError("no healthy windows extracted")
    return torch.stack(out)


def _slice_series(s, values):
    from benchmark.adapters import Series
    return Series(values=values, unit_id=s.unit_id, dataset=s.dataset,
                  channel_names=s.channel_names, condition=s.condition, fs=s.fs)


# --------------------------------------------------------------------------- #
# learned anomaly head (per-channel, shared weights) -- the neural substance
# --------------------------------------------------------------------------- #

class AnomalyHead(nn.Module):
    """X:[B,T,C] -> mu_anom:[B,T,C] in [0,1]. Per-channel temporal CNN, shared weights."""

    def __init__(self, hidden: int = 32, kernel: int = 5, n_layers: int = 2):
        super().__init__()
        pad = kernel // 2
        layers: list[nn.Module] = []
        prev = 1
        for _ in range(n_layers):
            layers += [nn.Conv1d(prev, hidden, kernel, padding=pad), nn.ReLU()]
            prev = hidden
        layers += [nn.Conv1d(prev, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, T, C = X.shape
        z = X.permute(0, 2, 1).reshape(B * C, 1, T)     # per-channel
        logits = self.net(z).reshape(B, C, T)
        return torch.sigmoid(logits.permute(0, 2, 1))   # [B,T,C]


def train_anomaly_head(windows: torch.Tensor, baseline: HealthyBaseline,
                       hidden: int = 32, kernel: int = 5, n_layers: int = 2,
                       epochs: int = 60, batch_size: int = 64, lr: float = 1e-3,
                       device_pref: str = "cpu", seed: int = 0,
                       verbose: bool = False) -> AnomalyHead:
    """Train the head on the privileged target (predicate-level BCE). windows:[N,T,C]."""
    torch.manual_seed(seed)
    device = select_device(device_pref)
    targets = torch.stack([baseline.target(w) for w in windows]).to(device)  # [N,T,C]
    X = windows.to(device)
    model = AnomalyHead(hidden, kernel, n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()
    n = X.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            opt.zero_grad()
            loss = bce(model(X[idx]), targets[idx])
            loss.backward()
            opt.step()
        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(f"  [anomaly] epoch {epoch:3d} bce={float(loss):.4f}")
    return model


@torch.no_grad()
def predict_anomaly(model: AnomalyHead, x: torch.Tensor) -> torch.Tensor:
    """mu_hat_anom:[T,C] for a single window x:[T,C]."""
    model.eval()
    device = next(model.parameters()).device
    return model(x.unsqueeze(0).to(device))[0].cpu()

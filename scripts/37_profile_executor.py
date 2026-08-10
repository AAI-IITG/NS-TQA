"""37 - Executor complexity profiling (WO-4).

The deterministic STL executor is parameter-free, but its cost scales with the window
length T and with the temporal operators used. This script measures that cost
empirically and confirms the analytic complexity:

  * Predicate / Not / And / Or   : O(1) or O(T)   (elementwise over time)
  * Eventually / Always          : O(T) per program (one min/max scan per anchor row)
  * Until                        : O(T^2)          (for each anchor t, scan split points t')

We time ``hard_logic.evaluate`` on random truth tensors for a grid of T, separately for
programs WITHOUT and WITH ``Until``, fit a power law on log-log, and report the practical
T limit (where a single ``Until`` execution exceeds a millisecond-scale budget).

Run:  python scripts/37_profile_executor.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import math

import torch

from executor.grammar import (Always, And, Eventually, Or, Predicate, Until)
from executor.hard_logic import evaluate
from perception.grounding import predicate_index

C = 8
REPEATS = 40
TS = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384]


def prog_no_until(T):
    """A depth-3 program using only elementwise + Eventually/Always (O(T))."""
    return Eventually(0, T - 1,
                      And(Always(0, T - 1, Predicate("high", 0)),
                          Or(Predicate("rising", 1), Predicate("low", 2))))


def prog_until(T):
    """A depth-3 program whose root is Until (the O(T^2) operator)."""
    return Until(0, T - 1,
                 Or(Predicate("high", 0), Predicate("rising", 1)),
                 And(Predicate("low", 2), Predicate("falling", 3)))


def time_eval(build, T, pidx):
    mu = torch.rand(T, 4 * C)                       # random truth tensor
    phi = build(T)
    # warmup
    evaluate(mu, phi, pidx, anchor=0)
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        evaluate(mu, phi, pidx, anchor=0)
    return (time.perf_counter() - t0) / REPEATS * 1e3   # ms per call


def loglog_slope(xs, ys):
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(xs)
    mx, my = sum(lx) / n, sum(ly) / n
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den


def main():
    pidx = predicate_index(C)
    rows = []
    for T in TS:
        t_no = time_eval(prog_no_until, T, pidx)
        t_un = time_eval(prog_until, T, pidx)
        rows.append((T, t_no, t_un))
        print(f"  T={T:4d}   no-Until={t_no:8.4f} ms   Until={t_un:9.4f} ms   "
              f"ratio={t_un/max(t_no,1e-9):6.1f}x", flush=True)

    s_no = loglog_slope(TS, [r[1] for r in rows])
    s_un = loglog_slope(TS, [r[2] for r in rows])
    # practical limit: largest T whose single Until call stays under 5 ms
    limit = max([r[0] for r in rows if r[2] < 5.0], default=TS[0])

    out_dir = ROOT / "runs" / "complexity"
    out_dir.mkdir(parents=True, exist_ok=True)
    md = ["# Executor complexity (WO-4)", "",
          f"Per-call wall-clock of `hard_logic.evaluate` (mean of {REPEATS}, C={C}), by "
          "window length T. Log-log slope estimates the exponent.", "",
          "| T | no-Until (ms) | Until (ms) | Until/no ratio |", "|---|---|---|---|"]
    for T, a, b in rows:
        md.append(f"| {T} | {a:.4f} | {b:.4f} | {b/max(a,1e-9):.1f}x |")
    md += ["",
           f"**Fitted exponents (log-log slope):** no-Until $\\approx$ T^{s_no:.2f} "
           f"(linear/O(T)); Until $\\approx$ T^{s_un:.2f} (quadratic/O(T^2)).",
           f"**Practical limit:** a single `Until` stays under 5 ms up to T $\\approx$ {limit}; "
           "our benchmarks use T $\\le$ 64, and `Until` is disabled by default "
           "(`allow_until=False`), so the answer path is O(T) in practice."]
    (out_dir / "complexity.md").write_text("\n".join(md))

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.plot(TS, [r[1] for r in rows], "o-", color="#0072b2",
                label=f"no Until  (fit T^{s_no:.2f})")
        ax.plot(TS, [r[2] for r in rows], "s-", color="#d55e00",
                label=f"with Until (fit T^{s_un:.2f})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("window length T"); ax.set_ylabel("executor time per call (ms)")
        ax.set_title("STL executor cost vs. T: Until is O(T$^2$)")
        ax.legend(); ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_complexity.png", dpi=200, bbox_inches="tight")
        import shutil
        shutil.copy(out_dir / "fig_complexity.png", ROOT / "paper" / "figures" / "fig_complexity.png")
        print("wrote figure -> paper/figures/fig_complexity.png")
    except Exception as e:
        print(f"(figure skipped: {e})")

    print("\n" + "\n".join(md))
    print(f"\nwrote -> {out_dir}/complexity.md")


if __name__ == "__main__":
    main()

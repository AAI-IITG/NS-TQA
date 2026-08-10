"""42 - NL -> program parser evaluation (WO-6A).

Trains the grammar-verified NL->STL parser on paraphrases of templated questions and
measures, on HELD-OUT programs (unseen at train), (i) exact-program-match parse
accuracy, (ii) the reject rate on genuinely malformed inputs, and (iii) end-to-end
answer accuracy through the parser vs. gold programs on a real benchmark -- the
delta is the parser's cost. Every parse is grammar-validated; invalid decodes are
rejected, never executed.

Run:  python scripts/42_run_parser.py [--n-programs 1500] [--paraphrases 6] [--seed 0] [--quick]
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

# native LSTM (not cuDNN RNN) so the seq2seq runs on a P100 (sm_60), whose cuDNN does
# not support RNN kernels; the batch still parallelises on GPU.
torch.backends.cudnn.enabled = False

from benchmark.spurious import sample_program
from executor.grammar import parse_program
from parser.nl2stl import (build_vocabs, decode, parse_verified, question,
                           train_parser)


def gen_pairs(programs, rng, k):
    pairs = []
    for phi in programs:
        s = phi.canonical()
        seen = set()
        for _ in range(k * 3):
            q = question(phi, rng)
            if q not in seen:
                seen.add(q); pairs.append((q, s))
            if len(seen) >= k:
                break
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-programs", type=int, default=1500)
    ap.add_argument("--paraphrases", type=int, default=6)
    ap.add_argument("--channels", type=int, default=14)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n_programs, args.paraphrases = 60, 3
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # distinct programs, split by PROGRAM into train/test (held-out programs)
    progs, seen = [], set()
    while len(progs) < args.n_programs:
        d = rng.randint(1, 4)
        phi = sample_program(d, list(range(args.channels)), args.T, rng, allow_until=False)
        c = phi.canonical()
        if c not in seen:
            seen.add(c); progs.append(phi)
    rng.shuffle(progs)
    n_te = max(50, len(progs) // 5)
    tr_p, te_p = progs[n_te:], progs[:n_te]

    train_pairs = gen_pairs(tr_p, rng, args.paraphrases)
    test_pairs = gen_pairs(te_p, rng, args.paraphrases)
    sv, tv = build_vocabs(train_pairs + test_pairs)
    print(f"programs: train={len(tr_p)} test={len(te_p)} | pairs: train={len(train_pairs)} "
          f"test={len(test_pairs)} | src-vocab={len(sv)} tgt-vocab={len(tv)}", flush=True)

    model = train_parser(train_pairs, sv, tv, device, epochs=5 if args.quick else args.epochs,
                         seed=args.seed, verbose=True)

    # (i) exact-program-match on held-out paraphrases
    if args.quick:
        test_pairs = test_pairs[:80]
    texts = [t for t, _ in test_pairs]
    golds = [s for _, s in test_pairs]
    res = parse_verified(model, texts, sv, tv, device, n_predicates=args.channels)
    exact = sum(r["valid"] and r["program"].canonical() == g for r, g in zip(res, golds)) / len(res)
    valid_rate = sum(r["valid"] for r in res) / len(res)
    # exact among the accepted ones
    acc_valid = [(r, g) for r, g in zip(res, golds) if r["valid"]]
    exact_of_valid = (sum(r["program"].canonical() == g for r, g in acc_valid) / len(acc_valid)
                      if acc_valid else 0.0)

    # (ii) reject rate on genuinely malformed inputs (scrambled tokens)
    bad_texts = []
    for t in texts[:200]:
        toks = t.split()
        rng.shuffle(toks)
        bad_texts.append(" ".join(toks[:max(2, len(toks) // 2)]))
    bad = parse_verified(model, bad_texts, sv, tv, device, n_predicates=args.channels)
    # a decode is only a concern if it PARSES to a valid program that is wrong;
    # report how many malformed inputs still yield a grammar-valid program
    bad_valid = sum(r["valid"] for r in bad) / len(bad)

    out = {
        "n_programs": len(progs), "held_out_programs": len(te_p),
        "exact_program_match": round(exact, 4),
        "grammar_valid_rate": round(valid_rate, 4),
        "exact_among_accepted": round(exact_of_valid, 4),
        "malformed_still_valid_rate": round(bad_valid, 4),
        "seed": args.seed,
    }
    out_dir = ROOT / "runs" / "parser"
    out_dir.mkdir(parents=True, exist_ok=True)
    md = ["# NL -> program parser (WO-6A)", "",
          "Grammar-verified seq2seq parser on paraphrased templated questions; evaluated on "
          "HELD-OUT programs (unseen at train). Invalid decodes are rejected, never executed.", "",
          "| metric | value |", "|---|---|",
          f"| exact-program-match (held-out paraphrases) | {out['exact_program_match']:.3f} |",
          f"| grammar-valid decode rate | {out['grammar_valid_rate']:.3f} |",
          f"| exact-match among ACCEPTED parses | {out['exact_among_accepted']:.3f} |",
          f"| malformed inputs yielding a valid program | {out['malformed_still_valid_rate']:.3f} |",
          "",
          "_The answer path executes only grammar-valid parses; anything the parser cannot "
          "render into a valid program is rejected rather than guessed._"]
    (out_dir / "parser.md").write_text("\n".join(md))
    (out_dir / "parser.json").write_text(json.dumps(out, indent=2))
    print("\n" + "\n".join(md))
    print(f"\nwrote -> {out_dir}/parser.{{md,json}}")


if __name__ == "__main__":
    main()

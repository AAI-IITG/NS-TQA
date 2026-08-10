"""NL -> STL-program parser (WO-6A): a subordinate, grammar-VERIFIED front-end.

The answer path is untouched: this maps a natural-language question to a program
S-expression, which is then executed deterministically by the same executor. Every
decode is validated with ``grammar.parse_program`` and REJECTED if invalid, so the
parser can never inject a malformed or out-of-grammar program into the answer path.

Design (dependency-light; core ``torch`` only, no transformers):
  * paraphrase generation -- each program is rendered into varied NL surface forms
    (synonym + template tables), so the parser must generalise over phrasing;
  * a small attention seq2seq (word-level encoder, char-level decoder) emits the
    canonical S-expression; character-level decoding lets it copy channel indices
    and interval bounds exactly;
  * grammar-constrained acceptance: decode -> ``parse_program`` -> accept only if it
    round-trips and validates, else reject.
"""
from __future__ import annotations

import random
import re
from typing import Optional

import torch
import torch.nn as nn

from executor.grammar import (Always, And, Eventually, Not, Or, Predicate, Until,
                              parse_program, STLSyntaxError)

# ---- paraphrase tables ---------------------------------------------------- #
_PRED = {
    "high": ["is high", "is elevated", "reads high", "is large"],
    "low": ["is low", "is depressed", "reads low", "is small"],
    "rising": ["is rising", "is increasing", "trends upward", "climbs"],
    "falling": ["is falling", "is decreasing", "trends downward", "drops"],
    "anomalous": ["is anomalous", "deviates from its healthy baseline", "looks abnormal"],
}
_CHAN = ["channel {c}", "sensor {c}", "signal {c}"]
_IV = ["[{a},{b}]", "snapshots {a} to {b}", "the window [{a},{b}]"]
_EV = ["at some snapshot in {iv}, {x}", "at some point during {iv}, {x}",
       "{x} somewhere within {iv}", "sometime in {iv}, {x}"]
_AL = ["throughout {iv}, {x}", "at every snapshot in {iv}, {x}",
       "{x} for all of {iv}", "always during {iv}, {x}"]
_AND = ["{l} and {r}", "both {l} and {r}"]
_OR = ["{l} or {r}", "either {l} or {r}"]
_NOT = ["it is not the case that {x}", "not {x}"]
_UNTIL = ["{l} until {r} within {iv}", "{l} holds until {r} in {iv}"]
_WRAP = ["is it true that {q}?", "does it hold that {q}?", "check whether {q}",
         "verify that {q}", "{q}?"]


def paraphrase(node, rng: random.Random) -> str:
    def iv(a, b):
        return rng.choice(_IV).format(a=a, b=b)
    if isinstance(node, Predicate):
        return rng.choice(_CHAN).format(c=node.channel) + " " + rng.choice(_PRED.get(
            node.name, [f"is {node.name}"]))
    if isinstance(node, Not):
        return rng.choice(_NOT).format(x=paraphrase(node.child, rng))
    if isinstance(node, And):
        return rng.choice(_AND).format(l=paraphrase(node.left, rng), r=paraphrase(node.right, rng))
    if isinstance(node, Or):
        return rng.choice(_OR).format(l=paraphrase(node.left, rng), r=paraphrase(node.right, rng))
    if isinstance(node, Eventually):
        return rng.choice(_EV).format(iv=iv(node.a, node.b), x=paraphrase(node.child, rng))
    if isinstance(node, Always):
        return rng.choice(_AL).format(iv=iv(node.a, node.b), x=paraphrase(node.child, rng))
    if isinstance(node, Until):
        return rng.choice(_UNTIL).format(l=paraphrase(node.left, rng),
                                         r=paraphrase(node.right, rng), iv=iv(node.a, node.b))
    raise TypeError(type(node).__name__)


def question(node, rng: random.Random) -> str:
    return rng.choice(_WRAP).format(q=paraphrase(node, rng))


# ---- tokenisation --------------------------------------------------------- #
def src_tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+|\d+|[^\s\w]", text.lower())


BOS, EOS, PAD = "\x02", "\x03", "\x00"


class Vocab:
    def __init__(self, items, reserved=(PAD,)):
        self.itos = list(reserved) + sorted(set(items) - set(reserved))
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def enc(self, seq):
        return [self.stoi.get(s, 0) for s in seq]


# ---- model ---------------------------------------------------------------- #
class Seq2Seq(nn.Module):
    def __init__(self, n_src, n_tgt, emb=96, hid=192):
        super().__init__()
        self.se = nn.Embedding(n_src, emb, padding_idx=0)
        self.enc = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.te = nn.Embedding(n_tgt, emb, padding_idx=0)
        self.dec = nn.LSTMCell(emb + 2 * hid, 2 * hid)
        self.att = nn.Linear(2 * hid, 2 * hid)
        self.out = nn.Linear(4 * hid, n_tgt)
        self.hid = hid

    def encode(self, src, src_len):
        e = self.se(src)
        packed = nn.utils.rnn.pack_padded_sequence(e, src_len.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        out, (h, c) = self.enc(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        h = torch.cat([h[0], h[1]], -1)
        c = torch.cat([c[0], c[1]], -1)
        return out, (h, c)

    def step(self, enc_out, mask, y_prev, hx):
        emb = self.te(y_prev)
        h, c = hx
        # attention over encoder states
        scores = (self.att(enc_out) * h.unsqueeze(1)).sum(-1)
        scores = scores.masked_fill(~mask, -1e9)
        a = torch.softmax(scores, -1)
        ctx = (a.unsqueeze(-1) * enc_out).sum(1)
        h, c = self.dec(torch.cat([emb, ctx], -1), (h, c))
        logits = self.out(torch.cat([h, ctx], -1))
        return logits, (h, c)


def _batch(pairs, sv, tv, device):
    src = [sv.enc(src_tokens(t)) for t, _ in pairs]
    tgt = [tv.enc([BOS] + list(s) + [EOS]) for _, s in pairs]
    sl = torch.tensor([len(x) for x in src])
    ms, mt = max(len(x) for x in src), max(len(x) for x in tgt)
    S = torch.zeros(len(src), ms, dtype=torch.long)
    T = torch.zeros(len(tgt), mt, dtype=torch.long)
    for i, x in enumerate(src):
        S[i, :len(x)] = torch.tensor(x)
    for i, x in enumerate(tgt):
        T[i, :len(x)] = torch.tensor(x)
    return S.to(device), sl.to(device), T.to(device)


def train_parser(pairs, sv, tv, device, epochs=30, batch=64, lr=1e-3, seed=0, verbose=False):
    torch.manual_seed(seed)
    model = Seq2Seq(len(sv), len(tv)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss(ignore_index=0)
    for ep in range(epochs):
        random.Random(ep).shuffle(pairs)
        model.train()
        last = 0.0
        for s in range(0, len(pairs), batch):
            S, sl, T = _batch(pairs[s:s + batch], sv, tv, device)
            enc_out, hx = model.encode(S, sl)
            mask = (S != 0)
            loss = 0.0
            for t in range(T.shape[1] - 1):
                logits, hx = model.step(enc_out, mask, T[:, t], hx)
                loss = loss + ce(logits, T[:, t + 1])
            loss = loss / (T.shape[1] - 1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); last = float(loss)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  [parser] epoch {ep:3d} ce={last:.4f}", flush=True)
    return model


@torch.no_grad()
def decode(model, texts, sv, tv, device, max_len=120):
    model.eval()
    outs = []
    for text in texts:
        x = sv.enc(src_tokens(text))
        S = torch.tensor(x, device=device).unsqueeze(0)
        enc_out, hx = model.encode(S, torch.tensor([len(x)]))
        mask = (S != 0)
        y = torch.tensor([tv.stoi[BOS]], device=device)
        chars = []
        for _ in range(max_len):
            logits, hx = model.step(enc_out, mask, y, hx)
            y = logits.argmax(-1)
            ch = tv.itos[int(y)]
            if ch == EOS:
                break
            chars.append(ch)
        outs.append("".join(chars))
    return outs


def parse_verified(model, texts, sv, tv, device, n_predicates: Optional[int] = None):
    """Decode + grammar-constrained accept/reject. Returns list of dicts with the
    decoded string, the parsed program (or None if rejected), and validity."""
    strings = decode(model, texts, sv, tv, device)
    out = []
    for s in strings:
        try:
            prog = parse_program(s, n_predicates=n_predicates)
            out.append({"string": s, "program": prog, "valid": True})
        except (STLSyntaxError, Exception):
            out.append({"string": s, "program": None, "valid": False})
    return out


def build_vocabs(pairs):
    sv = Vocab([tok for t, _ in pairs for tok in src_tokens(t)])
    chars = set()
    for _, s in pairs:
        chars |= set(s)
    tv = Vocab(chars, reserved=(PAD, BOS, EOS))
    return sv, tv

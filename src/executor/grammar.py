"""Abstract syntax tree for bounded-interval Signal Temporal Logic (STL).

A program ``phi`` is a tree of these nodes. Leaves are ``Predicate`` references
into the symbolic-state predicate axis; internal nodes are Boolean
(``Not/And/Or``) or temporal (``Eventually/Always/Until``) operators with bounded
intervals ``[a, b]`` (in timesteps).

The AST is deliberately inert: it carries structure only. Evaluation lives in
``hard_logic`` (exact) and ``soft_logic`` (differentiable). This separation keeps
the symbolic object trustworthy and independently testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


class Node:
    """Base class for all STL AST nodes."""

    def canonical(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def predicates(self) -> set[str]:
        """Set of predicate canonical names referenced in this subtree."""
        out: set[str] = set()
        for child in self._children():
            out |= child.predicates()
        return out

    def _children(self) -> list["Node"]:
        return []

    def __repr__(self) -> str:
        return self.canonical()


@dataclass
class Predicate(Node):
    """A grounded predicate, e.g. high(sensor_4).

    ``name`` is the predicate family (high/low/rising/falling/regime); ``channel``
    is the channel/regime index it applies to. ``index`` is resolved against the
    symbolic-state predicate axis by the executor via a predicate index map.
    """

    name: str
    channel: int

    def canonical(self) -> str:
        return f"{self.name}({self.channel})"

    def predicates(self) -> set[str]:
        return {self.canonical()}


@dataclass
class Not(Node):
    child: Node

    def canonical(self) -> str:
        return f"!{self.child.canonical()}"

    def _children(self) -> list[Node]:
        return [self.child]


@dataclass
class And(Node):
    left: Node
    right: Node

    def canonical(self) -> str:
        return f"({self.left.canonical()} & {self.right.canonical()})"

    def _children(self) -> list[Node]:
        return [self.left, self.right]


@dataclass
class Or(Node):
    left: Node
    right: Node

    def canonical(self) -> str:
        return f"({self.left.canonical()} | {self.right.canonical()})"

    def _children(self) -> list[Node]:
        return [self.left, self.right]


@dataclass
class Eventually(Node):
    """Diamond_[a,b] phi : phi holds at some t' in [t+a, t+b]."""

    a: int
    b: int
    child: Node

    def canonical(self) -> str:
        return f"<>[{self.a},{self.b}]{self.child.canonical()}"

    def _children(self) -> list[Node]:
        return [self.child]


@dataclass
class Always(Node):
    """Box_[a,b] phi : phi holds at every t' in [t+a, t+b]."""

    a: int
    b: int
    child: Node

    def canonical(self) -> str:
        return f"[][{self.a},{self.b}]{self.child.canonical()}"

    def _children(self) -> list[Node]:
        return [self.child]


@dataclass
class Until(Node):
    """phi1 U_[a,b] phi2 : phi2 holds at some t' in [t+a,t+b], and phi1 holds on [t,t']."""

    a: int
    b: int
    left: Node
    right: Node

    def canonical(self) -> str:
        return f"({self.left.canonical()} U[{self.a},{self.b}] {self.right.canonical()})"

    def _children(self) -> list[Node]:
        return [self.left, self.right]


# ---- validation -----------------------------------------------------------

class STLSyntaxError(ValueError):
    """Raised when an STL program is structurally invalid."""


def validate(node: Node, n_predicates: Optional[int] = None) -> None:
    """Recursively check structural validity of an STL program.

    Raises STLSyntaxError on: non-Node children, negative or inverted intervals,
    or predicate channel indices outside [0, n_predicates) when n_predicates given.
    """
    if not isinstance(node, Node):
        raise STLSyntaxError(f"expected Node, got {type(node).__name__}")

    if isinstance(node, Predicate):
        if node.channel < 0:
            raise STLSyntaxError(f"negative channel in {node.canonical()}")
        if n_predicates is not None and node.channel >= n_predicates:
            raise STLSyntaxError(
                f"predicate channel {node.channel} out of range [0,{n_predicates})"
            )
        return

    if isinstance(node, (Eventually, Always, Until)):
        if node.a < 0 or node.b < 0:
            raise STLSyntaxError(f"negative interval bound in {node.canonical()}")
        if node.a > node.b:
            raise STLSyntaxError(
                f"inverted interval [{node.a},{node.b}] in {node.canonical()}"
            )

    for child in node._children():
        validate(child, n_predicates=n_predicates)


# ---- inverse of canonical(): S-expression string -> Node ------------------
# Grammar (matches Node.canonical), fully parenthesised binaries + prefix unaries:
#   E := name(chan) | !E | <>[a,b]E | [][a,b]E | ( E & E ) | ( E | E ) | ( E U[a,b] E )
import re as _re

_PRED_RE = _re.compile(r"^([A-Za-z_][A-Za-z_0-9]*)\((\d+)\)")
_INT_RE = _re.compile(r"^\[(\d+),(\d+)\]")


def _split_top(inner: str) -> tuple[str, str, str]:
    """Split a parenthesised binary body 'L op R' at the top-level operator."""
    depth = 0
    i = 0
    while i < len(inner):
        ch = inner[i]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif depth == 0:
            if inner.startswith(" & ", i):
                return inner[:i], "&", inner[i + 3:]
            if inner.startswith(" | ", i):
                return inner[:i], "|", inner[i + 3:]
            if inner.startswith(" U[", i):
                m = _INT_RE.match(inner[i + 2:])
                if m:
                    a, b = int(m.group(1)), int(m.group(2))
                    rest = inner[i + 2 + m.end():]
                    if rest.startswith(" "):
                        return inner[:i], f"U:{a}:{b}", rest[1:]
        i += 1
    raise STLSyntaxError(f"no top-level operator in '{inner}'")


def _parse(s: str) -> tuple[Node, str]:
    s = s.lstrip()
    if s.startswith("!"):
        child, rest = _parse(s[1:])
        return Not(child), rest
    if s.startswith("<>[") or s.startswith("[]["):
        cls = Eventually if s[0] == "<" else Always
        m = _INT_RE.match(s[2:])
        if not m:
            raise STLSyntaxError(f"bad temporal interval in '{s}'")
        a, b = int(m.group(1)), int(m.group(2))
        child, rest = _parse(s[2 + m.end():])
        return cls(a, b, child), rest
    if s.startswith("("):
        depth, j = 0, 0
        for j, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
        inner = s[1:j]
        left, op, right = _split_top(inner)
        ln, _ = _parse(left)
        rn, _ = _parse(right)
        if op == "&":
            node = And(ln, rn)
        elif op == "|":
            node = Or(ln, rn)
        else:
            _, a, b = op.split(":")
            node = Until(int(a), int(b), ln, rn)
        return node, s[j + 1:]
    m = _PRED_RE.match(s)
    if m:
        return Predicate(m.group(1), int(m.group(2))), s[m.end():]
    raise STLSyntaxError(f"cannot parse '{s}'")


def parse_program(s: str, n_predicates: Optional[int] = None) -> Node:
    """Parse a canonical STL S-expression back into a Node (inverse of
    ``Node.canonical``); validates the result. Raises ``STLSyntaxError`` on any
    malformed string -- this is what lets a NL parser REJECT invalid decodings
    rather than guess (WO-6)."""
    node, rest = _parse(s.strip())
    if rest.strip():
        raise STLSyntaxError(f"trailing tokens after parse: '{rest}'")
    validate(node, n_predicates=n_predicates)
    return node
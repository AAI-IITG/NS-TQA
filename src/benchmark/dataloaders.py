"""PyTorch Dataset/DataLoader wrappers over generated QA instances."""
from __future__ import annotations
import torch
from torch.utils.data import Dataset


class QADataset(Dataset):
    """Wraps a list of instances (synthetic or semi-synthetic) for batching.

    Each item exposes the signal X, the grounded state mu (if present), the
    program phi*, and the boolean answer. Programs are returned as objects;
    collate them with a custom collate_fn since they are not tensors.
    """
    def __init__(self, instances: list):
        self.instances = instances

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, i: int):
        it = self.instances[i]
        if isinstance(it, dict):
            return it
        # SyntheticInstance dataclass
        return {
            "X": it.X, "mu_star": getattr(it, "mu_star", None),
            "phi_star": it.phi_star, "answer_star": it.answer_star,
            "question": it.question,
        }


def qa_collate(batch: list[dict]) -> dict:
    """Collate that stacks signals but keeps programs/questions as lists."""
    out = {"X": torch.stack([b["X"] for b in batch]),
           "answer_star": torch.tensor([b["answer_star"] for b in batch]),
           "phi_star": [b["phi_star"] for b in batch],
           "question": [b["question"] for b in batch]}
    return out

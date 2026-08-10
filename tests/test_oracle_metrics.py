import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.oracle_metrics import assemble_oracle_metrics


def _record(pred_t=3, gold_t=3):
    return {
        "pred_answer": True,
        "gold_answer": True,
        "pred_rho": 0.5,
        "gold_rho": 0.25,
        "pred_critical_t": pred_t,
        "gold_critical_t": gold_t,
        "template": "existence",
        "engine_id": "engine-1",
    }


def test_oracle_conjunction_is_one_for_exact_answer_and_critical_t():
    metrics = assemble_oracle_metrics([_record()])
    assert metrics["answer_accuracy"] == 1.0
    assert metrics["rho_sign_accuracy"] == 1.0
    assert metrics["critical_t_exact"] == 1.0
    assert metrics["critical_t_within_2"] == 1.0
    assert metrics["program_exact_match"] == 1.0
    assert metrics["conjunction_score"] == 1.0


def test_oracle_critical_t_mismatch_lowers_conjunction():
    metrics = assemble_oracle_metrics([_record(pred_t=5, gold_t=1)])
    assert metrics["answer_accuracy"] == 1.0
    assert metrics["critical_t_exact"] == 0.0
    assert metrics["critical_t_within_2"] == 0.0
    assert metrics["conjunction_score"] == 0.0

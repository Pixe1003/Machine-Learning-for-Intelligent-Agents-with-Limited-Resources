import torch

from src.core.loss import evaluate, report_evaluate
from src.monitors import M1Monitor, M2Monitor


def _load_binary_tracker(monitor) -> None:
    with torch.no_grad():
        monitor.W_H.copy_(torch.tensor([[-20.0, 20.0], [-20.0, 20.0]], dtype=torch.float64))
        monitor.W_T.copy_(torch.tensor([[20.0, -20.0], [20.0, -20.0]], dtype=torch.float64))
        monitor.omega.copy_(torch.tensor([-0.4, 0.4], dtype=torch.float64))


def test_report_evaluate_thresholds_m1_only() -> None:
    m1 = M1Monitor(k=2, seed=0)
    _load_binary_tracker(m1)

    soft_loss = evaluate(m1, 3)
    report_loss = report_evaluate(m1, 3)

    assert report_loss != soft_loss
    assert report_loss < soft_loss
    assert m1.hard_accept is False


def test_report_evaluate_leaves_soft_acceptance_monitors_unchanged() -> None:
    m2 = M2Monitor(k=2, seed=0)
    _load_binary_tracker(m2)

    assert report_evaluate(m2, 3) == evaluate(m2, 3)

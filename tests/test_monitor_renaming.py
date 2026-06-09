from src.monitors import M1HardMonitor, M1Monitor, M2Monitor, M3Monitor
from src.monitors.m1_unconstrained import M1HardMonitor as M1HardFromModule
from src.monitors.m2_soft_accept import M1Monitor as M1FromModule
from src.monitors.m3_counting import M2Monitor as M2FromModule
from src.monitors.m4_minimal import M3Monitor as M3FromModule


def test_monitor_exports_follow_presentation_hierarchy() -> None:
    assert M1Monitor is M1FromModule
    assert M1HardMonitor is M1HardFromModule
    assert M2Monitor is M2FromModule
    assert M3Monitor is M3FromModule


def test_monitor_acceptance_semantics_match_new_names() -> None:
    assert not hasattr(M1Monitor(k=2), "hard_accept")
    assert hasattr(M1HardMonitor(k=2), "hard_accept")

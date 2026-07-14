"""Unit + golden tests for the decomposed _stage_focus directive renderer.

The refactor of _stage_focus (agentic_orchestrator) into focus-selection + headline + per-evidence
renderers is required to be BEHAVIOUR-PRESERVING. These lock in the focus-mode selection, the
individual evidence renderers, and the early-return paths so future edits cannot silently drift the
LLM-facing directive. Runs under pytest or standalone: `.venv/bin/python .../test_stage_focus.py`.
"""
from __future__ import annotations

import os
import sys

# Standalone-runnable (pytest not installed in this venv): put policy_bias_lab + bootstrapping on path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_LLMFW = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_LLMFW, os.path.join(_LLMFW, "..", "bootstrapping")):
    if p not in sys.path:
        sys.path.insert(0, p)

from policy_bias_lab.agentic_orchestrator import (  # noqa: E402
    _select_focus, _focus_headline, _ev_discrepancy, _ev_self_lock, _ev_edit_menu, _stage_focus,
)

NAMES = ["approach", "grasp", "lift"]


def test_select_focus_modes():
    assert _select_focus({"reaches_terminal": True, "weakest_stage": 2}) == ("terminal_weakest", 2)
    assert _select_focus({"reaches_terminal": True, "weakest_stage": None}) == ("terminal_none", None)
    assert _select_focus({"reaches_terminal": False, "stall_stage": None}) == ("stall_none", None)
    # stall, no fall-back to predecessor -> normal stall
    normal = {"stall_stage": 1, "entered_frac": [1.0, 0.6, 0.0], "reverse_frac": [0.0, 0.0, None],
              "occupancy": [0.4, 0.4, 0.0]}
    assert _select_focus(normal) == ("stall_normal", 1)
    # flicker: entered but bounced back to k-1 (reverse[0] >= 0.5*entered[1]) and barely occupied
    flick = {"stall_stage": 1, "entered_frac": [1.0, 0.6, 0.0], "reverse_frac": [0.4, 0.0, None],
             "occupancy": [0.4, 0.05, 0.0]}
    assert _select_focus(flick) == ("flicker", 1)


def test_headline_directive_side():
    rep = {"stage_names": NAMES, "stall_stage": 1, "entered_frac": [1.0, 0.6, 0.0],
           "handoff_frac": [0.6, 0.0, None], "conversion": [0.6, 0.0, None],
           "reverse_frac": [0.0, 0.0, None], "occupancy": [0.4, 0.4, 0.0], "stall_name": "grasp"}
    entry = "\n".join(_focus_headline("stall_normal", 1, rep, "TBL", "entry"))
    exit_ = "\n".join(_focus_headline("stall_normal", 1, rep, "TBL", "exit"))
    neutral = "\n".join(_focus_headline("stall_normal", 1, rep, "TBL", None))
    assert "FOCUS = ENTRY side" in entry and "stage 2" in entry
    assert "FOCUS = EXIT side" in exit_
    assert "Revise ONLY stage 1" in neutral and "FOCUS =" not in neutral


def test_ev_discrepancy_subcases():
    base = {"authored_success_frac": [0.1, 0.1, 0.1], "handoff_frac": [0.9, 0.9, None],
            "entered_frac": [1.0, 1.0, 0.5]}
    for tag, needle in (("handoff_without_success", "hands off BEFORE"),
                        ("success_without_handoff", "entry gate looks"),
                        ("entered_without_success", "runs without")):
        rep = dict(base, success_discrepancy=[tag, None, None])
        out = _ev_discrepancy(rep, {}, 0)
        assert out and "DISCREPANCY" in out and needle in out
    assert _ev_discrepancy(dict(base, success_discrepancy=[None]), {}, 0) is None


def test_ev_self_lock():
    assert _ev_self_lock({"self_lock": True, "stage_names": NAMES}, {}, 0).startswith("SELF-LOCK")
    assert _ev_self_lock({"self_lock": False}, {}, 0) is None


def test_ev_edit_menu_selection():
    assert "(d)" in _ev_edit_menu({"self_lock": True}, {}, 0)
    rising = {"next_gate": {"value_early_late": [0.1, 0.5]}}
    assert "(b)" in _ev_edit_menu(rising, {}, 0)
    conv = {"next_gate": {"value_early_late": [0.5, 0.5]}}
    assert "(c)" in _ev_edit_menu(conv, {"training_report": {"verdict": "converged"}}, 0)
    assert "(a)" in _ev_edit_menu({"next_gate": {"value_early_late": [0.5, 0.5]}}, {}, 0)


def test_stage_focus_early_returns_and_order():
    assert "No per-stage report" in _stage_focus({})
    term = _stage_focus({"stage_report": {"stage_names": NAMES, "reaches_terminal": True,
                                          "weakest_stage": None}})
    assert "completes the stage chain" in term
    # Full path: headline first, edit-menu last; self-lock appears before the menu.
    rep = {"stage_names": NAMES, "stall_stage": 1, "stall_name": "grasp", "self_lock": True,
           "entered_frac": [1.0, 0.98, 0.0], "handoff_frac": [0.9, 0.0, None],
           "conversion": [0.9, 0.0, None], "reverse_frac": [0.0, 0.0, None],
           "occupancy": [0.02, 0.98, 0.0], "reached_frac": [1.0, 1.0, 0.0]}
    out = _stage_focus({"stage_report": rep})
    assert out.startswith("Stage residence")
    assert out.rstrip().endswith(".") and "Suggested EDIT MENU entry" in out.splitlines()[-1]
    assert out.index("SELF-LOCK") < out.index("Suggested EDIT MENU entry")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

from __future__ import annotations

from typing import Any

from llm_framework.core.state import WorldState


FUNDAMENTAL_UNIT_ORDER = ("base", "wrist", "thumb", "index", "middle", "ring", "little")


def fundamental_units(state: WorldState) -> list[str]:
    units = [unit for unit in FUNDAMENTAL_UNIT_ORDER if unit in state.appendages and state.appendages[unit]]
    if not units and state.appendages.get("hand"):
        units = ["hand"]
    return units


def joint_schema_for_state(
    state: WorldState,
    *,
    ctrl_open: dict[str, float] | None = None,
    ctrl_close: dict[str, float] | None = None,
) -> dict[str, Any]:
    ctrl_by_name = {name: float(value) for name, value in zip(state.actuator_names, state.ctrl, strict=False)}
    lo_by_name = {name: float(value) for name, value in zip(state.actuator_names, state.ctrl_lo, strict=False)}
    hi_by_name = {name: float(value) for name, value in zip(state.actuator_names, state.ctrl_hi, strict=False)}
    units: dict[str, Any] = {}
    for unit, joints in state.appendages.items():
        units[unit] = {
            "role": _unit_role(unit),
            "joints": [
                {
                    "name": joint,
                    "current": round(ctrl_by_name.get(joint, 0.0), 4),
                    "range": [round(lo_by_name.get(joint, 0.0), 4), round(hi_by_name.get(joint, 0.0), 4)],
                    "meaning": _joint_meaning(
                        joint,
                        lo_by_name.get(joint, 0.0),
                        hi_by_name.get(joint, 1.0),
                        None if ctrl_open is None else ctrl_open.get(joint),
                        None if ctrl_close is None else ctrl_close.get(joint),
                    ),
                }
                for joint in joints
            ],
        }
    return {"fundamental_units": fundamental_units(state), "units": units}


def schema_for_unit(state: WorldState, unit: str) -> dict[str, Any]:
    schema = joint_schema_for_state(state)
    return {
        "fundamental_units": schema["fundamental_units"],
        "unit": unit,
        "schema": schema["units"].get(unit, {"role": _unit_role(unit), "joints": []}),
    }


def _joint_meaning(name: str, lo: float, hi: float, open_value: float | None = None, close_value: float | None = None) -> dict[str, str]:
    span = hi - lo
    if abs(span) < 1e-9:
        return {"fixed": f"{name} is fixed at {lo:.4f}"}
    q1 = lo + 0.25 * span
    q2 = lo + 0.50 * span
    q3 = lo + 0.75 * span
    if name in {"base_x", "base_y"}:
        axis = "x" if name.endswith("_x") else "y"
        return {
            f"{lo:.4f}..{q1:.4f}": f"base is far negative along world {axis}",
            f"{q1:.4f}..{q2:.4f}": f"base is mildly negative along world {axis}",
            f"{q2:.4f}..{q3:.4f}": f"base is mildly positive along world {axis}",
            f"{q3:.4f}..{hi:.4f}": f"base is far positive along world {axis}",
        }
    if name == "base_z":
        return {
            f"{lo:.4f}..{q1:.4f}": "base/palm is low toward the table/object",
            f"{q1:.4f}..{q2:.4f}": "base/palm is moderately lowered",
            f"{q2:.4f}..{q3:.4f}": "base/palm is moderately high",
            f"{q3:.4f}..{hi:.4f}": "base/palm is high or retracted",
        }
    if open_value is not None and close_value is not None and close_value < open_value:
        return {
            f"{lo:.4f}..{q1:.4f}": "joint/actuator is flexed a large amount or near closed",
            f"{q1:.4f}..{q2:.4f}": "joint/actuator is flexed a medium amount",
            f"{q2:.4f}..{q3:.4f}": "joint/actuator is flexed a small amount",
            f"{q3:.4f}..{hi:.4f}": "joint/actuator is near extended/open",
            "open_target": f"{open_value:.4f}",
            "closed_target": f"{close_value:.4f}",
        }
    return {
        f"{lo:.4f}..{q1:.4f}": "joint/actuator is near extended/open",
        f"{q1:.4f}..{q2:.4f}": "joint/actuator is flexed a small amount",
        f"{q2:.4f}..{q3:.4f}": "joint/actuator is flexed a medium amount",
        f"{q3:.4f}..{hi:.4f}": "joint/actuator is flexed a large amount or near closed",
        **({"open_target": f"{open_value:.4f}", "closed_target": f"{close_value:.4f}"} if open_value is not None and close_value is not None else {}),
    }


def _unit_role(unit: str) -> str:
    if unit == "base":
        return "moves the whole hand/palm relative to the world; absolute commands should be resolved before finger commands"
    if unit == "wrist":
        return "orients the palm/base of the fingers"
    if unit in {"thumb", "index", "middle", "ring", "little"}:
        return f"controls the {unit} finger relative to the palm"
    return "appendage-local actuator group"

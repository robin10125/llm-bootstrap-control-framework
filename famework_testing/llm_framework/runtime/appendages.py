from __future__ import annotations

from typing import Any


SHADOW_APPENDAGE_HINTS = {
    "wrist": ("WRJ",),
    "thumb": ("THJ",),
    "index": ("FFJ",),
    "middle": ("MFJ",),
    "ring": ("RFJ",),
    "little": ("LFJ",),
}


def appendage_joint_map(env: Any) -> dict[str, list[str]]:
    """Map appendage names to controllable actuator names.

    The framework exposes these as joint-level controls even when the MuJoCo model uses
    actuator names. For tendon-coupled joints, the actuator remains the control boundary.
    """
    names = [env.model.actuator(i).name for i in range(env.nu)]
    groups: dict[str, list[str]] = {k: [] for k in SHADOW_APPENDAGE_HINTS}
    groups.update({"hand": [], "base": []})

    for name in names:
        if name in {"base_x", "base_y", "base_z"}:
            groups["base"].append(name)
            continue
        matched = False
        for appendage, hints in SHADOW_APPENDAGE_HINTS.items():
            if any(hint in name for hint in hints):
                groups[appendage].append(name)
                groups["hand"].append(name)
                matched = True
                break
        if not matched and name not in groups["hand"]:
            groups["hand"].append(name)

    return {k: v for k, v in groups.items() if v}


def appendage_names(env: Any) -> list[str]:
    return sorted(appendage_joint_map(env))

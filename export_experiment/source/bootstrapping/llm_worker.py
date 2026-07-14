#!/usr/bin/env python3
"""Always-busy async LLM worker that supplies waypoint-plan demonstrations.

The LLM is the expensive, slow system; the GPU trainer must never block on it. This worker
runs on a background thread doing only subprocess/HTTP LLM calls (no JAX — JAX preallocates
the GPU and must stay single-process on the main thread). Whenever it finishes a query and
is idle, it pulls the next queued sample, so the LLM is continuously utilized up to a fixed
**call budget**. Parsed plans go back to the main thread, which scores them in MJX and, if a
plan beats the current policy, clones it.

Two modes (`mode` on each submitted sample):
- `scratch`: author a fresh waypoint plan from the scene description (early acquisition).
- `residual`: correct a supplied policy rollout — the LLM looks at what the policy did and
  proposes a better plan (later refinement). The trainer enables this once the policy is
  competent.
"""
from __future__ import annotations

import json
import queue
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_backend import LLMResponse, LLMUsageTracker, call_llm


# --- waypoint plan parsing ---------------------------------------------------

def parse_waypoint_plan(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object containing a 'waypoints' list from a completion."""
    candidates: list[str] = []
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S):
        candidates.append(m.group(1))
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        candidates.append(m.group(0))
    for raw in candidates:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("waypoints"), list) and obj["waypoints"]:
            return obj
    return None


# --- prompt builders ---------------------------------------------------------

_SCHEMA = (
    "Return ONLY a JSON object of this form (no prose):\n"
    '{"waypoints": [{"t": <sec>, "pos": [x, y, z], "open": <0..1>}, ...]}\n'
    "Semantics:\n"
    "- pos = EFFECTOR base target. x,y in [-0.30,0.30] m (world, table plane); "
    "z in [-0.15,0.20] m where LARGER z LOWERS the effector toward the table (the base is "
    "mounted inverted). z=0 is the raised/home height; z~0.15-0.18 brings an open EFFECTOR "
    "down around an object resting on the table (deep enough to surround it).\n"
    "- open in [0,1]: 1 = EFFECTOR fully open, 0 = fully closed into a grasp.\n"
    "- t = seconds from episode start; the episode is EP_SECONDS s long.\n"
    "Use 3-6 waypoints. A lift is: open above the object, descend (raise z), close, raise "
    "z back to 0 to lift it clear of the table."
)

_EFFECTOR = {"gripper": "parallel-jaw gripper", "hand": "five-fingered robotic hand"}


def _schema(ep: float, embodiment: str = "gripper") -> str:
    return _SCHEMA.replace("EP_SECONDS", f"{ep:.1f}").replace(
        "EFFECTOR", _EFFECTOR.get(embodiment, "gripper"))


def build_scratch_prompt(ctx: dict[str, Any]) -> str:
    obj = ctx["object_xy"]
    emb = ctx.get("embodiment", "gripper")
    return (
        f"You control a {_EFFECTOR.get(emb, 'gripper')} on a movable base to LIFT a cube off "
        "a table and hold it up.\n"
        f"The cube is at x={obj[0]:+.3f}, y={obj[1]:+.3f} m, resting on the table "
        f"(top ~{ctx.get('object_top_z', 0.05):.3f} m).\n\n"
        f"{_schema(ctx['episode_seconds'], emb)}\n"
    )


def build_residual_prompt(ctx: dict[str, Any]) -> str:
    obj = ctx["object_xy"]
    trace = ctx.get("policy_trace", "")
    lifted = ctx.get("policy_lifted", False)
    peak_z = ctx.get("policy_peak_obj_z", ctx.get("object_top_z", 0.05))
    outcome = (
        "It DID lift the cube, but you should make the motion cleaner/more reliable."
        if lifted else
        f"It FAILED to lift the cube (peak object height only {peak_z:.3f} m). Diagnose "
        "what went wrong from the trace below (e.g. it did not descend far enough, closed "
        "too early/late, or missed in x/y) and fix it."
    )
    emb = ctx.get("embodiment", "gripper")
    return (
        f"You are correcting a learned {_EFFECTOR.get(emb, 'gripper')} policy that is trying "
        "to LIFT a cube.\n"
        f"The cube is at x={obj[0]:+.3f}, y={obj[1]:+.3f} m. {outcome}\n"
        f"The policy's own rollout (sampled every ~0.3 s) was:\n{trace}\n\n"
        "Propose a corrected waypoint plan that lifts the cube cleanly.\n\n"
        f"{_schema(ctx['episode_seconds'], emb)}\n"
    )


_PROMPT_BUILDERS = {"scratch": build_scratch_prompt, "residual": build_residual_prompt}


# --- reward / curriculum authoring (Eureka-style) ----------------------------

_REWARD_SCHEMA = (
    "Return ONLY a JSON object (no prose):\n"
    '{"milestones": [{"name": str, "predicate": {"op": "lt|le|gt|ge", "field": str, '
    '"thr": number}, "reward": [TERM, ...]}, ...], "penalties": [TERM, ...]}\n'
    "A TERM is one of:\n"
    '  {"op":"const","w":n} | {"op":"linear","field":F,"w":n} | '
    '{"op":"neg_dist","field":F,"w":n} (reward grows as a distance shrinks) | '
    '{"op":"proximity","field":F,"w":n,"scale":n} (w*exp(-field/scale)) | '
    '{"op":"clip","field":F,"lo":n,"hi":n,"w":n} | '
    '{"op":"ge|gt|lt|le","field":F,"thr":n,"w":n} (indicator bonus).\n'
    "Rules: order milestones EASIEST->HARDEST (each a prerequisite of the next); use only the "
    "fields listed; give partial-credit shaping so an incomplete attempt that makes progress "
    "still scores (e.g. proximity/neg_dist on distances, linear on n_contacts). The last "
    "milestone should be true task success."
)


def parse_reward_spec(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object containing a 'milestones' list."""
    candidates: list[str] = []
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S):
        candidates.append(m.group(1))
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        candidates.append(m.group(0))
    for raw in candidates:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("milestones"), list) and obj["milestones"]:
            return obj
    return None


def build_reward_prompt(ctx: dict[str, Any]) -> str:
    """Author (or, with `reflection`, revise) a reward/curriculum spec for the task."""
    parts = [
        f"Design a REWARD CURRICULUM for a {ctx.get('embodiment', 'gripper')} that must LIFT a "
        "cube off a table. Break the task into ordered milestones that give dense partial-credit "
        "feedback, so a policy that only partly solves it (gets closer, gets fingers on, lifts a "
        "little) still earns graded reward and can be ranked.\n",
        ctx["fields_doc"] + "\n",
    ]
    if ctx.get("reflection"):
        parts.append(
            "Your PREVIOUS spec was:\n" + json.dumps(ctx["prior_spec"]) + "\n"
            "Training results per milestone (fraction of envs reaching it) and true success:\n"
            + ctx["reflection"] + "\n"
            "Revise the spec to unblock the stalled milestone (re-weight, split it, or fix a term "
            "that doesn't correlate with real progress). Keep what works.\n")
    parts.append(_REWARD_SCHEMA)
    return "\n".join(parts)


# --- worker ------------------------------------------------------------------

@dataclass
class Candidate:
    ctx: dict[str, Any]
    plan: dict[str, Any] | None
    response_ok: bool
    error: str | None
    call_index: int


class LLMWorker:
    """Background thread: drains submitted samples, queries the LLM, returns parsed plans."""

    def __init__(self, backend: str, model: str | None, budget: int,
                 log_dir: Path | None = None, timeout_s: float = 360.0):
        self.backend = backend
        self.model = model
        self.budget = budget
        self.log_dir = Path(log_dir) if log_dir else None
        self.timeout_s = timeout_s
        self._in: queue.Queue = queue.Queue()
        self._out: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.calls_made = 0
        self.usage = LLMUsageTracker()
        self._lock = threading.Lock()

    def start(self) -> None:
        self._thread.start()

    def submit(self, ctx: dict[str, Any]) -> bool:
        """Queue a sample if budget remains and the worker is idle-hungry. Returns accepted."""
        with self._lock:
            queued = self._in.qsize()
            if self.calls_made + queued >= self.budget:
                return False
        self._in.put(ctx)
        return True

    def poll(self) -> list[Candidate]:
        out = []
        while True:
            try:
                out.append(self._out.get_nowait())
            except queue.Empty:
                break
        return out

    def budget_left(self) -> int:
        with self._lock:
            return max(0, self.budget - self.calls_made - self._in.qsize())

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                ctx = self._in.get(timeout=0.5)
            except queue.Empty:
                continue
            with self._lock:
                if self.calls_made >= self.budget:
                    continue
                idx = self.calls_made
                self.calls_made += 1
            mode = ctx.get("mode", "scratch")
            try:
                prompt = _PROMPT_BUILDERS[mode](ctx)
                resp: LLMResponse = call_llm(
                    self.backend, prompt, model=self.model, timeout_s=self.timeout_s,
                    log_dir=self.log_dir, tag=f"llm_{idx:03d}_{mode}",
                )
                self.usage.add(resp, tag=f"{mode}_{idx}")
                plan = parse_waypoint_plan(resp.text) if resp.ok else None
                err = resp.error or (None if plan else "no waypoint JSON in completion")
            except Exception as exc:  # keep the worker alive on any failure
                plan, err, resp_ok = None, f"{type(exc).__name__}: {exc}", False
                self._out.put(Candidate(ctx=ctx, plan=None, response_ok=False, error=err, call_index=idx))
                continue
            self._out.put(Candidate(
                ctx=ctx, plan=plan, response_ok=resp.ok, error=err, call_index=idx,
            ))


@dataclass
class DemoBuffer:
    """Fixed-capacity ring buffer of (obs, action) BC demonstrations (numpy)."""

    capacity: int
    obs_dim: int
    act_dim: int
    _obs: Any = None
    _act: Any = None
    size: int = 0
    _ptr: int = 0
    _accepted: int = 0

    def __post_init__(self):
        import numpy as np
        self._obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._act = np.zeros((self.capacity, self.act_dim), dtype=np.float32)
        self._wt = np.zeros((self.capacity,), dtype=np.float32)  # per-demo reward weight

    def add(self, obs, act, weight: float = 0.0) -> None:
        import numpy as np
        obs = np.asarray(obs, dtype=np.float32)
        act = np.asarray(act, dtype=np.float32)
        n = obs.shape[0]
        self._accepted += 1
        for i in range(n):
            self._obs[self._ptr] = obs[i]
            self._act[self._ptr] = act[i]
            self._wt[self._ptr] = weight
            self._ptr = (self._ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    @property
    def accepted_demos(self) -> int:
        return self._accepted

    def sample(self, n: int):
        import numpy as np
        if self.size == 0:
            return None
        idx = np.random.randint(0, self.size, size=n)
        return self._obs[idx], self._act[idx], self._wt[idx]

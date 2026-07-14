"""Live step-through viewer for a recorded episode.

This does NOT replay stored positions. It holds a live MuJoCo sim and re-integrates real
physics on demand: the episode is reproducible from (initial state + the agent's
piecewise-constant control timeline), so each click actually advances the simulation by a
chosen number of physics steps and re-renders. The sim is paused between clicks — you step
through it.

A tiny local HTTP server owns the sim + renderer (single-threaded, so the GL context stays
on one thread); the browser page is just buttons + an <img>.

  python -m agent_hand.stepper --latest          # newest run under runs/
  python -m agent_hand.stepper --run runs/2026... # a specific run
  python -m agent_hand.stepper --latest --port 8008

Needs a GL backend (MUJOCO_GL=glfw on a desktop / egl on a headless GPU box). Ctrl-C stops.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("MUJOCO_GL", "glfw")  # before mujoco initialises its renderer

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from .sim import HandSim  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _latest_run(runs_root: Path) -> Path:
    runs = [p for p in runs_root.glob("*") if (p / "trajectory.npz").exists()]
    if not runs:
        raise SystemExit(f"no runs with a trajectory.npz under {runs_root}")
    return max(runs, key=lambda p: p.stat().st_mtime)


class LiveEpisode:
    """A pausable, single-steppable re-integration of one recorded episode."""

    def __init__(self, run_dir: Path, width: int, height: int, camera: str):
        data = np.load(run_dir / "trajectory.npz")
        if "ctrl_steps" not in data:
            raise SystemExit(
                f"{run_dir/'trajectory.npz'} has no control timeline — re-run the episode "
                "with this version (older runs recorded positions only).")
        self._init_qpos = data["init_qpos"]
        self._init_qvel = data["init_qvel"]
        self._ctrl_steps = data["ctrl_steps"]
        self._ctrl_vals = data["ctrl_vals"]
        self.total = int(data["total_steps"])

        self.sim = HandSim()
        self.camera = camera if camera in [
            self.sim.model.camera(i).name for i in range(self.sim.model.ncam)] else -1
        self.renderer = mujoco.Renderer(self.sim.model, height, width)
        self.cur = 0
        self.reset_to_start()

    # -- physics --------------------------------------------------------------
    def reset_to_start(self) -> None:
        self.sim.data.qpos[:] = self._init_qpos
        self.sim.data.qvel[:] = self._init_qvel
        self.sim.data.ctrl[:] = self._ctrl_vals[0]
        mujoco.mj_forward(self.sim.model, self.sim.data)
        self.cur = 0

    def _ctrl_at(self, step: int):
        idx = int(np.searchsorted(self._ctrl_steps, step, side="right")) - 1
        return self._ctrl_vals[max(0, idx)]

    def advance_to(self, target: int) -> None:
        target = max(0, min(self.total, int(target)))
        if target < self.cur:                # seeking back = re-integrate from the start
            self.reset_to_start()
        while self.cur < target:
            self.sim.data.ctrl[:] = self._ctrl_at(self.cur)
            mujoco.mj_step(self.sim.model, self.sim.data)
            self.cur += 1

    # -- views ----------------------------------------------------------------
    def frame_png(self) -> str:
        self.renderer.update_scene(self.sim.data, camera=self.camera)
        buf = io.BytesIO()
        Image.fromarray(self.renderer.render()).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def state(self, with_image: bool = True) -> dict:
        s = {
            "step": self.cur, "total": self.total,
            "cube_z": round(self.sim.obj_pos("cube")[2], 4),
            "grasped": bool(self.sim.grasped("cube")),
            "ctrl": {n: round(self.sim.get_ctrl(n), 4) for n in self.sim.actuator_names()},
        }
        if with_image:
            s["img"] = "data:image/png;base64," + self.frame_png()
        return s


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        ep: LiveEpisode = self.server.episode  # type: ignore[attr-defined]
        q = parse_qs(url.query)
        if url.path == "/":
            self._send(200, _PAGE.encode(), "text/html; charset=utf-8")
        elif url.path == "/api/step":
            ep.advance_to(ep.cur + int(q.get("n", ["1"])[0]))
            self._json(ep.state())
        elif url.path == "/api/seek":
            ep.advance_to(int(q.get("to", ["0"])[0]))
            self._json(ep.state())
        elif url.path == "/api/reset":
            ep.reset_to_start()
            self._json(ep.state())
        elif url.path == "/api/state":
            self._json(ep.state())
        else:
            self._send(404, b"not found", "text/plain")

    def _json(self, obj: dict) -> None:
        self._send(200, json.dumps(obj).encode(), "application/json")


_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>hand stepper (live)</title>
<style>
 body{background:#1b1b1f;color:#ddd;font:14px system-ui,sans-serif;text-align:center;margin:0;padding:18px}
 h1{font-size:15px;font-weight:600;color:#aaa;margin:0 0 12px}
 img{background:#000;border-radius:8px;max-width:90vw;box-shadow:0 4px 24px #0008}
 .bar{margin:14px auto;display:flex;gap:8px;align-items:center;justify-content:center;flex-wrap:wrap}
 button{background:#2d2d35;color:#eee;border:1px solid #444;border-radius:6px;padding:7px 14px;font-size:14px;cursor:pointer}
 button:hover{background:#3a3a44}
 input[type=range]{width:min(60vw,520px)} input[type=number]{width:64px;background:#2d2d35;color:#eee;border:1px solid #444;border-radius:5px;padding:5px}
 .stat{font-variant-numeric:tabular-nums;color:#9fd}.grasp-yes{color:#6f6}.grasp-no{color:#f88}
 label{color:#999}
</style></head><body>
<h1>live step-through &mdash; the sim is paused; each click integrates physics</h1>
<img id="view">
<div class="bar">
 <button onclick="reset()">&#8634; Reset</button>
 <button onclick="stepN(-1)">&laquo; Back</button>
 <button id="play" onclick="toggle()">&#9658; Play</button>
 <button onclick="stepN(1)">Step &raquo;</button>
 <label>steps/click <input type="number" id="chunk" value="10" min="1" max="500"></label>
</div>
<div class="bar">
 <input type="range" id="slider" min="0" value="0" oninput="seek(+this.value)">
</div>
<div class="bar stat">
 step <span id="step"></span>/<span id="total"></span> &nbsp;|&nbsp; sim&nbsp;t=<span id="t"></span>s
 &nbsp;|&nbsp; cube_z=<span id="z"></span> m &nbsp;|&nbsp; grasped=<span id="g"></span>
</div>
<div class="bar stat" id="ctrl" style="color:#89a;font-size:12px"></div>
<script>
let timer=null;
const $=id=>document.getElementById(id);
const chunk=()=>Math.max(1,+$('chunk').value||1);
async function call(path){ const r=await fetch(path); return r.json(); }
function paint(s){
 $('view').src=s.img; $('step').textContent=s.step; $('total').textContent=s.total;
 $('t').textContent=(s.step*0.002).toFixed(3); $('z').textContent=s.cube_z.toFixed(3);
 const g=$('g'); g.textContent=s.grasped?'YES':'no'; g.className=s.grasped?'grasp-yes':'grasp-no';
 $('slider').max=s.total; $('slider').value=s.step;
 $('ctrl').textContent=Object.entries(s.ctrl).map(([k,v])=>k+'='+v.toFixed(3)).join('   ');
 if(timer && s.step>=s.total) pause();
}
async function stepN(d){ pause(); paint(await call('/api/step?n='+d*chunk())); }
async function seek(k){ pause(); paint(await call('/api/seek?to='+k)); }
async function reset(){ pause(); paint(await call('/api/reset')); }
function toggle(){ timer?pause():play(); }
function play(){ $('play').innerHTML='&#10073;&#10073; Pause';
 timer=setInterval(async()=>{ paint(await call('/api/step?n='+chunk())); },90); }
function pause(){ if(timer){clearInterval(timer);timer=null;} $('play').innerHTML='&#9658; Play'; }
document.addEventListener('keydown',e=>{ if(e.key==='ArrowRight')stepN(1);
 if(e.key==='ArrowLeft')stepN(-1); if(e.key===' '){e.preventDefault();toggle();} });
(async()=>paint(await call('/api/state')))();
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Step through a recorded episode live in the browser.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="path to a run dir containing trajectory.npz")
    g.add_argument("--latest", action="store_true", help="use the newest run under --runs")
    ap.add_argument("--runs", default=str(REPO / "runs"), help="runs root for --latest")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--camera", default="closeup", help="camera name in the model XML")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--no-open", action="store_true", help="don't auto-open the browser")
    args = ap.parse_args()

    run_dir = _latest_run(Path(args.runs)) if args.latest else Path(args.run)
    episode = LiveEpisode(run_dir, args.width, args.height, args.camera)

    server = HTTPServer(("127.0.0.1", args.port), _Handler)
    server.episode = episode  # type: ignore[attr-defined]
    url = f"http://127.0.0.1:{args.port}/"
    print(f"live stepper for {run_dir.name} ({episode.total} steps) -> {url}\n"
          "  Step/Back/Play/Reset + slider; arrows step, space plays. Ctrl-C to stop.")
    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()

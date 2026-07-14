"""Self-contained HTML dashboard for agentic-selection runs.

Writes `<out>/dashboard.html` from the orchestrator checkpoint payload: pure inline SVG (no JS, no
CDN, no server), with a meta-refresh so a browser tab left open on the file live-updates while the
run goes. Regenerated after every evaluation (hooked into `save_checkpoint`) -- generation is
string formatting over a few hundred floats, so the overhead per ~8-min eval is nil.

The page is built to answer "how much longer should I run this?":
  - budget / plateau / per-chain status + wall-clock ETA of the remaining budget,
  - objective per iteration with the global best-so-far envelope,
  - trained-policy success metrics per iteration,
  - stage frontier progression per refinement chain,
  - per-stage gate performance (entered / hand-off / conversion / authored success) bars,
  - PPO learning curve of the latest evaluation (under-training check),
  - a heuristic guidance list (plateau imminent, undertrained verdict, recent unlocks, ...).

Standalone regeneration for a paused/finished run:
  python -m policy_bias_lab.cli.dashboard runs/<dir>
"""
from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Any

PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#be185d", "#4b5563"]


# ----------------------------------------------------------------------------------------------
# SVG primitives
# ----------------------------------------------------------------------------------------------

def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / max(n, 1)
    mag = 10 ** __import__("math").floor(__import__("math").log10(raw))
    step = min((s for s in (1 * mag, 2 * mag, 5 * mag, 10 * mag) if s >= raw), default=raw)
    first = __import__("math").ceil(lo / step) * step
    out, v = [], first
    while v <= hi + 1e-12:
        out.append(round(v, 10))
        v += step
    return out or [lo, hi]


def _fmt(v: float) -> str:
    return f"{v:g}" if abs(v) >= 1e-3 or v == 0 else f"{v:.1e}"


def line_chart(title: str, series: list[dict], *, w: int = 520, h: int = 240,
               x_label: str = "iteration", y_range: tuple | None = None,
               x_int: bool = True) -> str:
    """series: [{label, points:[(x,y)...], color, dash?:bool, marker?:bool, step?:bool}]"""
    pad_l, pad_r, pad_t, pad_b = 46, 10, 26, 34
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b
    pts_all = [(x, y) for s in series for (x, y) in s["points"] if y is not None]
    if not pts_all:
        return (f'<svg width="{w}" height="{h}" class="chart"><text x="{w/2}" y="{h/2}" '
                f'text-anchor="middle" class="muted">{html.escape(title)}: no data yet</text></svg>')
    xs, ys = [p[0] for p in pts_all], [p[1] for p in pts_all]
    x0, x1 = min(xs), max(xs)
    if x1 == x0:
        x0, x1 = x0 - 0.5, x1 + 0.5
    if y_range is not None:
        y0, y1 = y_range
    else:
        y0, y1 = min(ys), max(ys)
        span = (y1 - y0) or max(abs(y1), 1e-3)
        y0, y1 = y0 - 0.08 * span, y1 + 0.08 * span

    def X(x):
        return pad_l + (x - x0) / (x1 - x0) * iw

    def Y(y):
        return pad_t + (1 - (y - y0) / (y1 - y0)) * ih

    parts = [f'<svg width="{w}" height="{h}" class="chart" viewBox="0 0 {w} {h}">',
             f'<text x="{pad_l}" y="15" class="ctitle">{html.escape(title)}</text>']
    for tv in _ticks(y0, y1):
        if y0 <= tv <= y1:
            parts.append(f'<line x1="{pad_l}" y1="{Y(tv):.1f}" x2="{w - pad_r}" y2="{Y(tv):.1f}" '
                         f'class="grid"/>'
                         f'<text x="{pad_l - 5}" y="{Y(tv) + 3:.1f}" text-anchor="end" '
                         f'class="tick">{_fmt(tv)}</text>')
    xticks = sorted({int(round(t)) for t in _ticks(x0, x1)} if x_int else set(_ticks(x0, x1)))
    for tv in xticks:
        if x0 <= tv <= x1:
            parts.append(f'<text x="{X(tv):.1f}" y="{h - pad_b + 14}" text-anchor="middle" '
                         f'class="tick">{_fmt(tv)}</text>')
    parts.append(f'<text x="{pad_l + iw / 2:.0f}" y="{h - 6}" text-anchor="middle" class="tick">'
                 f'{html.escape(x_label)}</text>')
    for s in series:
        pts = [(x, y) for (x, y) in s["points"] if y is not None]
        if not pts:
            continue
        col = s.get("color", PALETTE[0])
        if s.get("step") and len(pts) > 1:  # step-after line (best-so-far envelopes, frontiers)
            path = f'M {X(pts[0][0]):.1f} {Y(pts[0][1]):.1f}'
            for (xa, _), (xb, yb) in zip(pts, pts[1:]):
                path += f' H {X(xb):.1f} V {Y(yb):.1f}'
            parts.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="1.8"'
                         + (' stroke-dasharray="5 3"' if s.get("dash") else "") + '/>')
        elif len(pts) > 1:
            pl = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
            parts.append(f'<polyline points="{pl}" fill="none" stroke="{col}" stroke-width="1.8"'
                         + (' stroke-dasharray="5 3"' if s.get("dash") else "") + '/>')
        if s.get("marker", True):
            for x, y in pts:
                parts.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="2.6" fill="{col}"/>')
    lx = pad_l + 6
    for s in series:
        lab = html.escape(str(s["label"]))
        parts.append(f'<rect x="{lx}" y="{pad_t + 2}" width="9" height="9" '
                     f'fill="{s.get("color", PALETTE[0])}"/>'
                     f'<text x="{lx + 12}" y="{pad_t + 10}" class="legend">{lab}</text>')
        lx += 16 + 6.2 * len(str(s["label"]))
    parts.append("</svg>")
    return "".join(parts)


def bar_chart(title: str, groups: list[str], series: list[dict], *, w: int = 520,
              h: int = 240) -> str:
    """Grouped bars in [0,1]. groups = stage names; series = [{label, values, color}]."""
    pad_l, pad_r, pad_t, pad_b = 40, 10, 26, 44
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b
    ng, ns = max(len(groups), 1), max(len(series), 1)
    gw = iw / ng
    bw = min(gw * 0.8 / ns, 26)
    parts = [f'<svg width="{w}" height="{h}" class="chart" viewBox="0 0 {w} {h}">',
             f'<text x="{pad_l}" y="15" class="ctitle">{html.escape(title)}</text>']
    for tv in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = pad_t + (1 - tv) * ih
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w - pad_r}" y2="{y:.1f}" class="grid"/>'
                     f'<text x="{pad_l - 5}" y="{y + 3:.1f}" text-anchor="end" class="tick">'
                     f'{tv:g}</text>')
    for gi, g in enumerate(groups):
        cx = pad_l + gw * (gi + 0.5)
        for si, s in enumerate(series):
            v = s["values"][gi] if gi < len(s["values"]) else None
            if v is None:
                continue
            v = max(0.0, min(1.0, float(v)))
            x = cx - (ns * bw) / 2 + si * bw
            parts.append(f'<rect x="{x:.1f}" y="{pad_t + (1 - v) * ih:.1f}" width="{bw - 1.5:.1f}" '
                         f'height="{max(v * ih, 0.5):.1f}" fill="{s.get("color", PALETTE[si % 8])}"/>')
        label = html.escape(g if len(g) <= 14 else g[:13] + "…")
        parts.append(f'<text x="{cx:.1f}" y="{h - pad_b + 13}" text-anchor="middle" class="tick" '
                     f'transform="rotate(-14 {cx:.1f} {h - pad_b + 13})">{gi}:{label}</text>')
    lx = pad_l + 6
    for si, s in enumerate(series):
        lab = html.escape(str(s["label"]))
        parts.append(f'<rect x="{lx}" y="{pad_t + 2}" width="9" height="9" '
                     f'fill="{s.get("color", PALETTE[si % 8])}"/>'
                     f'<text x="{lx + 12}" y="{pad_t + 10}" class="legend">{lab}</text>')
        lx += 16 + 6.2 * len(str(s["label"]))
    parts.append("</svg>")
    return "".join(parts)


# ----------------------------------------------------------------------------------------------
# Payload -> page
# ----------------------------------------------------------------------------------------------

def _rec_frontier(rec: dict, completion_frac: float = 0.25) -> int | None:
    sr = (rec.get("diagnostics") or {}).get("stage_report")
    if not isinstance(sr, dict) or not sr.get("stage_names"):
        return None
    names = sr.get("stage_names") or []
    entered = sr.get("entered_frac")
    handoff = sr.get("handoff_frac")
    threshold = max(0.0, min(1.0, float(completion_frac)))
    if entered and handoff and len(entered) == len(names):
        if entered[0] is None or float(entered[0]) < threshold:
            return 0
        d = 0
        while (d < len(names) - 1 and d < len(handoff) and handoff[d] is not None
               and float(handoff[d]) >= threshold):
            d += 1
        return len(names) if d == len(names) - 1 else d
    if sr.get("reaches_terminal"):
        return len(names)
    s = sr.get("stall_stage")
    return None if s is None else int(s)


def _chain_key(rec: dict) -> str:
    src = str(rec.get("source") or "")
    return src.split(":", 1)[1] if src.startswith("refine:") else str(rec.get("name"))


def _guidance(config: dict, state: dict, evaluated: list[dict]) -> list[str]:
    out = []
    budget, iters = state.get("budget", config.get("budget", 0)), state.get("iters", 0)
    patience = state.get("patience", config.get("patience", 3))
    sig = state.get("since_improve_global", 0)
    chains = state.get("chains") or []
    chosen = state.get("chosen_idx")
    active = [c for i, c in enumerate(chains) if (chosen is None or i in chosen)
              and getattr(c, "active", False)]
    if iters >= budget:
        out.append("BUDGET EXHAUSTED -- resume with a larger --budget to continue, or accept the "
                   "current best.")
    elif chosen is not None and not active:
        out.append("All refinement chains have plateaued -- the run will finish on its own; more "
                   "wall-clock only helps via --budget + a fresh patience window (--resume "
                   "--budget N reactivates chains).")
    if sig >= patience - 1 and iters < budget:
        out.append(f"Global plateau counter at {sig}/{patience}: one more non-improving evaluation "
                   f"stops the run.")
    last = evaluated[-1] if evaluated else None
    tr = ((last or {}).get("diagnostics") or {}).get("training_report") or {}
    if tr.get("verdict") == "undertrained":
        out.append("Latest candidate was still LEARNING when its PPO budget ran out "
                   f"(rising: {', '.join(tr.get('still_improving') or [])}) -- scores may be "
                   "understating the priors; consider resuming with a larger --ppo-train-seconds.")
    recent = evaluated[-3:]
    completion_frac = float(config.get("frontier_completion_frac", 0.25))
    fr = [f for f in (_rec_frontier(r, completion_frac) for r in recent) if f is not None]
    if len(fr) >= 2 and fr[-1] > fr[0]:
        out.append(f"Stage frontier advanced recently ({fr[0]} -> {fr[-1]}): structural progress "
                   f"is still happening -- worth continuing.")
    best = max((r.get("objective", -1e18) for r in evaluated), default=None)
    if best is not None and len(evaluated) >= patience + 1:
        prev = max(r.get("objective", -1e18) for r in evaluated[:-patience])
        if best > prev + config.get("eps", 1e-3):
            out.append(f"The best objective improved within the last {patience} evaluations "
                       f"-- the run is still paying off.")
    if not out:
        out.append("No strong stop/continue signal yet -- watch the best-so-far envelope and the "
                   "frontier chart.")
    return out


def build_dashboard(payload: dict) -> str:
    config, state = payload.get("config", {}), payload.get("state", {})
    evaluated: list[dict] = state.get("evaluated") or []
    chains = state.get("chains") or []
    chosen = state.get("chosen_idx")
    budget = state.get("budget", config.get("budget", 0))
    iters = state.get("iters", 0)
    patience = state.get("patience", config.get("patience", 3))
    completion_frac = float(config.get("frontier_completion_frac", 0.25))

    # -- header stats ---------------------------------------------------------------------------
    best = max(evaluated, key=lambda r: r.get("objective", -1e18)) if evaluated else None
    walls = [r["t_wall"] for r in evaluated if r.get("t_wall")]
    eta = ""
    if len(walls) >= 2 and iters < budget:
        dts = [b - a for a, b in zip(walls, walls[1:])][-5:]
        per = sum(dts) / len(dts)
        eta = (f"{per / 60:.1f} min/eval -> ≤ {(budget - iters) * per / 3600:.1f} h to budget "
               f"(plateau may stop it sooner)")
    stats = [
        ("iterations", f"{iters} / {budget}"),
        ("best objective", f"{best['objective']:+.4f}" if best else "--"),
        ("best candidate", f"{best.get('name')} ({best.get('source')})" if best else "--"),
        ("global plateau", f"{state.get('since_improve_global', 0)} / {patience}"),
        ("remaining budget ETA", eta or "--"),
        ("arbiter", f"{config.get('arbiter')} @ {config.get('ppo_train_seconds')}s PPO"),
    ]
    tr = ((evaluated[-1] if evaluated else {}).get("diagnostics") or {}).get("training_report") or {}
    if tr.get("verdict"):
        stats.append(("latest training verdict", tr["verdict"]))

    # -- chain table ----------------------------------------------------------------------------
    chain_rows = []
    for i, c in enumerate(chains):
        role = ("refining" if chosen is not None and i in chosen
                else "seed" if chosen is not None else "explore")
        sr = (getattr(c, "diagnostics", None) or {}).get("stage_report") or {}
        n_st = len(sr.get("stage_names") or [])
        fr = getattr(c, "frontier", -1)
        chain_rows.append(
            f"<tr><td>{html.escape(str(getattr(c, 'name', '?')))}</td><td>{role}</td>"
            f"<td>{getattr(c, 'best_obj', float('nan')):+.4f}</td>"
            f"<td>{fr if fr >= 0 else '--'}{f' / {n_st}' if n_st else ''}</td>"
            f"<td>{getattr(c, 'focus_side', '--')}</td>"
            f"<td>{getattr(c, 'since_improve', 0)} / {patience}</td>"
            f"<td>{'yes' if getattr(c, 'active', False) else 'no'}</td></tr>")

    # -- chart 1: objective per iteration + best-so-far ------------------------------------------
    by_chain: dict[str, list] = {}
    for r in evaluated:
        by_chain.setdefault(_chain_key(r), []).append((r["iter"], r.get("objective")))
    series = [{"label": k[:18], "points": v, "color": PALETTE[i % 8]}
              for i, (k, v) in enumerate(by_chain.items())]
    env_pts, hi = [], -1e18
    for r in evaluated:
        hi = max(hi, r.get("objective", -1e18))
        env_pts.append((r["iter"], hi))
    series.append({"label": "best-so-far", "points": env_pts, "color": "#111827",
                   "dash": True, "marker": False, "step": True})
    c_obj = line_chart("Objective per evaluation", series)

    # -- chart 2: trained-policy success metrics -------------------------------------------------
    met_keys = ["success_rate", "grasp_rate", "reach_rate", "lift_reached_rate"]
    mseries = []
    for i, mk in enumerate(met_keys):
        # "trained_success" is the pre-rename key for success_rate (old checkpoints).
        pts = [(r["iter"], (r.get("diagnostics") or {}).get(
                    mk, (r.get("diagnostics") or {}).get("trained_success") if mk == "success_rate" else None))
               for r in evaluated]
        mseries.append({"label": mk, "points": pts, "color": PALETTE[i]})
    c_met = line_chart("Trained-policy metrics per evaluation", mseries, y_range=(0.0, 1.0))

    # -- chart 3: stage frontier progression -----------------------------------------------------
    fr_by_chain: dict[str, list] = {}
    n_stages_max = 0
    for r in evaluated:
        f = _rec_frontier(r, completion_frac)
        if f is None:
            continue
        sr = (r.get("diagnostics") or {}).get("stage_report") or {}
        n_stages_max = max(n_stages_max, len(sr.get("stage_names") or []))
        fr_by_chain.setdefault(_chain_key(r), []).append((r["iter"], f))
    fseries = [{"label": k[:18], "points": v, "color": PALETTE[i % 8], "step": True}
               for i, (k, v) in enumerate(fr_by_chain.items())]
    if n_stages_max:
        lo = min(x for s in fseries for x, _ in s["points"])
        hi_x = max(x for s in fseries for x, _ in s["points"])
        fseries.append({"label": "terminal", "points": [(lo, n_stages_max), (hi_x, n_stages_max)],
                        "color": "#9ca3af", "dash": True, "marker": False})
    c_front = line_chart("Stage frontier (stall depth; top line = chain complete)", fseries,
                         y_range=(-0.3, max(n_stages_max, 1) + 0.3)) if fseries else ""

    # -- chart 4: per-stage gate performance of the current chains -------------------------------
    stage_charts = []
    show_idx = chosen if chosen is not None else list(range(len(chains)))
    for i in show_idx or []:
        if i >= len(chains):
            continue
        c = chains[i]
        sr = (getattr(c, "diagnostics", None) or {}).get("stage_report") or {}
        names = sr.get("stage_names") or []
        if not names:
            continue
        bars = [{"label": "entered", "values": sr.get("entered_frac") or [], "color": PALETTE[0]},
                {"label": "hand-off", "values": sr.get("handoff_frac") or [], "color": PALETTE[2]},
                {"label": "conversion", "values": sr.get("conversion") or [], "color": PALETTE[3]}]
        asf = sr.get("authored_success_frac")
        if asf and any(v is not None for v in asf):
            bars.append({"label": "authored success", "values": asf, "color": PALETTE[4]})
        stage_charts.append(bar_chart(
            f"Gate performance -- chain '{getattr(c, 'name', '?')}' (current base)", names, bars))

    # -- chart 5: latest PPO learning curve -------------------------------------------------------
    c_train = ""
    for r in reversed(evaluated):
        rows = (r.get("full") or {}).get("telemetry")
        if rows:
            tseries = []
            for i, mk in enumerate(("success", "grasp_rate", "reach_rate", "lift_reached_rate")):
                pts = [(j, float(row.get(mk, 0.0))) for j, row in enumerate(rows)]
                tseries.append({"label": mk, "points": pts, "color": PALETTE[i], "marker": False})
            c_train = line_chart(
                f"PPO learning curve -- latest eval (iter {r['iter']}: {r.get('name')})",
                tseries, x_label="training iteration", y_range=(0.0, 1.0))
            break

    # -- recent-eval table ------------------------------------------------------------------------
    rows_html = []
    for r in evaluated[-10:][::-1]:
        d = r.get("diagnostics") or {}
        fm = d.get("likely_failure_modes")
        fm = ", ".join(f.split(" ")[0] for f in fm) if isinstance(fm, list) else "--"
        f = _rec_frontier(r, completion_frac)
        rows_html.append(
            f"<tr><td>{r['iter']}</td><td>{html.escape(str(r.get('source')))}</td>"
            f"<td>{html.escape(str(r.get('name')))}</td><td>{r.get('objective', 0):+.4f}</td>"
            f"<td>{d.get('success_rate', d.get('trained_success', '--'))}</td>"
            f"<td>{f if f is not None else '--'}</td>"
            f"<td>{((d.get('training_report') or {}).get('verdict')) or '--'}</td>"
            f"<td>{html.escape(fm)}</td></tr>")

    guidance = "".join(f"<li>{html.escape(g)}</li>"
                       for g in _guidance(config, state, evaluated))
    stat_cards = "".join(f'<div class="card"><div class="k">{html.escape(k)}</div>'
                         f'<div class="v">{html.escape(str(v))}</div></div>' for k, v in stats)
    charts = "".join(x for x in [c_obj, c_met, c_front, *stage_charts, c_train] if x)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="20">
<title>agentic run -- {html.escape(str(config.get('rep')))}</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:18px;background:#f8fafc;color:#111827}}
 h1{{font-size:18px;margin:0 0 4px}} .muted{{fill:#9ca3af;color:#9ca3af;font-size:12px}}
 .cards{{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}}
 .card{{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:8px 14px}}
 .card .k{{font-size:11px;color:#6b7280;text-transform:uppercase}} .card .v{{font-size:15px}}
 .guide{{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:8px 16px;margin:10px 0}}
 .guide li{{margin:3px 0;font-size:13.5px}}
 .charts{{display:flex;flex-wrap:wrap;gap:14px}}
 .chart{{background:#fff;border:1px solid #e5e7eb;border-radius:8px}}
 .ctitle{{font-size:12.5px;font-weight:600;fill:#111827}}
 .tick{{font-size:10px;fill:#6b7280}} .legend{{font-size:10.5px;fill:#374151}}
 .grid{{stroke:#f1f5f9;stroke-width:1}}
 table{{border-collapse:collapse;background:#fff;margin-top:12px;font-size:12.5px}}
 th,td{{border:1px solid #e5e7eb;padding:4px 9px;text-align:left}} th{{background:#f1f5f9}}
</style></head><body>
<h1>Agentic prior selection -- {html.escape(str(config.get('rep')))} / {html.escape(str(config.get('arbiter')))}</h1>
<div class="muted">task: {html.escape(str(config.get('task')))} &middot; updated {now} (auto-refreshes every 20 s)</div>
<div class="cards">{stat_cards}</div>
<div class="guide"><b>Run guidance</b><ul>{guidance}</ul></div>
<div class="charts">{charts}</div>
<h1 style="margin-top:16px">Chains</h1>
<table><tr><th>chain</th><th>role</th><th>best obj</th><th>frontier</th><th>focus</th>
<th>plateau</th><th>active</th></tr>{''.join(chain_rows)}</table>
<h1 style="margin-top:16px">Recent evaluations</h1>
<table><tr><th>iter</th><th>source</th><th>name</th><th>objective</th><th>success</th>
<th>frontier</th><th>training</th><th>failure modes</th></tr>{''.join(rows_html)}</table>
</body></html>"""


def write_dashboard(payload: dict, path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(".html.tmp")
    tmp.write_text(build_dashboard(payload))
    tmp.replace(path)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Regenerate dashboard.html from a run checkpoint.")
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    import pickle
    with (args.run_dir / "checkpoint.pkl").open("rb") as f:
        payload = pickle.load(f)
    out = args.run_dir / "dashboard.html"
    write_dashboard(payload, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

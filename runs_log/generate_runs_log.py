#!/usr/bin/env python3
"""Build a provenance-heavy Markdown log from the repository's run artifacts.

This is intentionally descriptive: it never touches an environment or re-runs an
experiment.  It normalizes the heterogeneous historical artifact formats into one
report per top-level run directory while retaining the original reports verbatim.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
LEGACY_RUNS = ROOT / "famework_testing" / "runs"
OUT = ROOT / "runs_log"

EXCLUDED_NAMES = {
    "dynamic_reward_rewrite_combo_foreground_debug": "empty debug directory; no run artifacts",
    "prior_only_videos": "render-only collection, not an experimental run",
}

PREFERRED_METRICS = [
    "eval_graded_objective",
    "graded_objective",
    "eval_task_fitness",
    "eval_success_rate",
    "success_rate",
    "success",
    "mean_success",
    "best_success",
    "successes",
    "mean_score",
    "n",
    "eval_success",
    "eval_fitness",
    "auc",
    "eval_instant_success_rate",
    "eval_reach_rate",
    "reach_rate",
    "eval_grasp_rate",
    "grasp_rate",
    "eval_grasp_lift_rate",
    "grasp_lift_rate",
    "eval_lift_reached_rate",
    "lift_reached_rate",
    "eval_lift_max",
    "lift_max",
    "eval_base_return",
    "base_return",
    "eval_train_return",
    "train_return",
    "eval_shaped_return",
    "shaped_return",
    "eval_stage_return",
    "eval_action_abs_mean",
    "action_abs_mean",
    "eval_saturation_frac",
    "saturation_frac",
    "mean_total_env_steps",
    "iters",
    "best_iter",
]


def load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None


def scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        if value == 0:
            return "0"
        if abs(value) >= 100000:
            return f"{value:,.0f}"
        if abs(value) >= 100:
            return f"{value:,.3f}".rstrip("0").rstrip(".")
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, tuple)):
        return ", ".join(fmt(v) for v in value)
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= 240 else text[:237] + "…"


def md_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> list[str]:
    rendered = [[fmt(v) for v in row] for row in rows]
    if not rendered:
        return []
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out.extend("| " + " | ".join(row) + " |" for row in rendered)
    return out


def clean_title(name: str) -> str:
    words = name.replace("_", " ")
    words = re.sub(r"\b(20\d{6})(?:-(\d{4,6}))?\b", "", words)
    words = re.sub(r"\s+", " ", words).strip()
    replacements = {"ppo": "PPO", "dsl": "DSL", "dofmode": "DOF-mode", "mv": "Marginal-value", "kl": "KL"}
    return " ".join(replacements.get(w.lower(), w.capitalize()) for w in words.split())


def infer_date(run: Path) -> tuple[str, str]:
    match = re.search(r"(20\d{6})(?:-(\d{4,6}))?", run.name)
    summary = load_json(run / "summary.json")
    if isinstance(summary, dict) and isinstance(summary.get("manifest"), dict) and summary["manifest"].get("started_at"):
        return str(summary["manifest"]["started_at"]).replace("T", " "), "summary.json manifest started_at"
    if match:
        raw_date, raw_time = match.groups()
        dt = datetime.strptime(raw_date, "%Y%m%d")
        if raw_time:
            raw_time = raw_time.ljust(6, "0")
            tm = datetime.strptime(raw_time, "%H%M%S")
            return f"{dt:%Y-%m-%d} {tm:%H:%M:%S}", "directory timestamp"
        return f"{dt:%Y-%m-%d}", "directory timestamp"

    date_patterns = [
        (re.compile(r"Created:\s*(20\d{2}-\d{2}-\d{2}(?:\s+\d{2}:\d{2}:\d{2})?)"), None),
        (re.compile(r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(\d{2}\s+[A-Z][a-z]{2}\s+20\d{2})"), "%d %b %Y"),
    ]
    for path in sorted(run.glob("*.md")) + sorted(run.glob("*.log")):
        try:
            text = path.read_text(errors="replace")[:200_000]
        except OSError:
            continue
        for pattern, date_fmt in date_patterns:
            found = pattern.search(text)
            if found:
                if date_fmt:
                    return f"{datetime.strptime(found.group(1), date_fmt):%Y-%m-%d}", f"recorded in {path.name}"
                return found.group(1), f"recorded in {path.name}"

    files = [p for p in run.rglob("*") if p.is_file()]
    if files:
        dt = datetime.fromtimestamp(min(p.stat().st_mtime for p in files))
        return f"{dt:%Y-%m-%d}", "earliest surviving artifact mtime (approximate)"
    return "Unknown", "no timestamp evidence"


def family(name: str) -> str:
    low = name.lower()
    if "shadowhand" in low or "real_shadow" in low or "conditional_vs_flat" in low:
        return "early interface/controller comparison"
    if "policy_bias_lab_shadow_isolation" in low:
        return "early policy-bias isolation experiment"
    if "saturation" in low:
        return "action-output transform comparison"
    if low.startswith("mv_"):
        return "candidate-search breadth/depth study"
    if "critic" in low:
        return "critic-feature study"
    if "framework_cmp" in low:
        return "training-framework comparison"
    if "dsl_vs_freeform" in low or "dofmode" in low:
        return "prior-representation study"
    if "stalldir" in low:
        return "structural diagnostic/revision study"
    if any(token in low for token in ("agentic", "prior_v", "prior_refine", "prior_fable", "prior_only", "priortest", "wrist", "timing_gen", "pace_gen", "limits_gen", "spawnattitude", "ab_", "generation_only", "env_contact", "monotone_prioronly", "primitives_gen")):
        return "agentic prior generation/selection study"
    if any(token in low for token in ("longppo", "armtest", "full_long", "monotone_prior_6h", "arm_")):
        return "long-PPO training study"
    return "PPO bias/reward/prior experiment"


def discover_task(run: Path) -> str:
    for path in [run / "report.json", run / "config.json"]:
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        for key in ("task", "tasks"):
            if key in data:
                return fmt(data[key])
    for path in run.glob("*/config.json"):
        data = load_json(path)
        if isinstance(data, dict) and data.get("task"):
            return fmt(data["task"])
    summary = load_json(run / "summary.json")
    if isinstance(summary, dict):
        conditional = summary.get("conditional")
        if isinstance(conditional, dict) and isinstance(conditional.get("result"), dict) and conditional["result"].get("task"):
            return fmt(conditional["result"]["task"])
    for result_path in run.glob("*/result.json"):
        result_data = load_json(result_path)
        if not isinstance(result_data, dict):
            continue
        task_value = result_data.get("task")
        if isinstance(task_value, dict) and task_value.get("name"):
            return fmt(task_value["name"])
        if task_value:
            return fmt(task_value)
    if isinstance(summary, dict) and isinstance(summary.get("manifest"), dict) and summary["manifest"].get("tasks"):
        return fmt(summary["manifest"]["tasks"])
    return "not explicitly recorded at the run root"


def count_jsonl(path: Path) -> tuple[int, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    except OSError:
        pass
    return len(rows), rows


def direct_markdown_sources(run: Path) -> list[Path]:
    names = {"REPORT.md", "report.md", "STUDY_SUMMARY.md", "summary.md", "comparison.md"}
    return [p for p in sorted(run.iterdir()) if p.is_file() and p.name in names]


def structured_result_paths(run: Path) -> list[Path]:
    selected: list[Path] = []
    names = {"summary.json", "report.json", "final_report.json", "comparison.json", "frontier_report.json", "eval_corrected.json", "eval.json", "base_diag.json", "revised_diag.json", "marginal_value.json", "marginal_value_partial.json", "posthoc_stage_eval.json", "dsl_scored.json", "freeform_scored.json", "raw_seeds.json", "result.json", "compiled_summary.json"}
    for path in run.rglob("*.json"):
        if path.name in names and len(path.relative_to(run).parts) <= 3:
            selected.append(path)
    return sorted(selected)


def infer_status(run: Path) -> str:
    results = structured_result_paths(run)
    if results or direct_markdown_sources(run):
        if any(p.name in {"final_report.json", "summary.json", "comparison.json", "REPORT.md", "report.json", "STUDY_SUMMARY.md"} for p in results + direct_markdown_sources(run)):
            return "completed or completed-with-caveats"
    metric_rows = sum(count_jsonl(p)[0] for p in run.rglob("metrics*.jsonl"))
    if metric_rows:
        return "partial/interrupted; training metrics survive but no complete run-level report"
    logs = "\n".join(p.read_text(errors="replace")[-50_000:] for p in run.glob("*.log"))
    if re.search(r"Traceback|Error|Killed|FAILED|No space left", logs, re.I):
        return "failed/aborted attempt"
    return "artifact-only or generation-only; no terminal evaluation recorded"


def flatten_config(data: Any, prefix: str = "", depth: int = 0) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if depth > 3:
        return rows
    if isinstance(data, dict):
        skip = {"program", "prior_library", "init_params", "bias_spec", "reward_templates"}
        for key, value in data.items():
            if key in skip:
                continue
            path = f"{prefix}.{key}" if prefix else key
            if scalar(value):
                if not isinstance(value, str) or len(value) <= 300:
                    rows.append((path, value))
            elif isinstance(value, list) and len(value) <= 24 and all(scalar(v) for v in value):
                rows.append((path, value))
            elif isinstance(value, dict):
                rows.extend(flatten_config(value, path, depth + 1))
    return rows


def configuration_section(run: Path) -> list[str]:
    configs = [run / "config.json"] if (run / "config.json").exists() else sorted(run.glob("*/config.json"))
    if not configs:
        return ["No machine-readable configuration file survives at the run root. Structure below is reconstructed from reports, result files, subdirectory names, and logs."]
    out: list[str] = []
    for idx, path in enumerate(configs[:12]):
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        if len(configs) > 1:
            out.append(f"#### Configuration: `{path.relative_to(run)}`")
            out.append("")
        rows = flatten_config(data)[:90]
        out.extend(md_table(["Field", "Recorded value"], rows))
        out.append("")
    return out or ["Configuration files were present but could not be decoded."]


def metric_columns(records: dict[str, dict[str, Any]]) -> list[str]:
    present = {k for record in records.values() for k, v in record.items() if scalar(v)}
    cols = [k for k in PREFERRED_METRICS if k in present]
    return cols[:16]


def summary_result_section(path: Path, data: dict[str, Any]) -> list[str]:
    container = data.get("summary") if isinstance(data.get("summary"), dict) else data
    records = {str(k): v for k, v in container.items() if isinstance(v, dict)}
    out = []
    if records:
        cols = metric_columns(records)
        if not cols:
            present = {k for record in records.values() for k, value in record.items() if scalar(value)}
            cols = sorted(present)[:16]
        if cols:
            rows = [[name] + [record.get(col) for col in cols] for name, record in records.items()]
            out.extend(md_table(["Arm/interface"] + cols, rows))
    manifest = data.get("manifest")
    if isinstance(manifest, dict) and isinstance(manifest.get("runs"), list):
        run_rows = []
        for item in manifest["runs"]:
            if not isinstance(item, dict) or not isinstance(item.get("result"), dict):
                continue
            result = item["result"]
            run_rows.append([item.get("tag"), result.get("interface"), result.get("task"), result.get("seed"), result.get("success"), result.get("score"), result.get("total_return"), result.get("max_object_z"), result.get("errors")])
        if run_rows:
            if out:
                out.append("")
            out.extend(md_table(["Tag", "Interface", "Task", "Seed", "Success", "Score", "Return", "Max object z", "Errors"], run_rows))
    variant_rows = []
    for variant in ("conditional", "without_conditionals"):
        block = data.get(variant)
        if not isinstance(block, dict) or not isinstance(block.get("result"), dict):
            continue
        result = block["result"]
        reactive = block.get("metadata", {}).get("reactive_execution", {}) if isinstance(block.get("metadata"), dict) else {}
        variant_rows.append([variant, result.get("success"), result.get("score"), result.get("total_return"), result.get("max_object_z"), reactive.get("online"), reactive.get("steps_used"), len(reactive.get("loops", [])) if isinstance(reactive.get("loops"), list) else None])
    if variant_rows:
        if out:
            out.append("")
        out.extend(md_table(["Variant", "Success", "Score", "Return", "Max object z", "Online conditionals", "Steps", "Loops"], variant_rows))
    return out

def agentic_result_section(data: dict[str, Any]) -> list[str]:
    best = data.get("best")
    if not isinstance(best, dict):
        return []
    out: list[str] = []
    headline = [(k, best.get(k)) for k in ("name", "source", "objective") if k in best]
    accounting = best.get("accounting", {})
    if isinstance(accounting, dict):
        headline.extend((f"accounting.{k}", v) for k, v in accounting.items() if scalar(v))
    out.extend(md_table(["Best-candidate field", "Value"], headline))
    diagnostics = best.get("diagnostics", {})
    if isinstance(diagnostics, dict):
        metric_rows = [(k, diagnostics[k]) for k in PREFERRED_METRICS if k in diagnostics and scalar(diagnostics[k])]
        extras = [(k, v) for k, v in diagnostics.items() if scalar(v) and k not in {r[0] for r in metric_rows}]
        out.append("")
        out.extend(md_table(["Diagnostic", "Value"], (metric_rows + extras)[:35]))
        stage = diagnostics.get("stage_report")
        if isinstance(stage, dict) and isinstance(stage.get("stage_names"), list):
            names = stage["stage_names"]
            stage_rows = []
            for i, name in enumerate(names):
                def at(key: str) -> Any:
                    value = stage.get(key)
                    return value[i] if isinstance(value, list) and i < len(value) else None
                stage_rows.append([i, name, at("occupancy"), at("reached_frac"), at("entered_frac"), at("handoff_frac"), at("conversion"), at("authored_success_frac")])
            out.append("")
            out.append("Stage-flow measurements for the selected candidate:")
            out.append("")
            out.extend(md_table(["#", "Stage", "Occupancy", "Reached", "Entered", "Handoff", "Conversion", "Authored success"], stage_rows))
            if stage.get("stall_name") is not None:
                out.append("")
                out.append(f"The recorded structural stall was `{fmt(stage.get('stall_name'))}` (index {fmt(stage.get('stall_stage'))}); terminal stage reached: {fmt(stage.get('reaches_terminal'))}.")
    trajectory = data.get("trajectory")
    if isinstance(trajectory, list) and trajectory:
        keys = ["iter", "source", "name", "objective", "frontier", "wrist_driven", "n_driven"]
        out.append("")
        out.append("Candidate trajectory:")
        out.append("")
        out.extend(md_table(keys, ([row.get(k) for k in keys] for row in trajectory if isinstance(row, dict))))
    return out


def final_result_section(data: dict[str, Any]) -> list[str]:
    out: list[str] = []
    headline = [(k, v) for k, v in data.items() if scalar(v)]
    if headline:
        out.extend(md_table(["Run result", "Value"], headline))
    eval_data = data.get("eval")
    if isinstance(eval_data, dict):
        out.append("")
        rows = [(k, v) for k, v in eval_data.items() if scalar(v)]
        out.extend(md_table(["Evaluation metric", "Value"], rows))
    return out


def generic_json_section(data: Any) -> list[str]:
    records = data if isinstance(data, list) else data.get("candidates") if isinstance(data, dict) and isinstance(data.get("candidates"), list) else None
    if records:
        flat_records = []
        for record in records:
            if not isinstance(record, dict):
                continue
            flat_record = {}
            for key, value in record.items():
                if scalar(value):
                    flat_record[key] = value
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if scalar(subvalue):
                            flat_record[f"{key}.{subkey}"] = subvalue
            flat_records.append(flat_record)
        keys = []
        for record in flat_records:
            for key in record:
                if key not in keys:
                    keys.append(key)
        if flat_records and keys:
            return md_table(keys[:18], ([record.get(key) for key in keys[:18]] for record in flat_records[:50]))
        return [f"The file records {len(records)} structured entries; see the linked source artifact for nested fields."]
    if isinstance(data, dict):
        flat = flatten_config(data)
        return md_table(["Recorded field", "Value"], flat[:100])
    return []

def metrics_section(run: Path) -> tuple[list[str], dict[str, dict[str, Any]], int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total = 0
    for path in sorted(run.rglob("metrics*.jsonl")):
        n, rows = count_jsonl(path)
        total += n
        for row in rows:
            label = str(row.get("arm") or path.parent.relative_to(run) or "run")
            grouped[label].append(row)
    if not grouped:
        return [], {}, total
    summaries: dict[str, dict[str, Any]] = {}
    table_rows: list[list[Any]] = []
    for label, rows in grouped.items():
        last = rows[-1]
        success_key = next((k for k in ("eval_success_rate", "success_rate", "success") if any(finite_number(r.get(k)) for r in rows)), None)
        graded_key = next((k for k in ("eval_graded_objective", "graded_objective", "eval_objective", "best_objective") if any(finite_number(r.get(k)) for r in rows)), None)
        best_success = max((r.get(success_key) for r in rows if success_key and finite_number(r.get(success_key))), default=None)
        best_graded = max((r.get(graded_key) for r in rows if graded_key and finite_number(r.get(graded_key))), default=None)
        summaries[label] = {"last": last, "best_success": best_success, "best_graded": best_graded, "rows": len(rows)}
        table_rows.append([
            label,
            len(rows),
            last.get("iter"),
            last.get("env_steps"),
            last.get("elapsed_seconds"),
            last.get("success", last.get("eval_success_rate")),
            best_success,
            last.get("lift_max", last.get("eval_lift_max")),
            last.get("base_return", last.get("eval_base_return")),
            best_graded,
        ])
    out = md_table(["Arm/sub-run", "Rows", "Last iter", "Last env steps", "Elapsed s", "Last success", "Best success", "Last lift max", "Last base return", "Best graded/objective"], table_rows)
    out.append("")
    out.append("These are training-log endpoints and maxima, not substitutes for held-out evaluation. They are included especially for interrupted runs whose terminal report is missing.")
    return out, summaries, total


def log_evidence(run: Path) -> list[str]:
    evidence: list[str] = []
    pattern = re.compile(r"(\[done\]|_DONE|ALL_ARMS_DONE|Traceback|Error|FAILED|Killed|No space left|stop_reason|graded_objective|success=)", re.I)
    for path in sorted(run.glob("*.log")):
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        hits = [line.strip() for line in lines if pattern.search(line) and "Failed to import warp" not in line and "Failed to import mujoco_warp" not in line]
        for line in hits[-12:]:
            evidence.append(f"`{path.name}`: {line}")
    return evidence[:40]


def artifact_inventory(run: Path) -> list[str]:
    repo_path = run.relative_to(ROOT).as_posix()
    files = [p for p in run.rglob("*") if p.is_file()]
    suffixes = Counter((p.suffix.lower() or "[no suffix]") for p in files)
    rows = [[suffix, count] for suffix, count in sorted(suffixes.items(), key=lambda item: (-item[1], item[0]))]
    out = md_table(["Artifact type", "Count"], rows)
    important_names = {"config.json", "summary.json", "report.json", "final_report.json", "comparison.json", "metrics.jsonl", "REPORT.md", "report.md", "STUDY_SUMMARY.md", "summary.md", "frontier_report.json", "eval_corrected.json"}
    important = [p for p in files if p.name in important_names]
    if important:
        out.append("")
        out.append("Primary evidence files:")
        out.append("")
        out.extend(f"- [`{p.relative_to(run)}`](../{repo_path}/{p.relative_to(run).as_posix()})" for p in sorted(important))
    return out


def interpretation(run: Path, json_results: list[tuple[Path, Any]], metric_summaries: dict[str, dict[str, Any]], status: str) -> list[str]:
    observations: list[str] = []
    if run.name == "conditional_vs_flat_video":
        summary = load_json(run / "summary.json")
        if isinstance(summary, dict):
            conditional = summary.get("conditional", {}).get("result", {})
            flat = summary.get("without_conditionals", {}).get("result", {})
            observations.append(f"Both the online-conditional and flattened variants failed with score {fmt(conditional.get('score'))} and max object z {fmt(conditional.get('max_object_z'))}. The three online loops therefore changed execution structure but produced no measured success or lift advantage in this episode; its return ({fmt(conditional.get('total_return'))}) was also below the flattened return ({fmt(flat.get('total_return'))}).")
    comparisons: list[tuple[str, dict[str, Any]]] = []
    agentic_best: list[dict[str, Any]] = []
    finals: list[tuple[str, dict[str, Any]]] = []
    for path, data in json_results:
        if isinstance(data, list):
            scored = [(float(record["score"]["objective_score"]), str(record.get("name"))) for record in data if isinstance(record, dict) and isinstance(record.get("score"), dict) and finite_number(record["score"].get("objective_score"))]
            if scored:
                value, name = max(scored)
                observations.append(f"In `{path.name}`, `{name}` had the highest recorded open-loop objective score ({fmt(value)}). All candidates should still be read against the contact, lift, displacement, and saturation measurements in the table; this prefilter score is not trained-policy success.")
            continue
        if isinstance(data, dict) and path.name == "marginal_value.json":
            breadth = data.get("breadth_best_of_first_k_mean")
            depth = data.get("depth_best_so_far_mean")
            if isinstance(breadth, list) and breadth and isinstance(depth, list) and depth:
                observations.append(f"The mean best result after the recorded breadth sweep ended at {fmt(breadth[-1])}, while the revision-depth trajectory ended at {fmt(depth[-1])}. In this small study, later revisions produced the larger final gain, but the standard deviations and only {fmt(data.get("reps"))} repetitions make the magnitude provisional.")
        if isinstance(data, dict) and path.name == "comparison.json" and isinstance(data.get("deltas_prior_minus_baseline"), dict):
            deltas = data["deltas_prior_minus_baseline"]
            key = next((key for key in ("eval_success_rate", "eval_graded_objective", "eval_task_fitness") if finite_number(deltas.get(key))), None)
            if key:
                direction = "improved" if deltas[key] > 0 else "reduced"
                observations.append(f"The prior-minus-baseline comparison {direction} `{key}` by {fmt(deltas[key])}. Because this is a direct within-run delta, it is the clearest headline comparison for this experiment.")
        if isinstance(data, dict) and path.name == "raw_seeds.json" and isinstance(data.get("candidates"), list):
            observations.append(f"The generation-only run produced {len(data["candidates"])} candidate program(s). No rollout or training evaluation survives, so this establishes generation output, not behavioral quality.")
        if not isinstance(data, dict):
            continue
        if path.name in {"summary.json", "eval_corrected.json"}:
            comparison_data = data.get("summary") if isinstance(data.get("summary"), dict) else data
            comparisons.extend((str(k), v) for k, v in comparison_data.items() if isinstance(v, dict))
        if path.name == "report.json" and isinstance(data.get("best"), dict):
            agentic_best.append(data["best"])
        if path.name == "final_report.json" and isinstance(data.get("eval"), dict):
            finals.append((path.parent.name, data["eval"]))

    if comparisons:
        success_keys = ("eval_success_rate", "success_rate", "success")
        graded_keys = ("eval_graded_objective", "graded_objective", "eval_task_fitness", "eval_fitness", "mean_score")
        for label, keys, noun in (("held-out success", success_keys, "success"), ("graded/task objective", graded_keys, "objective")):
            scored = []
            for name, record in comparisons:
                key = next((k for k in keys if finite_number(record.get(k))), None)
                if key:
                    scored.append((float(record[key]), name, key))
            if scored:
                best_value, best_name, key = max(scored)
                observations.append(f"Among the recorded arms, `{best_name}` had the highest {label} ({key} = {fmt(best_value)}). This is an ordering within this run, not a general ranking across experiments with different metric definitions or budgets.")
        if len(comparisons) > 1:
            observations.append("The arm comparison is most informative when arms shared the same environment, seed, and budget. Where the configuration records only one seed, apparent gaps remain vulnerable to optimization variance.")

    for best in agentic_best:
        diagnostics = best.get("diagnostics", {})
        objective = best.get("objective")
        success = diagnostics.get("success_rate") if isinstance(diagnostics, dict) else None
        observations.append(f"The selection loop chose `{fmt(best.get('name'))}` with objective {fmt(objective)}. Selection therefore established a best candidate relative to the tested pool; it did not by itself establish task mastery.")
        if finite_number(success) and success == 0:
            lift = diagnostics.get("lift_reached_rate", diagnostics.get("lift_max"))
            observations.append(f"Recorded success was zero while the associated lift/progress measurement was {fmt(lift)}. The candidate generated partial behavior under the run's diagnostics but did not satisfy the full success predicate in that evaluation.")
        stage = diagnostics.get("stage_report") if isinstance(diagnostics, dict) else None
        if isinstance(stage, dict) and stage.get("stall_name"):
            observations.append(f"The stage-flow diagnostic localized the dominant structural bottleneck to `{fmt(stage['stall_name'])}`. This is a measured control-flow finding from the authored program, useful for revision without treating the stage label as a framework-authored diagnosis.")

    if finals:
        graded = [(float(v[k]), name, k) for name, v in finals for k in ("eval_graded_objective", "eval_task_fitness") if finite_number(v.get(k))]
        success = [(float(v[k]), name, k) for name, v in finals for k in ("eval_success_rate", "success_rate") if finite_number(v.get(k))]
        if graded:
            val, name, key = max(graded)
            observations.append(f"Across terminal sub-run reports, `{name}` recorded the strongest {key} ({fmt(val)}).")
        if success:
            val, name, key = max(success)
            observations.append(f"The best terminal success measurement was `{name}` at {fmt(val)} ({key}).")

    if metric_summaries and not (comparisons or finals or agentic_best):
        best = [(s["best_success"], name) for name, s in metric_summaries.items() if finite_number(s.get("best_success"))]
        if best:
            value, name = max(best)
            observations.append(f"Only training metrics survive; their largest recorded success value was {fmt(value)} for `{name}`. Because there is no terminal held-out report, this is progress evidence rather than a defensible final result.")

    if "failed/aborted" in status or "partial/interrupted" in status:
        observations.append("The missing terminal artifact is itself important: this attempt cannot support a comparative scientific conclusion. Its value is operational—showing what was launched, how far it progressed, and where the evidence trail ends.")
    if not observations:
        observations.append("The surviving artifacts document the attempted structure but do not contain enough terminal quantitative evidence for a strong result claim. Treat this entry as provenance and negative/unfinished evidence.")
    return observations


def demote_markdown(text: str) -> str:
    return re.sub(r"^(#{1,5})\s", lambda m: "#" + m.group(1) + " ", text, flags=re.M)


def write_report(run: Path) -> tuple[Path, str, str]:
    date, date_basis = infer_date(run)
    status = infer_status(run)
    kind = family(run.name)
    article = "an" if kind[0].lower() in "aeiou" else "a"
    task = discover_task(run)
    title = clean_title(run.name)
    repo_path = run.relative_to(ROOT).as_posix()
    report_stem = run.name if run.parent == RUNS else f"famework_testing__{run.name}"
    json_results: list[tuple[Path, Any]] = []
    for path in structured_result_paths(run):
        data = load_json(path)
        if isinstance(data, (dict, list)):
            json_results.append((path, data))
    metric_lines, metric_summaries, metric_count = metrics_section(run)

    lines = [
        f"# {title}",
        "",
        f"- **Run directory:** [`{repo_path}`](../{repo_path})",
        f"- **Date:** {date} ({date_basis})",
        f"- **Status:** {status}",
        f"- **Experiment class:** {kind}",
        f"- **Recorded task:** {task}",
        f"- **Training metric rows recovered:** {metric_count:,}",
        "",
        "## Abstract",
        "",
        f"This entry reconstructs `{run.name}`, {article} {kind}. The account is based only on artifacts that survive in the repository: machine-readable configuration, candidate programs, selection diagnostics, training metrics, evaluations, reports, and logs. The run targeted {task}. Its purpose and comparisons are inferred from the recorded arm names and configuration where no authored abstract survives; those inferences are identified by this provenance-first framing rather than presented as new experimental facts.",
        "",
        "## Experimental structure",
        "",
    ]
    lines.extend(configuration_section(run))

    report_data = next((data for path, data in json_results if path == run / "report.json"), None)
    if isinstance(report_data, dict):
        meta = [(k, report_data.get(k)) for k in ("representation", "dof_mode", "arbiter", "budget", "iters_used", "n_seeds", "wall_hours") if k in report_data]
        if meta:
            lines.append("### Selection-loop structure")
            lines.append("")
            lines.extend(md_table(["Field", "Value"], meta))
            lines.append("")

    child_dirs = sorted(p.name for p in run.iterdir() if p.is_dir() and not p.name.startswith("."))
    if child_dirs:
        lines.append("### Arms and sub-runs represented by directories")
        lines.append("")
        lines.append(", ".join(f"`{name}`" for name in child_dirs))
        lines.append("")

    lines.append("## Results")
    lines.append("")
    if not json_results and not metric_lines:
        lines.append("No decodable run-level result JSON or non-empty training metric stream survives.")
        lines.append("")
    for path, data in json_results:
        rel = path.relative_to(run)
        section: list[str]
        if path.name in {"summary.json", "eval_corrected.json"}:
            section = summary_result_section(path, data)
        elif path.name == "report.json":
            section = agentic_result_section(data)
        elif path.name == "final_report.json":
            section = final_result_section(data)
        else:
            section = generic_json_section(data)
        if section:
            lines.append(f"### Recorded results: `{rel}`")
            lines.append("")
            lines.extend(section)
            lines.append("")

    if metric_lines:
        lines.append("### Training-metric reconstruction")
        lines.append("")
        lines.extend(metric_lines)
        lines.append("")

    evidence = log_evidence(run)
    if evidence:
        lines.append("### Terminal and failure evidence from logs")
        lines.append("")
        lines.extend(f"- {item}" for item in evidence)
        lines.append("")

    lines.append("## What the results mean and major discoveries")
    lines.append("")
    lines.extend(f"- {item}" for item in interpretation(run, json_results, metric_summaries, status))
    lines.append("")
    lines.append("## Limitations and reading guidance")
    lines.append("")
    lines.extend([
        "- Metrics are reported under the names used by the historical environment and runner. Similar-looking metrics from different generations of the code are not assumed to be numerically interchangeable.",
        "- A maximum observed in training is not treated as held-out performance. Terminal evaluation values are preferred whenever present.",
        "- Missing summaries, empty metric streams, failed retries, and incomplete arms are preserved as negative evidence rather than silently dropped.",
        "- Candidate names and stage names are LLM-authored run artifacts. Interpretations in this historical log do not alter framework prompts, defaults, diagnostics, or task logic.",
    ])
    lines.append("")
    lines.append("## Artifact provenance")
    lines.append("")
    lines.extend(artifact_inventory(run))
    lines.append("")

    sources = direct_markdown_sources(run)
    if sources:
        lines.append("## Appendix: original authored run report")
        lines.append("")
        lines.append("The following contemporaneous report is retained (with headings demoted) so that details and caveats are not lost in normalization.")
        lines.append("")
        for source in sources:
            lines.append(f"### Source: [`{source.name}`](../{repo_path}/{source.name})")
            lines.append("")
            lines.append(demote_markdown(source.read_text(errors="replace").strip()))
            lines.append("")

    out_path = OUT / f"{report_stem}.md"
    out_path.write_text("\n".join(lines).rstrip() + "\n")
    return out_path, date, status


def is_smoke(run: Path) -> bool:
    return "smoke" in run.name.lower()


def write_standalone_training_log(path: Path) -> tuple[Path, str, str]:
    text = path.read_text(errors="replace")
    date = f"{datetime.fromtimestamp(path.stat().st_mtime):%Y-%m-%d %H:%M:%S}"
    config_lines = [line.strip() for line in text.splitlines() if line.startswith(("env:", "arm=", "LLM supervision:"))]
    rows = []
    pattern = re.compile(r"it\s+(\d+)\s+\| return\s+(-?[0-9.]+)\s+\| success\s+([0-9.]+)\s+\| lift\s+([0-9.]+)\s+\| kl\s+([0-9.]+)")
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            rows.append([int(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4)), float(match.group(5))])
    final_match = re.search(r"final success=([0-9.]+) return=(-?[0-9.]+)", text)
    status = "completed" if final_match else "failed/aborted attempt; no training rows or terminal result survive"
    title = clean_title(path.stem)
    source = path.relative_to(ROOT).as_posix()
    report = OUT / f"exports_bootstrapping__{path.stem}.md"
    lines = [f"# {title}", "", f"- **Run artifact:** [`{source}`](../{source})", f"- **Date:** {date} (artifact mtime)", f"- **Status:** {status}", "- **Experiment class:** early bootstrapping PPO training run", "- **Recorded task:** lift-like object-raising task (inferred from the logged `lift` metric)", f"- **Training metric rows recovered:** {len(rows)}", "", "## Abstract", "", f"This standalone log records one arm of the early bootstrapping PPO study. It preserved {len(rows)} progress checkpoints and {"a terminal result" if final_match else "no terminal result"}. The configuration and quantitative trajectory below are transcribed from the log rather than reconstructed from code defaults.", "", "## Experimental structure", ""]
    if config_lines:
        lines.extend(f"- `{line}`" for line in config_lines)
    else:
        lines.append("The log contains only an initialization warning; no environment, arm, or optimizer configuration was recorded.")
    lines.extend(["", "## Results", ""] )
    if rows:
        lines.extend(md_table(["Iteration", "Return", "Success", "Lift", "KL"], [rows[0], rows[-1]] if len(rows) > 1 else rows))
        best = max(rows, key=lambda row: (row[2], row[1]))
        lines.extend(["", f"Best logged checkpoint by success then return: iteration {best[0]}, success {fmt(best[2])}, return {fmt(best[1])}, lift {fmt(best[3])}." ])
    else:
        lines.append("No training iterations were logged.")
    if final_match:
        lines.extend(["", f"Terminal line: success {final_match.group(1)}, return {final_match.group(2)}."])
    lines.extend(["", "## What the results mean and major discoveries", ""] )
    if rows and final_match:
        lines.append(f"- The arm progressed from success {fmt(rows[0][2])} at iteration {rows[0][0]} to terminal success {final_match.group(1)}. This establishes convergence in this recorded single run, not a variance-controlled advantage over another arm.")
        if "LLM supervision:" in text:
            usage = re.search(r"LLM usage: (.+)", text)
            lines.append(f"- This was the LLM-supervised arm. The log records {usage.group(1) if usage else "LLM supervision but no terminal usage summary"}; any comparison to the baseline must account for that extra supervision budget.")
    else:
        lines.append("- The surviving file contains no experimental outcome. It documents an attempted launch only and cannot support a performance conclusion.")
    lines.extend(["", "## Limitations and reading guidance", "", "- This is a standalone console log without the saved parameter directory it references.", "- Only sampled checkpoints are available; intermediate behavior and held-out evaluation are absent.", "- Comparisons with later Shadow Hand runs are invalid because the log records a different 27-observation, 5-action environment.", "", "## Artifact provenance", "", f"- Source console log: [`{source}`](../{source})", "- All tabulated values were parsed from `it ...` and `final ...` lines in that file.", ""] )
    report.write_text("\n".join(lines))
    return report, date, status


def main() -> None:
    OUT.mkdir(exist_ok=True)
    included: list[tuple[Path, str, str, Path]] = []
    excluded: list[tuple[str, str]] = []
    for run in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        if is_smoke(run):
            excluded.append((f"runs/{run.name}", "explicit smoke test by directory name"))
            continue
        if run.name in EXCLUDED_NAMES:
            excluded.append((f"runs/{run.name}", EXCLUDED_NAMES[run.name]))
            continue
        report, date, status = write_report(run)
        included.append((run, date, status, report))

    for run in sorted(p for p in LEGACY_RUNS.iterdir() if p.is_dir()):
        legacy_name = f"famework_testing/runs/{run.name}"
        if is_smoke(run):
            excluded.append((legacy_name, "explicit smoke test by directory name"))
            continue
        if run.name.startswith("check_"):
            excluded.append((legacy_name, "functional check directory, treated as a smoke/integration test"))
            continue
        report, date, status = write_report(run)
        included.append((run, date, status, report))

    standalone_root = ROOT / "exports" / "export_experiment" / "source" / "bootstrapping"
    for log_path in sorted(standalone_root.glob("runs_rl_*.log")):
        report, date, status = write_standalone_training_log(log_path)
        included.append((log_path, date, status, report))

    included.sort(key=lambda item: (item[1] == "Unknown", item[1], item[0].name))
    lines = [
        "# Experimental runs log",
        "",
        "This directory is a normalized historical log of every substantive top-level experimental run found under `runs/` and the older `famework_testing/runs/` archive. Each run has a separate Markdown report with date evidence, an abstract, experimental structure, quantitative results, interpretation, limitations, and links to the original artifacts.",
        "",
        f"- **Substantive run entries:** {len(included)}",
        f"- **Excluded smoke/support entries:** {len(excluded)}",
        "- **Inventory date:** 2026-07-14",
        "",
        "## Inclusion policy",
        "",
        "A directory is included when it represents a substantive training, comparison, generation, selection, diagnostic, or failed/interrupted experimental attempt. A completed success is not required: preserving negative and partial evidence prevents survivorship bias. Directories explicitly named `smoke` are excluded, as are render-only collections and empty debug placeholders. Short runs not named as smoke are retained when their artifacts show they were used as a scientific diagnostic or comparison.",
        "",
        "Dates come first from run-directory timestamps, then from contemporaneous reports/logs, and finally from the earliest surviving artifact modification time. The per-run report states which source was used.",
        "",
        "## Chronological index",
        "",
        "| Date | Run | Status |",
        "|---|---|---|",
    ]
    for run, date, status, report in included:
        lines.append(f"| {date} | [{run.relative_to(ROOT).as_posix()}]({report.name}) | {status} |")
    lines.extend(["", "## Excluded smoke tests and non-run support directories", "", "| Directory | Reason |", "|---|---|"])
    lines.extend(f"| `{name}` | {reason} |" for name, reason in sorted(excluded))
    lines.extend([
        "",
        "## Cross-run cautions",
        "",
        "The history spans several runner generations. Episode horizons, environment counts, reward definitions, success predicates, action transforms, selection objectives, and report schemas changed. Cross-run comparisons should therefore be made only when the reports establish compatible conditions. Within-run controlled arm comparisons are generally stronger evidence than comparing headline numbers from unrelated runs.",
        "",
        "This log is documentary only. It introduces no task knowledge into `policy_bias_lab`, changes no environment/reward code, and does not transplant discoveries into framework prompts or defaults.",
    ])
    (OUT / "README.md").write_text("\n".join(lines).rstrip() + "\n")


if __name__ == "__main__":
    main()

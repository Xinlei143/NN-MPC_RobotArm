"""Summarise counterfactual candidate calibration and ranking diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SELECTED, BASELINE, ALTERNATIVE = 1, 2, 4
BRANCH_KINDS = {"strict_exact_action": 0, "activation_projected": 1}


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64); result[order] = np.arange(len(values), dtype=np.float64)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse predicted-vs-realized MPC candidate branches.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--branch_kind", choices=tuple(BRANCH_KINDS), default="strict_exact_action")
    args = parser.parse_args()
    root = ROOT / args.input_dir if not Path(args.input_dir).is_absolute() else Path(args.input_dir)
    rows: list[dict[str, object]] = []
    total_kind_count = 0
    infeasible_kind_count = 0
    for path in sorted(root.glob("branches_*.npz")):
        with np.load(path) as data:
            for index in range(len(data["branch_id"])):
                kind_id = int(data["branch_kind_id"][index]) if "branch_kind_id" in data.files else 0
                if kind_id != BRANCH_KINDS[args.branch_kind]:
                    continue
                total_kind_count += 1
                if data["activation_infeasible"][index]:
                    infeasible_kind_count += 1
                    continue
                pred, real = float(data["predicted_cost"][index]), float(data["realized_cost"][index])
                if not (np.isfinite(pred) and np.isfinite(real)):
                    continue
                pstate, rstate = data["predicted_state_sequence"][index], data["realized_state_sequence"][index]
                anchor_error = float(data["anchor_prediction_error"][index]) if "anchor_prediction_error" in data.files else float(np.linalg.norm(pstate[0] - rstate[0]))
                rows.append({"parent": int(data["parent_main_episode_id"][index]), "activation": int(data["activation_step"][index]), "group": int(data["candidate_group_id"][index]) if "candidate_group_id" in data.files else int(data["parent_main_episode_id"][index]) * 10_000 + int(data["activation_step"][index]), "role": int(data["role_mask"][index]), "pred": pred, "real": real, "anchor_error": anchor_error, "q_rmse": np.sqrt(np.mean((pstate[:, : pstate.shape[1] // 2] - rstate[:, : rstate.shape[1] // 2]) ** 2, axis=1)).tolist()})
    predicted = np.asarray([row["pred"] for row in rows], dtype=float); realized = np.asarray([row["real"] for row in rows], dtype=float)
    spearman = float(np.corrcoef(rank(predicted), rank(realized))[0, 1]) if len(rows) > 1 else float("nan")
    groups: dict[tuple[int, int], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((int(row["group"]),), []).append(row)
    comparisons = {"selected_vs_baseline": [], "selected_vs_alternative": [], "selection_regret": []}
    for group in groups.values():
        selected = next((row for row in group if int(row["role"]) & SELECTED), None)
        if selected is None:
            continue
        for name, bit in (("selected_vs_baseline", BASELINE), ("selected_vs_alternative", ALTERNATIVE)):
            other = next((row for row in group if int(row["role"]) & bit), None)
            if other is not None:
                comparisons[name].append(float((float(selected["pred"]) <= float(other["pred"])) == (float(selected["real"]) <= float(other["real"]))))
        best_real = min(float(row["real"]) for row in group)
        comparisons["selection_regret"].append(float(selected["real"]) / best_real if best_real > 0.0 else float("nan"))
    q = np.asarray([row["q_rmse"] for row in rows], dtype=float)
    q_by_step = [] if not len(q) else np.mean(q, axis=0).tolist()
    k_metrics = {f"q_rmse_k{k}": (float(q_by_step[k]) if len(q_by_step) > k else float("nan")) for k in (1, 5, 10, 20, 25)}
    finite_regret = np.asarray(comparisons["selection_regret"], dtype=float)
    summary = {"branch_kind": args.branch_kind, "branch_count": total_kind_count, "activation_infeasible_count": infeasible_kind_count, "activation_infeasible_rate": float(infeasible_kind_count / total_kind_count) if total_kind_count else float("nan"), "candidate_count": len(rows), "predicted_realized_cost_spearman": spearman, "selected_vs_baseline_ranking_accuracy": float(np.mean(comparisons["selected_vs_baseline"])) if comparisons["selected_vs_baseline"] else float("nan"), "selected_vs_alternative_ranking_accuracy": float(np.mean(comparisons["selected_vs_alternative"])) if comparisons["selected_vs_alternative"] else float("nan"), "selection_regret_mean": float(np.mean(finite_regret[np.isfinite(finite_regret)])) if np.any(np.isfinite(finite_regret)) else float("nan"), "anchor_prediction_error_mean": float(np.mean([row["anchor_error"] for row in rows])) if rows else float("nan"), "q_rmse_by_step": q_by_step, **k_metrics}
    output = ROOT / args.output_path if not Path(args.output_path).is_absolute() else Path(args.output_path); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

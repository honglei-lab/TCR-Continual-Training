"""Require complete native evaluation grids before computing Table 6 metrics."""
import argparse
import statistics
from pathlib import Path

from common import ARMS, ROOT, read_json, write_json


def suite_score(path, suite, n):
    tasks = read_json(path)["per_task"]
    if len(tasks) != 10 or {t["task_id"] for t in tasks} != set(range(10)):
        raise ValueError(f"Incomplete task set: {path}")
    means = []
    for task in tasks:
        values = task["metrics"]["successes"]
        if task["task_group"] != suite or len(values) != n or any(type(v) is not bool for v in values):
            raise ValueError(f"Invalid episode records: {path}")
        means.append(100 * sum(values) / n)
    return statistics.mean(means)


def curve_metrics(grid, scores, threshold):
    if len(grid) != len(scores) or len(grid) < 2 or grid[0] != 0 or any(b <= a for a, b in zip(grid, grid[1:])):
        raise ValueError("Invalid full evaluation grid")
    area = sum((b - a) * (x + y) / 2 for a, b, x, y in zip(grid, grid[1:], scores, scores[1:])) / grid[-1]
    crossing = None if threshold is None else next((b for b, s in zip(grid, scores) if s >= threshold), f">{grid[-1]}")
    return {"S0": scores[0], "SB": scores[-1], "nAUC": area, "T_tau": crossing}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/table6.json")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = read_json(args.config)
    results = {}
    for arm in ARMS:
        repeats = []
        for seed in cfg["training_seeds"]:
            suites = {s: [] for s in cfg["suites"]}
            for step in cfg["eval_grid"]:
                folder = args.results / arm / f"seed{seed}" / f"step{step:06d}"
                for suite in suites:
                    suites[suite].append(suite_score(folder / suite / "eval_info.json", suite, cfg["episodes_per_task"]))
            means = [statistics.mean(values) for values in zip(*suites.values())]
            repeats.append({"seed": seed, "curve": means, "per_suite": suites,
                            **curve_metrics(cfg["eval_grid"], means, cfg["threshold_percent"])})
        results[arm] = {"repeats": repeats, "summary": {
            metric: {"mean": statistics.mean(r[metric] for r in repeats),
                     "std": statistics.stdev(r[metric] for r in repeats)} for metric in ("S0", "SB", "nAUC")},
            "T_tau_per_repeat": [r["T_tau"] for r in repeats]}
    write_json(args.output, {"recipe": cfg, "results": results})
    print("Complete paired grids summarized. Censored T_tau is not averaged as a number.")


if __name__ == "__main__":
    main()

"""Evaluate one checkpoint on all four suites, sequentially on one GPU."""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from common import ROOT, read_json, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/table6.json")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    cfg = read_json(args.config)
    policy = args.checkpoint.resolve()
    if args.execute:
        if args.output.exists():
            raise FileExistsError(args.output)
        for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
            if not (policy / name).is_file():
                raise FileNotFoundError(policy / name)
        if not (policy / "model.safetensors").exists() and not (policy / "adapter_model.safetensors").exists():
            raise FileNotFoundError("Missing model weights")
        if not os.environ.get("LIBERO_CONFIG_PATH"):
            raise ValueError("Set LIBERO_CONFIG_PATH using setup_libero.py first")
    for suite in cfg["suites"]:
        command = [sys.executable, str(ROOT / "scripts/eval_entry.py"),
                   f"--output_dir={args.output.resolve() / suite}", "--env.type=libero",
                   f"--env.task={suite}", "--env.hard_reset=true", "--env.init_states=true",
                   "--eval.batch_size=1", f"--eval.n_episodes={cfg['episodes_per_task']}",
                   f"--seed={cfg['eval_seed']}", f"--policy.path={policy}",
                   "--policy.device=cuda", "--policy.compile_model=false",
                   "--policy.gradient_checkpointing=false", "--policy.n_action_steps=10"]
        print(shlex.join(command), flush=True)
        if args.execute:
            env = os.environ.copy()
            env.setdefault("MUJOCO_GL", "egl")
            env["PALIGEMMA_TOKENIZER_PATH"] = str(policy / "tokenizer")
            subprocess.run(command, env=env, check=True)
    if args.execute:
        write_json(args.output / "evaluation_receipt.json", {"policy": str(policy), "recipe": cfg,
                   "note": "Fresh native clean-LIBERO envs, fixed batch=1 and initial-state order; not a replay of main-table episodes"})


if __name__ == "__main__":
    main()

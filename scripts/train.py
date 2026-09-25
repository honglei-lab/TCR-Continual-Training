"""Print or launch a paired five-arm run; each node invokes the same command with its rank."""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from common import ARMS, ROOT, episodes, per_rank_batch, read_json, sha256, verify_checkpoint, write_json

TARGET = (r"model\.paligemma_with_expert\.(?:paligemma\.model\.(?:language_model\.layers\.\d+"
          r"|vision_tower\.vision_model\.encoder\.layers\.\d+)|gemma_expert\.model\.layers\.\d+)"
          r"\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj|out_proj)|mlp\.(?:gate_proj|up_proj|down_proj|fc1|fc2))")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--config", type=Path, default=ROOT / "configs/table6.json")
    p.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints/initializations-v1")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--gpus-per-node", type=int, default=4)
    p.add_argument("--node-rank", type=int, default=0)
    p.add_argument("--master-addr", default="127.0.0.1")
    p.add_argument("--master-port", type=int, default=29500)
    p.add_argument("--smoke", action="store_true", help="Two steps, isolated output; NOT formal results")
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def command(args, cfg):
    batch = per_rank_batch(cfg["global_batch"], args.nodes, args.gpus_per_node)
    if not 0 <= args.node_rank < args.nodes:
        raise ValueError("Invalid node rank")
    if args.nodes > 1 and args.master_addr in ("127.0.0.1", "localhost"):
        raise ValueError("Multi-node jobs require a reachable master address")
    if args.seed not in cfg["training_seeds"]:
        raise ValueError("Use one of the paired training seeds in config")
    if cfg["arms"] != list(ARMS):
        raise ValueError("This protocol contains exactly the five agreed arms")
    steps = 2 if args.smoke else cfg["steps"]
    save = 1 if args.smoke else cfg["save_freq"]
    if cfg["eval_grid"][0] != 0 or cfg["eval_grid"][-1] != cfg["steps"]:
        raise ValueError("Evaluation grid must include zero and final budget")
    if any(x % cfg["save_freq"] for x in cfg["eval_grid"]):
        raise ValueError("Evaluation grid points must be saved checkpoints")
    init = (args.checkpoint_root / args.arm).resolve()
    output = (args.output_root / ("smoke" if args.smoke else "formal") / args.arm / f"seed{args.seed}" / "train").resolve()
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nnodes={args.nodes}",
           f"--nproc_per_node={args.gpus_per_node}", f"--node_rank={args.node_rank}",
           f"--master_addr={args.master_addr}", f"--master_port={args.master_port}",
           str(ROOT / "scripts/train_entry.py")]
    options = {
        "policy.path": init, "policy.push_to_hub": "false", "policy.n_action_steps": 10,
        "policy.gradient_checkpointing": "true", "policy.compile_model": "false",
        "policy.dtype": "bfloat16", "policy.device": "cuda",
        "policy.optimizer_lr": cfg["learning_rate"], "policy.scheduler_warmup_steps": cfg["warmup_steps"],
        "policy.scheduler_decay_steps": cfg["steps"], "policy.scheduler_decay_lr": cfg["decay_lr"],
        "accelerator.mixed_precision": "bf16", "accelerator.gradient_accumulation.steps": 1,
        "parallelism.dp_replicate": args.nodes * args.gpus_per_node,
        "dataset.repo_id": cfg["dataset_repo"], "dataset.root": args.dataset_root.resolve(),
        "dataset.episodes": str(episodes()).replace(" ", ""), "dataset.video_backend": "torchcodec",
        "dataset.return_uint8": "true", "dataset.eval_split": 0.0,
        "peft.method_type": "LORA", "peft.r": cfg["lora_rank"], "peft.lora_alpha": cfg["lora_alpha"],
        "peft.target_modules": TARGET,
        "peft.full_training_modules": '["action_in_proj","action_out_proj","time_mlp_in","time_mlp_out"]',
        "steps": steps, "batch_size": batch, "num_workers": 0, "persistent_workers": "false",
        "env_eval_freq": 0, "eval_steps": 0, "log_freq": 1 if args.smoke else 20,
        "save_freq": save, "seed": args.seed, "wandb.enable": "false",
        "output_dir": output, "job_name": f"{args.arm}_seed{args.seed}",
    }
    cmd += [f"--{key}={value}" for key, value in options.items()]
    return cmd, init, output, batch


def main():
    args = arguments()
    cfg = read_json(args.config)
    cmd, init, output, batch = command(args, cfg)
    print(f"global_batch={cfg['global_batch']} = {args.nodes} * {args.gpus_per_node} * {batch}; accumulation=1")
    print(shlex.join(cmd), flush=True)
    if not args.execute:
        return
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite/resume automatically: {output}")
    expected = read_json(ROOT / "configs/initializations.json")[args.arm]["model_sha256"]
    verify_checkpoint(init, expected)
    info = read_json(args.dataset_root / "meta/info.json")
    if info.get("total_episodes") != 1693 or info.get("total_tasks") != 40:
        raise ValueError("Expected the audited 1693-episode / 40-task LIBERO dataset")
    if not (args.dataset_root / "DATASET_RECEIPT.json").exists():
        raise FileNotFoundError("Run prepare_data.py to pin and fingerprint the dataset first")
    receipt = output.parent / f"launch_node{args.node_rank}.json"
    if receipt.exists():
        raise FileExistsError(f"Launch receipt already exists: {receipt}")
    write_json(receipt, {"recipe": cfg, "command": cmd, "model_sha256": expected,
                         "dataset_receipt": read_json(args.dataset_root / "DATASET_RECEIPT.json"),
                         "config_sha256": sha256(args.config), "smoke": args.smoke})
    env = os.environ.copy()
    env["LORA_INIT_SEED"] = str(cfg["lora_init_seed"])
    env["PALIGEMMA_TOKENIZER_PATH"] = str(init / "tokenizer")
    env.setdefault("OMP_NUM_THREADS", "4")
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()

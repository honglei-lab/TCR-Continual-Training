"""Validate native LeRobot CLI config only: no model, GPU, data loading, or training."""
from runtime import bootstrap

bootstrap()
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # Registers the policy type.


@parser.wrap()
def check(cfg: TrainPipelineConfig):
    cfg.validate()
    print("PASS: native config parsed and validated", flush=True)
    print({"steps": cfg.steps, "per_rank_batch": cfg.batch_size,
           "lr": cfg.optimizer.lr, "dp_replicate": cfg.parallelism.dp_replicate,
           "accumulation": cfg.accelerator.gradient_accumulation.steps,
           "episodes": len(cfg.dataset.episodes), "peft_rank": cfg.peft.r})


if __name__ == "__main__":
    check()

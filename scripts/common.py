"""Small dependency-free helpers for the experiment launchers."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("base", "soup", "regmeanpp", "featcal", "tcr")
PREFIX = "pi05-continued-training/initializations-v1"
REPO = "Velixx/TCR"
SIDECARS = (
    "config.json", "policy_preprocessor.json", "policy_postprocessor.json",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episodes():
    parts = [read_json(ROOT / "data/splits" / (s + ".json")) for s in
             ("libero_spatial", "libero_object", "libero_goal", "libero_10")]
    values = [x for part in parts for x in part]
    if len(values) != 1693 or sorted(values) != list(range(1693)):
        raise ValueError("Expected the disjoint 1693-episode four-suite union")
    return sorted(values)


def per_rank_batch(global_batch, nodes, gpus):
    world = nodes * gpus
    if min(global_batch, nodes, gpus) < 1 or global_batch % world:
        raise ValueError("global_batch must be divisible by nodes * gpus_per_node")
    return global_batch // world


def verify_checkpoint(path, expected=None):
    path = Path(path)
    for name in ("model.safetensors", *SIDECARS, "tokenizer/tokenizer.json",
                 "tokenizer/tokenizer_config.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    if (path / "adapter_config.json").exists():
        raise ValueError("Initializations must be dense checkpoints, not old adapters")
    cfg = read_json(path / "config.json")
    if cfg.get("type") != "pi05" or cfg.get("use_peft"):
        raise ValueError("Expected a dense pi05 config")
    if expected and sha256(path / "model.safetensors") != expected:
        raise ValueError(f"Weight hash mismatch: {path}")
    return cfg

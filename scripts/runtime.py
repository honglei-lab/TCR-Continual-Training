"""Explicit runtime behavior shared by training and clean-LIBERO evaluation."""
import os
import sys
from pathlib import Path

from common import ROOT


def bootstrap():
    sys.path.insert(0, str(ROOT / "vendor/lerobot/src"))
    if os.environ.get("LIBERO_PRO_REPO"):
        raise RuntimeError("This package evaluates clean LIBERO, not LIBERO-PRO")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def local_tokenizer():
    raw = os.environ.get("PALIGEMMA_TOKENIZER_PATH")
    if not raw:
        return
    path = Path(raw).resolve()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    from lerobot.processor.tokenizer_processor import TokenizerProcessorStep
    original = TokenizerProcessorStep.__post_init__

    def post_init(self):
        if self.tokenizer_name == "google/paligemma-3b-pt-224":
            self.tokenizer_name = str(path)
        original(self)

    TokenizerProcessorStep.__post_init__ = post_init


def fixed_lora_seed():
    from lerobot.policies.pretrained import PreTrainedPolicy
    from lerobot.utils.random_utils import seeded_context
    original = PreTrainedPolicy.wrap_with_peft

    def wrap(self, peft_config=None, peft_cli_overrides=None):
        with seeded_context(int(os.environ.get("LORA_INIT_SEED", "3407"))):
            return original(self, peft_config=peft_config, peft_cli_overrides=peft_cli_overrides)

    PreTrainedPolicy.wrap_with_peft = wrap


def preserve_checkpoint_normalizers():
    # The upstream fresh-training path replaces checkpoint stats with dataset stats.
    # Continuing a merged dense checkpoint must preserve its action/state contract.
    import lerobot.policies as policies
    original = policies.make_pre_post_processors

    def make(*args, **kwargs):
        if kwargs.get("pretrained_path"):
            kwargs.pop("dataset_stats", None)
            for field in ("preprocessor_overrides", "postprocessor_overrides"):
                for step in kwargs.get(field, {}).values():
                    step.pop("stats", None)
        return original(*args, **kwargs)

    policies.make_pre_post_processors = make


def validate_real_cameras():
    from lerobot.policies import factory
    original = factory.validate_visual_features_consistency

    def validate(cfg, features):
        saved = cfg.input_features
        cfg.input_features = {k: v for k, v in saved.items() if ".empty_camera_" not in k}
        try:
            return original(cfg, features)
        finally:
            cfg.input_features = saved

    factory.validate_visual_features_consistency = validate

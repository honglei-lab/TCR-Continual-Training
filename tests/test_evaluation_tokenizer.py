import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import resolve_tokenizer


def assets(path):
    path.mkdir(parents=True)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (path / name).write_text("{}")
    return path


def test_dense_checkpoint_tokenizer(tmp_path, monkeypatch):
    monkeypatch.delenv("PALIGEMMA_TOKENIZER_PATH", raising=False)
    target = assets(tmp_path / "dense/tokenizer")
    assert resolve_tokenizer(target.parent) == target


def test_adapter_uses_initialization_tokenizer(tmp_path, monkeypatch):
    monkeypatch.delenv("PALIGEMMA_TOKENIZER_PATH", raising=False)
    target = assets(tmp_path / "base/tokenizer")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(target.parent)}))
    assert resolve_tokenizer(adapter) == target


def test_explicit_override_and_missing_assets(tmp_path, monkeypatch):
    target = assets(tmp_path / "base/tokenizer")
    monkeypatch.setenv("PALIGEMMA_TOKENIZER_PATH", str(tmp_path / "missing"))
    assert resolve_tokenizer(tmp_path, target) == target
    with pytest.raises(FileNotFoundError, match="--tokenizer"):
        resolve_tokenizer(tmp_path)


def test_runtime_override_relocates_saved_processor(tmp_path, monkeypatch):
    import types
    from runtime import local_tokenizer
    target = assets(tmp_path / "tokenizer")
    monkeypatch.setenv("PALIGEMMA_TOKENIZER_PATH", str(target))
    class TokenizerStep:
        tokenizer_name = "/old/server/tokenizer"
        tokenizer = None
        def __post_init__(self):
            self.loaded_from = self.tokenizer_name
    fake = types.ModuleType("lerobot.processor.tokenizer_processor")
    fake.TokenizerProcessorStep = TokenizerStep
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    local_tokenizer()
    step = TokenizerStep()
    step.__post_init__()
    assert step.loaded_from == str(target)

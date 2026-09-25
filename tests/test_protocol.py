import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ARMS, ROOT, episodes, per_rank_batch, read_json
from modelscope_upload import matches
from summarize import curve_metrics
from train import TARGET, command


def test_joint_data():
    assert episodes() == list(range(1693))


def test_batch_budget():
    assert per_rank_batch(128, 8, 8) == 2
    assert per_rank_batch(128, 1, 4) == 32
    with pytest.raises(ValueError):
        per_rank_batch(128, 3, 8)


def test_metrics():
    m = curve_metrics([0, 10, 20], [20, 40, 60], 50)
    assert m == {"S0": 20, "SB": 60, "nAUC": 40, "T_tau": 20}
    assert curve_metrics([0, 20], [20, 40], 50)["T_tau"] == ">20"
    assert curve_metrics([0, 20], [20, 40], None)["T_tau"] is None
    with pytest.raises(ValueError):
        curve_metrics([0, 20], [20], 50)


def test_launch():
    cfg = read_json(ROOT / "configs/table6.json")
    args = SimpleNamespace(nodes=8, gpus_per_node=8, node_rank=0, master_addr="10.1.2.3",
                           master_port=29500, seed=1000, smoke=False,
                           checkpoint_root=Path('/models'), output_root=Path('/runs'),
                           dataset_root=Path('/data'), arm='tcr')
    cmd, init, out, batch = command(args, cfg)
    assert batch == 2 and init == Path('/models/tcr')
    assert '--accelerator.gradient_accumulation.steps=1' in cmd
    assert '--parallelism.dp_replicate=64' in cmd
    assert '--resume=true' not in cmd
    assert '--steps=5000' in cmd
    assert cfg['arms'] == list(ARMS)
    args.smoke = True
    cmd, _, out, _ = command(args, cfg)
    assert '--steps=2' in cmd and '/smoke/' in str(out)


def test_target():
    import re
    assert re.fullmatch(TARGET, 'model.paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj')
    assert not re.fullmatch(TARGET, 'model.action_out_proj')


def test_upload_guard():
    assert matches({'Size': 5, 'Sha256': 'abc'}, 5, 'abc')
    assert not matches({'Size': 5}, 5, 'abc')
    assert not matches({'Size': 4, 'Sha256': 'abc'}, 5, 'abc')

"""Create an isolated LIBERO config from a user-supplied official asset tree."""
import argparse
from pathlib import Path

from common import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark-root", type=Path, help="Default: pinned hf-libero package's bddl_files/ and init_files/")
    p.add_argument("--assets", type=Path, default=ROOT / "local/libero-assets", help="Extracted official LIBERO assets directory")
    p.add_argument("--download-assets", action="store_true")
    p.add_argument("--config-dir", type=Path, default=ROOT / "local/libero-config")
    args = p.parse_args()
    import yaml
    if args.benchmark_root is None:
        from importlib.metadata import distribution
        args.benchmark_root = Path(distribution("hf-libero").locate_file("libero/libero"))
    benchmark = args.benchmark_root.resolve()
    if args.download_assets:
        from huggingface_hub import snapshot_download
        snapshot_download("lerobot/libero-assets", repo_type="dataset",
                          revision="0b3ea86be5fe169d0fd036ae63d1070ec09e90f6", local_dir=args.assets)
    for directory in (benchmark / "bddl_files", benchmark / "init_files", args.assets):
        if not directory.is_dir() or not any(directory.iterdir()):
            raise FileNotFoundError(directory)
    out = args.config_dir / "config.yaml"
    if out.exists():
        raise FileExistsError(out)
    args.config_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump({"benchmark_root": str(benchmark), "bddl_files": str(benchmark / "bddl_files"),
               "init_states": str(benchmark / "init_files"), "assets": str(args.assets.resolve()),
               "datasets": str(ROOT / "local/libero-unused-hdf5")}))
    print(f"export LIBERO_CONFIG_PATH={args.config_dir.resolve()}")


if __name__ == "__main__":
    main()

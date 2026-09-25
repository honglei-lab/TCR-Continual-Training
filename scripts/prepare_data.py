"""Download a pinned LeRobot LIBERO dataset, or fingerprint an existing local copy."""
import argparse
from pathlib import Path

from common import episodes, read_json, sha256, write_json

REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--download", action="store_true")
    args = p.parse_args()
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download("lerobot/libero", repo_type="dataset", revision=REVISION, local_dir=args.root)
    info = read_json(args.root / "meta/info.json")
    if info.get("total_episodes") != 1693 or info.get("total_tasks") != 40:
        raise ValueError("Wrong dataset: expected 1693 episodes, 40 tasks")
    files = sorted(f for sub in ("meta", "data", "videos") for f in (args.root / sub).rglob("*") if f.is_file())
    for sub in ("meta", "data", "videos"):
        if not any(f.is_relative_to(args.root / sub) for f in files):
            raise FileNotFoundError(f"Missing dataset {sub}")
    manifest = [{"path": f.relative_to(args.root).as_posix(), "size": f.stat().st_size, "sha256": sha256(f)} for f in files]
    write_json(args.root / "DATASET_RECEIPT.json", {"repo": "lerobot/libero",
               "revision": REVISION if args.download else "local_copy_content_identified_by_hashes",
               "episodes": episodes(), "files": manifest})
    print("Fingerprint recorded; copy this exact dataset to all nodes.")


if __name__ == "__main__":
    main()

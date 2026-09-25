"""Download only the five selected initializations; verify each against its manifest."""
import argparse
import shutil
from pathlib import Path

from common import ARMS, PREFIX, REPO, ROOT, read_json, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "checkpoints/initializations-v1")
    p.add_argument("--arm", choices=ARMS, action="append")
    args = p.parse_args()
    from modelscope.hub.file_download import model_file_download
    mf = model_file_download(REPO, file_path=PREFIX + "/manifest.json")
    manifest = read_json(mf)
    if manifest["repo_id"] != REPO or manifest["prefix"] != PREFIX:
        raise ValueError("Invalid remote manifest")
    arms = args.arm or list(ARMS)
    expected = read_json(ROOT / "configs/initializations.json")
    for arm in arms:
        if manifest["arms"][arm]["model_sha256"] != expected[arm]["model_sha256"]:
            raise ValueError("Remote initialization identity changed")
    for item in manifest["files"]:
        rel = Path(item["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe manifest path")
        if rel.parts[0] not in arms:
            continue
        dest = args.output / rel
        if dest.exists():
            if dest.stat().st_size != item["size"] or sha256(dest) != item["sha256"]:
                raise ValueError(f"Existing file differs; refusing overwrite: {dest}")
            continue
        cached = Path(model_file_download(REPO, file_path=PREFIX + "/" + item["path"]))
        if cached.stat().st_size != item["size"] or sha256(cached) != item["sha256"]:
            raise ValueError(f"Download checksum mismatch: {rel}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, dest)
        print("Verified", rel, flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mf, args.output / "manifest.json")


if __name__ == "__main__":
    main()

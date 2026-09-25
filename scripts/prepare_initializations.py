"""Create a checksummed, portable upload tree without copying 47 GB of weights."""
import argparse
import os
import shutil
from pathlib import Path

from common import ARMS, PREFIX, REPO, ROOT, SIDECARS, read_json, sha256, verify_checkpoint, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, required=True, help="Private JSON mapping arm to source directory")
    p.add_argument("--output", type=Path, default=ROOT / "local/upload")
    args = p.parse_args()
    sources = read_json(args.sources)
    identities = read_json(ROOT / "configs/initializations.json")
    if set(sources) != set(ARMS):
        raise ValueError("Exactly five initialization paths required")
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace staging: {args.output}")
    manifest = {"repo_id": REPO, "prefix": PREFIX, "files": [], "arms": identities}
    for arm in ARMS:
        source = Path(sources[arm]).resolve()
        print(f"Verifying {arm} ({source})", flush=True)
        verify_checkpoint(source, identities[arm]["model_sha256"])
        target = args.output / arm
        target.mkdir(parents=True)
        files = [source / "model.safetensors", *(source / n for n in SIDECARS)]
        files += sorted(f for f in (source / "tokenizer").rglob("*") if f.is_file())
        for original in files:
            rel = original.relative_to(source)
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if rel.as_posix() == "config.json":
                config = read_json(original)
                config["pretrained_path"] = None  # Never publish obsolete server-local paths.
                write_json(dest, config)
            elif original.name == "model.safetensors":
                # Symlink is only local staging; uploader reads file bytes, not links.
                dest.symlink_to(original)
            else:
                shutil.copy2(original, dest)
            digest = identities[arm]["model_sha256"] if original.name == "model.safetensors" else sha256(dest)
            manifest["files"].append({"path": dest.relative_to(args.output).as_posix(),
                                      "size": dest.stat().st_size, "sha256": digest})
        print(f"Verified and staged {arm}", flush=True)
    write_json(args.output / "manifest.json", manifest)
    print("Prepared", sum(f["size"] for f in manifest["files"]), "bytes; no upload performed")


if __name__ == "__main__":
    main()

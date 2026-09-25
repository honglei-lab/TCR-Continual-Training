"""Upload only a manifest-listed subtree, refusing conflicting remote files."""
import argparse
import os
from pathlib import Path

from common import PREFIX, REPO, ROOT, read_json, sha256, write_json


def matches(remote, size, digest):
    remote_hash = remote.get("Sha256") or remote.get("SHA256")
    return int(remote.get("Size", -1)) == size and remote_hash == digest


def remote_files():
    # modelscope 1.39's legacy get_model_files shim drops SHA256. Use the
    # underlying public SDK's typed listing, which retains the server digest.
    from modelscope_hub.api import HubApi
    return {f.path: {"Path": f.path, "Size": f.size, "Sha256": f.sha256}
            for f in HubApi().list_repo_files(REPO, "model", recursive=True)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--staging", type=Path, default=ROOT / "local/upload")
    p.add_argument("--execute", action="store_true", help="Without this flag: print plan only")
    p.add_argument("--workers", type=int, default=5, help="Concurrent SDK transfer workers")
    args = p.parse_args()
    manifest = read_json(args.staging / "manifest.json")
    if manifest["repo_id"] != REPO or manifest["prefix"] != PREFIX:
        raise ValueError("Unexpected publication target")
    entries = list(manifest["files"])
    entries.append({"path": "manifest.json", "size": (args.staging / "manifest.json").stat().st_size,
                    "sha256": sha256(args.staging / "manifest.json")})
    for item in entries:
        rel = Path(item["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Invalid manifest path")
    print(f"Target: {REPO}/{PREFIX}; {len(entries)} files; {sum(e['size'] for e in entries):,} bytes", flush=True)
    if not args.execute:
        return
    from modelscope.hub.api import HubApi, ModelScopeConfig
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if not token and not ModelScopeConfig.get_cookies():
        raise RuntimeError("Log in first with scripts/modelscope_login.py; never put tokens in Git")
    api = HubApi()
    if token:
        api.login(token)
    remote = remote_files()
    for item in entries:
        key = PREFIX + "/" + item["path"]
        if key in remote and not matches(remote[key], item["size"], item["sha256"]):
            raise RuntimeError(f"Conflicting/unverifiable remote file; refusing overwrite: {key}")
    for item in entries:
        file = args.staging / item["path"]
        if file.stat().st_size != item["size"] or sha256(file) != item["sha256"]:
            raise ValueError(f"Staging file changed: {file}")
    print(f"Uploading with {args.workers} workers and SDK transfer cache", flush=True)
    # Only explicit manifest paths are eligible. No remote pruning/deletion.
    # Publish manifest last so consumers do not see an apparently complete release early.
    api.upload_folder(repo_id=REPO, folder_path=str(args.staging), repo_type="model",
                      path_in_repo=PREFIX, allow_patterns=[x["path"] for x in manifest["files"]],
                      max_workers=args.workers, use_cache=True, disable_tqdm=True,
                      sync_remote_repo=False)
    uploaded = remote_files()
    for item in manifest["files"]:
        key = PREFIX + "/" + item["path"]
        if not matches(uploaded.get(key, {}), item["size"], item["sha256"]):
            raise RuntimeError(f"Remote verification failed before publishing manifest: {key}")
    if PREFIX + "/manifest.json" not in remote:
        api.upload_file(repo_id=REPO, path_or_fileobj=str(args.staging / "manifest.json"),
                        path_in_repo=PREFIX + "/manifest.json", repo_type="model", disable_tqdm=True)
    final = remote_files()
    for item in entries:
        key = PREFIX + "/" + item["path"]
        if not matches(final.get(key, {}), item["size"], item["sha256"]):
            raise RuntimeError(f"Remote size/hash verification failed: {key}")
    write_json(ROOT / "local/upload_verified.json", {"repo": REPO, "prefix": PREFIX, "verified_files": entries})
    print("All remote file sizes and SHA256 verified.")


if __name__ == "__main__":
    main()

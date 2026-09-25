"""Read-only dependency and source-origin checks. Does not allocate a model."""
import importlib
import importlib.metadata
import sys

from common import ROOT, episodes, read_json
from runtime import bootstrap


def main():
    bootstrap()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use Python 3.12 for the documented environment")
    failures = []
    for line in (ROOT / "requirements-constraints.txt").read_text().splitlines():
        if "==" not in line or line.startswith("#"):
            continue
        name, expected = line.split("==")
        try:
            version = importlib.metadata.version(name)
            print(name, version)
            if version.split("+")[0] != expected:
                failures.append(f"{name}: expected {expected}, got {version}")
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"Missing {name}")
    for module in ("torchcodec", "lerobot.policies.pi05.modeling_pi05", "lerobot.scripts.lerobot_train"):
        try:
            imported = importlib.import_module(module)
            print(module, imported.__file__)
        except Exception as exc:
            failures.append(f"{module}: {type(exc).__name__}: {exc}")
    import lerobot
    if not __import__('pathlib').Path(lerobot.__file__).is_relative_to(ROOT / "vendor"):
        failures.append("Imported wrong LeRobot source")
    print("joint episodes:", len(episodes()))
    print("arms:", read_json(ROOT / "configs/table6.json")["arms"])
    if failures:
        raise SystemExit("\n".join(failures))
    print("PASS: imports and versions only; GPU/multi-node execution is not validated by this check.")


if __name__ == "__main__":
    main()

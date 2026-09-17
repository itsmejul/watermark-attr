"""Single-GPU Trainer checkpoint selection, ignoring interrupted saves."""
import json
from pathlib import Path


def latest_complete_checkpoint(output_dir):
    candidates = [p for p in Path(output_dir).glob("checkpoint-*")
                  if p.is_dir() and p.name.removeprefix("checkpoint-").isdigit()]
    for path in sorted(candidates, key=lambda p: int(p.name.split("-")[-1]), reverse=True):
        required = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_config.json")
        if not all((path / name).is_file() and (path / name).stat().st_size for name in required):
            continue
        if not any((path / name).is_file() and (path / name).stat().st_size
                   for name in ("adapter_model.safetensors", "adapter_model.bin")):
            continue
        try:
            state = json.loads((path / "trainer_state.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if state.get("global_step") == int(path.name.split("-")[-1]):
            return str(path)
    return None

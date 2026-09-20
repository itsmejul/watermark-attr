"""Generate and directly verify a small Qwen Waterfall kappa ablation.

This experiment deliberately stops after watermark insertion: it does not
fine-tune a model.  All conditions use the same first 100 abstracts, IDs,
keys, and per-text random seeds so that kappa is the only changed parameter.

Examples:
    python -m src.experiments.ablations.qwen_kappa_ablation 6
    python -m src.experiments.ablations.qwen_kappa_ablation 6 --temperature 1.0
    python -m src.experiments.ablations.qwen_kappa_ablation 10 --dry-run
    python -m src.experiments.ablations.qwen_kappa_ablation --aggregate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from src.util.filereader import REPO_ROOT, load_abstracts, write_path_file_atomic


KAPPAS = (6, 10, 14)
N_SAMPLES = 100
GENERATION_SEED = 20260920
OUTPUT_ROOT = Path("results/ablations/qwen_kappa_source")
TEMPERATURE_OUTPUT_ROOT = Path("results/ablations/qwen_temperature_source")
BASE_CONFIG_PATH = Path("data/watermark_config_qwen3_5_9b.json")
KEYS_PATH = Path("data/keys.json")


def _load_json(path: Path):
    with (REPO_ROOT / path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _temperature_slug(temperature: float) -> str:
    return f"{temperature:g}".replace("-", "minus_").replace(".", "p")


def _output_path(kappa: int, temperature: float | None = None) -> Path:
    if temperature is not None:
        return TEMPERATURE_OUTPUT_ROOT / f"temperature_{_temperature_slug(temperature)}"
    return OUTPUT_ROOT / f"kappa_{kappa}"


def _condition_config(kappa: int, temperature: float | None = None) -> dict:
    config = dict(_load_json(BASE_CONFIG_PATH))
    config.update(
        n_samples=N_SAMPLES,
        batch_size=N_SAMPLES,
        kappa=float(kappa),
        generation_seed=GENERATION_SEED,
        checkpoint_every=10,
    )
    if temperature is not None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        config["temperature_watermark"] = float(temperature)
    return config


def _condition_data():
    keys = _load_json(KEYS_PATH)
    ids = keys["ids"][:N_SAMPLES]
    k_ps = keys["k_ps"][:N_SAMPLES]
    texts = load_abstracts(N_SAMPLES)
    if not (len(texts) == len(ids) == len(k_ps) == N_SAMPLES):
        raise ValueError("Expected exactly 100 abstracts, IDs, and watermark keys")
    if len(set(ids)) != 1:
        raise ValueError("Direct verification expects the 100 inputs to share one ID")
    if len(set(k_ps)) != N_SAMPLES:
        raise ValueError("The 100 candidate watermark keys must be unique")
    return texts, ids, k_ps


def _write_or_validate(path: Path, filename: str, value) -> None:
    destination = REPO_ROOT / path / filename
    if destination.exists():
        with destination.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != value:
            raise ValueError(
                f"Refusing to resume incompatible ablation output: {destination}"
            )
        return
    write_path_file_atomic(list(path.parts), filename, value)


def summarize_verification(verification: dict) -> dict:
    ranks = verification["correct_ranks"]
    correct_scores = verification["correct_scores"]
    top_keys = verification["top_k_p"]
    top_scores = verification["top_scores"]
    correct_keys = verification["correct_k_ps"]
    if not ranks:
        raise ValueError("Cannot summarize an empty verification result")

    margins = []
    for correct_key, correct_score, keys, scores in zip(
        correct_keys, correct_scores, top_keys, top_scores
    ):
        best_wrong_score = next(
            score for key, score in zip(keys, scores) if key != correct_key
        )
        margins.append(correct_score - best_wrong_score)

    count = len(ranks)
    return {
        "n_texts": count,
        "n_candidates": verification["n_candidates"],
        "top1_accuracy": sum(rank <= 1 for rank in ranks) / count,
        "top10_accuracy": sum(rank <= 10 for rank in ranks) / count,
        "top100_accuracy": sum(rank <= 100 for rank in ranks) / count,
        "mean_reciprocal_rank": sum(1.0 / rank for rank in ranks) / count,
        "mean_rank": sum(ranks) / count,
        "median_rank": median(ranks),
        "mean_correct_score": sum(correct_scores) / count,
        "mean_correct_vs_best_wrong_margin": sum(margins) / count,
    }


def run_condition(
    kappa: int,
    dry_run: bool = False,
    temperature: float | None = None,
) -> dict | None:
    if kappa not in KAPPAS:
        raise ValueError(f"kappa must be one of {KAPPAS}; got {kappa}")

    output_path = _output_path(kappa, temperature)
    config = _condition_config(kappa, temperature)
    texts, ids, k_ps = _condition_data()
    manifest = {
        "experiment": (
            "qwen_temperature_source"
            if temperature is not None
            else "qwen_kappa_source"
        ),
        "purpose": "direct watermark detection without fine-tuning",
        "kappa": float(kappa),
        "temperature": config["temperature_watermark"],
        "n_samples": N_SAMPLES,
        "candidate_count": N_SAMPLES,
        "source_indices": [0, N_SAMPLES - 1],
        "base_config": str(BASE_CONFIG_PATH),
        "keys_file": str(KEYS_PATH),
        "generation_seed": GENERATION_SEED,
        "output_dir": str(output_path),
    }

    print(json.dumps(manifest, indent=2))
    if dry_run:
        return None

    _write_or_validate(output_path, "config.json", config)
    _write_or_validate(output_path, "keys.json", {"ids": ids, "k_ps": k_ps})
    _write_or_validate(output_path, "original_texts.json", texts)
    _write_or_validate(output_path, "manifest.json", manifest)

    # Importing this module loads Transformers, Torch, and Waterfall, so leave
    # it until after argument/data validation and --dry-run handling.
    from src.util.watermark import (
        init_watermarker,
        verify_watermarks_full,
        watermark,
    )

    watermarked_texts = watermark(
        texts,
        ids,
        k_ps,
        config,
        experiment_path=list(output_path.parts),
        return_sts_scores=False,
        resume=True,
        checkpoint_every=config["checkpoint_every"],
        progress_metadata=manifest,
    )

    _, _, watermarker = init_watermarker(config, load_model=False)
    verify_watermarks_full(
        watermarked_texts,
        [ids[0]],
        k_ps,
        watermarker,
        list(output_path.parts),
        top_k=N_SAMPLES,
        candidate_k_ps=k_ps,
    )
    verification = _load_json(output_path / "verification_closed.json")
    summary = {
        "kappa": float(kappa),
        "temperature": config["temperature_watermark"],
        **summarize_verification(verification),
    }
    write_path_file_atomic(list(output_path.parts), "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def aggregate() -> list[dict]:
    summaries = []
    missing = []
    for kappa in KAPPAS:
        path = _output_path(kappa) / "summary.json"
        if not (REPO_ROOT / path).is_file():
            missing.append(str(path))
        else:
            summaries.append(_load_json(path))
    if missing:
        raise FileNotFoundError("Missing completed conditions: " + ", ".join(missing))
    summaries.sort(key=lambda item: item["kappa"])
    write_path_file_atomic(list(OUTPUT_ROOT.parts), "summary.json", summaries)
    print(json.dumps(summaries, indent=2))
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kappa", nargs="?", type=int, choices=KAPPAS)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "override temperature_watermark and write to the isolated "
            "qwen_temperature_source result tree"
        ),
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="combine the three completed per-kappa summaries",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print the resolved condition without loading Qwen",
    )
    args = parser.parse_args(argv)
    if args.aggregate:
        if args.kappa is not None or args.dry_run or args.temperature is not None:
            parser.error(
                "--aggregate cannot be combined with kappa, --temperature, or --dry-run"
            )
        return aggregate()
    if args.kappa is None:
        parser.error("kappa is required unless --aggregate is used")
    return run_condition(
        args.kappa,
        dry_run=args.dry_run,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()

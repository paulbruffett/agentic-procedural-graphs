"""Optional Weights & Biases metrics logging (metrics only, no LLM tracing).

Enabled when PG_WANDB_PROJECT is set; requires `uv sync --extra wandb`. Local files remain the source of truth.
"""
from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict
from pathlib import Path

from pg.config import Config


def start_run(cfg: Config, job_type: str, name: str, extra: dict) -> AbstractContextManager:
    """A wandb run context, or a no-op context yielding None when W&B is not configured."""
    if not cfg.wandb_project:
        return nullcontext()
    try:
        import wandb
    except ImportError as e:
        raise RuntimeError("PG_WANDB_PROJECT is set but wandb is not installed; run `uv sync --extra wandb`.") from e
    config = {k: str(v) if isinstance(v, Path) else v for k, v in {**asdict(cfg), **extra}.items()}
    return wandb.init(project=cfg.wandb_project, job_type=job_type, name=name, config=config)


def log_table(run, key: str, rows: list[dict]) -> None:
    if not rows:
        return
    import wandb

    columns = list(dict.fromkeys(k for r in rows for k in r))
    data = [[r.get(c) for c in columns] for r in rows]
    run.log({key: wandb.Table(columns=columns, data=data, allow_mixed_types=True)})


def log_files(run, name: str, artifact_type: str, paths: list[Path]) -> None:
    import wandb

    artifact = wandb.Artifact(name, type=artifact_type)
    for p in paths:
        artifact.add_file(str(p))
    run.log_artifact(artifact)

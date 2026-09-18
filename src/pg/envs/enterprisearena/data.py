"""Fetch the EnterpriseArena simulator (CFO-Env) and write seeded task splits.

The simulator is not redistributed with this repo. It is downloaded from the benchmark authors' anonymous review
mirror into third_party/cfo-env/ (gitignored), and every file is checked against the pinned sha256 manifest
cfo_env_sha256.json (snapshot of 2026-05-07). The code has no license file; use it for non-commercial research.
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from pg.config import ROOT
from pg.envs.base import Task, write_tasks

CODE_DIR = ROOT / "third_party" / "cfo-env"
MIRROR_ZIP = "https://anonymous.4open.science/api/repo/CFO-Env-F1B9/zip"
MANIFEST: dict[str, str] = json.loads(Path(__file__).with_name("cfo_env_sha256.json").read_text())
FULL_MONTHS = 131  # last month index; the simulator caps its configured 132 at the 2015-2025 data length
PROBE_MONTHS = 36  # covers the first hidden growth surge (months ~23-36)
EVOLVE_MONTHS = 66  # train/val horizon: covers the first two growth surges at about half the cost per episode


def verify(code_dir: Path = CODE_DIR) -> list[str]:
    """Files that are missing or differ from the pinned manifest (empty list = verified)."""
    return [
        name for name, digest in MANIFEST.items()
        if not (code_dir / name).is_file() or hashlib.sha256((code_dir / name).read_bytes()).hexdigest() != digest
    ]


def fetch_code(code_dir: Path = CODE_DIR) -> None:
    if not verify(code_dir):
        return
    print(f"downloading CFO-Env from {MIRROR_ZIP}", flush=True)
    request = urllib.request.Request(MIRROR_ZIP, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as resp:
        payload = resp.read()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for name in archive.namelist():
                if not (root / name).resolve().is_relative_to(root):
                    raise RuntimeError(f"unsafe path in CFO-Env zip: {name}")
            archive.extractall(root)
        mismatched = verify(root)
        if mismatched:
            raise RuntimeError(f"downloaded CFO-Env does not match the pinned manifest: {mismatched[:5]}")
        shutil.copytree(root, code_dir, dirs_exist_ok=True)


def make_task(seed: int, months: int) -> Task:
    # The horizon is deliberately not disclosed to the solver, matching the benchmark's own agent.
    prompt = (
        "You are the CFO of a consumer-lending fintech. Keep the company's cash non-negative every month. "
        "Each month, optionally gather information, then end the month with exactly one action."
    )
    return Task(id=f"ea-{months}m-seed{seed}", prompt=prompt, meta={"seed": seed, "months": months})


def generate(
    out_dir: Path, seed: int = 0, n_train: int = 20, n_val: int = 20, n_test: int = 20, n_probe: int = 5
) -> dict[str, int]:
    """Episodes differ only by simulator seed (same company, same macro timeline), 20/20/20 as in the
    Procedural Graphs self-evolution study. train/val run to EVOLVE_MONTHS so evolution stays affordable;
    `test` keeps the benchmark's full horizon, and `probe` is a short split for cost checks."""
    fetch_code()
    splits = {"train": (n_train, EVOLVE_MONTHS), "val": (n_val, EVOLVE_MONTHS), "test": (n_test, FULL_MONTHS),
              "probe": (n_probe, PROBE_MONTHS)}
    for k, (split, (n, months)) in enumerate(splits.items()):
        write_tasks(out_dir / f"{split}.jsonl", [make_task(seed * 10_000 + 1_000 * k + i, months) for i in range(n)])
    return {split: n for split, (n, _) in splits.items()}

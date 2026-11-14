"""Vendored snapshot of the official CLaS shared-task repository.

``evaluate.py`` in that checkout is the scorer. This module only puts it on
``sys.path`` so we can reuse its language list and JSON helpers. It does not
reimplement language identification or the relevance judge.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Snapshot of https://github.com/d-gurgurov/CLaS-SemEval-2027
# copied into third_party so a clone of this repo needs no second checkout.
# The same hash is in third_party/CLaS-SemEval-2027/UPSTREAM_COMMIT.
OFFICIAL_COMMIT = "2702627c3bb7d0cf8ed0598cdc326f77f2ed362e"

# Top-level keys evaluate.py indexes, plus the extra keys run_baseline.py writes.
# The scorer itself reads source_language, target_language, results[].output,
# and results[].original_prompt. The glob is *_to_*.json.
EVALUATOR_TOP_KEYS = (
    "source_language",
    "target_language",
    "model",
    "method",
    "track",
    "results",
)
EVALUATOR_RESULT_KEYS = (
    "question_idx",
    "input",
    "output",
    "original_prompt",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def official_root() -> Path:
    return project_root() / "third_party" / "CLaS-SemEval-2027"


def checkout_commit(root: Path | None = None) -> str | None:
    """Return the upstream commit recorded for the vendored official code."""
    root = official_root() if root is None else root
    pin = root / "UPSTREAM_COMMIT"
    if pin.is_file():
        return pin.read_text(encoding="utf-8").strip()
    git_dir = root / ".git"
    head = git_dir / "HEAD"
    if not head.is_file():
        return None
    text = head.read_text(encoding="utf-8").strip()
    if not text.startswith("ref:"):
        return text
    ref = text.split(" ", 1)[1].strip()
    ref_file = git_dir / ref
    if ref_file.is_file():
        return ref_file.read_text(encoding="utf-8").strip()
    packed = git_dir / "packed-refs"
    if packed.is_file():
        prefix = f"{ref} "
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or line.startswith("^"):
                continue
            sha, _, name = line.partition(" ")
            if name == ref:
                return sha.strip()
    return None


def import_official():
    """Import ``clas.languages`` and ``clas.io_utils`` from the pinned checkout."""
    root = official_root()
    if not (root / "clas" / "languages.py").is_file():
        raise FileNotFoundError(
            "Vendored CLaS code is missing at "
            f"{root}. It ships inside this repo at commit {OFFICIAL_COMMIT}."
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from clas import io_utils, languages

    return languages, io_utils

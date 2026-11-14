"""Pinned checkout of the official CLaS shared-task repository.

``evaluate.py`` in that checkout is the scorer. This module only puts it on
``sys.path`` so we can reuse its language list and JSON helpers. It does not
reimplement language identification or the relevance judge.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Shallow clone of https://github.com/d-gurgurov/CLaS-SemEval-2027
# `git rev-parse HEAD` at the time this project was wired up.
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
    """Return the git commit of the official checkout, if the clone is intact."""
    root = official_root() if root is None else root
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
            "Official CLaS repo is missing at "
            f"{root}. Clone https://github.com/d-gurgurov/CLaS-SemEval-2027 "
            f"(commit {OFFICIAL_COMMIT})."
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from clas import io_utils, languages

    return languages, io_utils

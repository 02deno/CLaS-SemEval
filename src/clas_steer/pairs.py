"""Language pairs for the first DiffMean slice.

Track labels follow ``clas.languages.pairs`` in the official repo:

- T1 is high-resource to high-resource, excluding the same language.
- T2 is high-resource to low-resource.
- T3 is any task language into one selected target (ja, de, or ru).
"""

from __future__ import annotations

DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("en", "de"),
    ("en", "tr"),
    ("ar", "en"),
    ("en", "sw"),
    ("en", "mt"),
    ("ru", "uk"),
]


def parse_pairs(text: str | None) -> list[tuple[str, str]]:
    """Parse ``en:de,en:tr`` or ``en-de,ar-en``. Empty input is the default slice."""
    if text is None or not text.strip():
        return list(DEFAULT_PAIRS)
    pairs: list[tuple[str, str]] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        separator = ":" if ":" in chunk else "-"
        if separator not in chunk:
            raise ValueError(
                f"pair {chunk!r} needs a ':' or '-' between source and target"
            )
        source, target = chunk.split(separator, 1)
        source, target = source.strip(), target.strip()
        if not source or not target:
            raise ValueError(f"pair {chunk!r} is missing a language code")
        pairs.append((source, target))
    if not pairs:
        raise ValueError("no language pairs given")
    return pairs


def infer_track(source: str, target: str, languages) -> str:
    """Pick T1 or T2 when the pair fits those tracks. Otherwise T3 if it qualifies."""
    if source in languages.HIGH and target in languages.HIGH and source != target:
        return "t1"
    if source in languages.HIGH and target in languages.LOW and source != target:
        return "t2"
    if (
        target in languages.T3_TARGETS
        and source in languages.TRACK_LANGS
        and source != target
    ):
        return "t3"
    raise ValueError(
        f"{source}->{target} is not a T1, T2, or T3 pair. "
        "Same-language pairs belong to the bonus track, which this scaffold "
        "does not generate yet."
    )


def generation_dir(out_root, model_name: str, track: str, target: str):
    """Match the official layout, with ``diffmean`` in place of the prompt name."""
    short = model_name.rstrip("/").split("/")[-1]
    if track == "t3":
        return out_root / short / "t3" / target / "diffmean"
    return out_root / short / track / "diffmean"


def method_name(contrast: str, layers: list[int], alpha: float) -> str:
    layer_text = "+".join(str(layer) for layer in layers)
    alpha_text = format(alpha, "g")
    return f"diffmean:{contrast}:L{layer_text}:a{alpha_text}"

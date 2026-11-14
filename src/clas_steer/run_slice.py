"""Run DiffMean steering on a small language-pair slice.

Dry run, which prints pairs and loads any text that is already on disk.
It does not import PyTorch and does not download a model:

    python src/clas_steer/run_slice.py --dry-run

A real run collects residual means on FLORES lines, adds the unit DiffMean
vector at the chosen layers, and writes the official JSON:

    python src/clas_steer/run_slice.py --layers 28 --alpha 1 --num-questions 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from clas_steer.official import (  # noqa: E402
    OFFICIAL_COMMIT,
    checkout_commit,
    import_official,
    project_root,
)
from clas_steer.pairs import (  # noqa: E402
    generation_dir,
    infer_track,
    method_name,
    parse_pairs,
)


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (root / path)


def parse_layers(text: str) -> list[int]:
    try:
        layers = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "layers must be integers, for example 28 or 16,20,28"
        ) from exc
    if not layers:
        raise argparse.ArgumentTypeError("pass at least one layer")
    return layers


def parse_lang_list(text: str | None) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def positive_or_all(value: int | None) -> int | None:
    """``0`` or negative means 'use every line in the file'."""
    if value is None or value <= 0:
        return None
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pairs",
        default=None,
        help="comma-separated source:target pairs (default: the course slice)",
    )
    parser.add_argument(
        "--layers",
        type=parse_layers,
        default=[28],
        help="0-based decoder layers (default: 28)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="multiplier on the unit DiffMean vector (default: 1)",
    )
    parser.add_argument(
        "--contrast",
        choices=("pair", "others"),
        default="pair",
        help="pair = target minus source; others = target minus the other languages",
    )
    parser.add_argument(
        "--other-langs",
        default=None,
        help="contrast languages for --contrast others (default: every language in --pairs)",
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--questions-dir", type=Path, default=Path("data/validation"))
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--out", type=Path, default=Path("generations"))
    parser.add_argument("--vectors-dir", type=Path, default=Path("vectors"))
    parser.add_argument(
        "--track",
        choices=("t1", "t2", "t3"),
        default=None,
        help="write this track label instead of inferring it from the pair",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="first k validation questions (default: all 70)",
    )
    parser.add_argument(
        "--num-sentences",
        type=int,
        default=512,
        help="FLORES lines per language for the means (0 = all downloaded lines)",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="truncation length while collecting residual means",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--recollect",
        action="store_true",
        help="recompute residual means even if a cached .pt file exists",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="regenerate JSON files that already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print pairs and load text files; do not load a model",
    )
    return parser


def _count_note(path: Path, io_utils, limit: int | None) -> str:
    if not path.is_file():
        return f"MISSING {path}"
    lines = io_utils.read_lines(path, limit)
    return f"{len(lines)} lines  {path}"


def _warn_commit() -> str | None:
    commit = checkout_commit()
    print(f"official checkout: {commit or 'unknown'}")
    print(f"pinned commit:     {OFFICIAL_COMMIT}")
    if commit and commit != OFFICIAL_COMMIT:
        print(
            "warning: this checkout is not the pinned commit. "
            "JSON fields were checked against the pin."
        )
    return commit


def dry_run(args, root: Path) -> int:
    languages, io_utils = import_official()
    pairs = parse_pairs(args.pairs)
    other_langs = parse_lang_list(args.other_langs)
    questions_dir = _resolve(root, args.questions_dir)
    train_dir = _resolve(root, args.train_dir)
    question_limit = positive_or_all(args.num_questions)
    sentence_limit = positive_or_all(args.num_sentences)
    _warn_commit()
    print(
        f"method: {method_name(args.contrast, args.layers, args.alpha)}  "
        f"model: {args.model}"
    )
    print(f"questions: {questions_dir}")
    print(f"flores:     {train_dir}")
    if args.contrast == "pair" and other_langs:
        print("--other-langs is unused when --contrast pair")
    print(f"{len(pairs)} pairs")
    missing = False
    seen_train: set[str] = set()
    for source, target in pairs:
        for code in (source, target):
            languages.get(code)
        track = args.track or infer_track(source, target, languages)
        question_path = questions_dir / f"{source}.txt"
        out_dir = generation_dir(_resolve(root, args.out), args.model, track, target)
        json_path = out_dir / f"{source}_to_{target}.json"
        question_note = _count_note(question_path, io_utils, question_limit)
        if "MISSING" in question_note:
            missing = True
        print(f"  {source} -> {target}  track={track}")
        print(f"    questions {question_note}")
        for code in (source, target):
            if code in seen_train:
                continue
            seen_train.add(code)
            train_path = train_dir / f"{code}.txt"
            train_note = _count_note(train_path, io_utils, sentence_limit)
            if "MISSING" in train_note:
                missing = True
            print(f"    flores {code} {train_note}")
        print(f"    json {json_path}")
    if args.contrast == "others":
        pool = other_langs or sorted({code for pair in pairs for code in pair})
        print(f"contrast pool: {', '.join(pool)}")
        for code in pool:
            languages.get(code)
            if code in seen_train:
                continue
            train_path = train_dir / f"{code}.txt"
            train_note = _count_note(train_path, io_utils, sentence_limit)
            if "MISSING" in train_note:
                missing = True
            print(f"    flores {code} {train_note}")
    if missing:
        print()
        print("Some text files are not on disk yet. Download them with:")
        print("  python third_party/CLaS-SemEval-2027/download_data.py --out data")
    print()
    print("dry run only: no model was loaded")
    return 0


def _means_path(vectors_dir: Path, model_name: str, code: str) -> Path:
    short = model_name.rstrip("/").split("/")[-1]
    return vectors_dir / short / "means" / f"{code}.pt"


def _load_cached_means(path: Path, layers: list[int], num_sentences, model_name: str):
    from clas_steer.vectors import load_vectors

    stored, meta = load_vectors(path)
    cached_layers = set(meta.get("layers") or stored.keys())
    same_sample = meta.get("num_sentences") == num_sentences
    same_model = meta.get("model") == model_name
    if not same_sample or not same_model or not set(layers).issubset(cached_layers):
        return None
    return {layer: stored[layer] for layer in layers}


def _collect_language(
    model,
    tokenizer,
    io_utils,
    code: str,
    train_dir: Path,
    vectors_dir: Path,
    model_name: str,
    layers: list[int],
    num_sentences,
    batch_size: int,
    max_length: int,
    recollect: bool,
) -> dict:
    from clas_steer.collect import collect_residual_means
    from clas_steer.vectors import save_vectors

    path = _means_path(vectors_dir, model_name, code)
    if path.is_file() and not recollect:
        cached = _load_cached_means(path, layers, num_sentences, model_name)
        if cached is not None:
            print(f"means {code}: cache {path}")
            return cached
        print(f"means {code}: cache does not cover this sample, recollecting")
    train_path = train_dir / f"{code}.txt"
    if not train_path.is_file():
        raise FileNotFoundError(
            f"missing FLORES lines for '{code}': {train_path}\n"
            "Download with: python third_party/CLaS-SemEval-2027/download_data.py --out data"
        )
    texts = io_utils.read_lines(train_path, num_sentences)
    if not texts:
        raise RuntimeError(f"no sentences in {train_path}")
    print(f"means {code}: {len(texts)} sentences from {train_path}")
    collected = collect_residual_means(
        model,
        tokenizer,
        texts,
        layers,
        batch_size=batch_size,
        max_length=max_length,
    )
    save_vectors(
        path,
        collected,
        {
            "kind": "residual_mean",
            "language": code,
            "model": model_name,
            "num_sentences": num_sentences,
            "layers": list(layers),
            "position": "last_real_token",
        },
    )
    print(f"means {code}: wrote {path}")
    return collected


def _pair_vectors(means, source: str, target: str, contrast: str, pool: list[str], layers: list[int]):
    from clas_steer.vectors import average_means, diffmean

    vectors = {}
    for layer in layers:
        target_mean = means[target][layer]
        if contrast == "pair":
            other = means[source][layer]
        else:
            others = [means[code][layer] for code in pool if code != target]
            if not others:
                raise ValueError(f"no contrast languages left for target {target}")
            other = average_means(others)
        vectors[layer] = diffmean(target_mean, other)
    return vectors


def run_steering(args, root: Path) -> int:
    languages, io_utils = import_official()
    from clas_steer.generate import build_record, generate_answers, load_causal_lm
    from clas_steer.hooks import SteeringHook

    pairs = parse_pairs(args.pairs)
    other_langs = parse_lang_list(args.other_langs)
    for source, target in pairs:
        languages.get(source)
        languages.get(target)
        if args.track is None:
            infer_track(source, target, languages)
    pool = other_langs or sorted({code for pair in pairs for code in pair})
    if args.contrast == "others":
        for code in pool:
            languages.get(code)
    needed = sorted({code for pair in pairs for code in pair} | set(pool if args.contrast == "others" else []))

    questions_dir = _resolve(root, args.questions_dir)
    train_dir = _resolve(root, args.train_dir)
    out_root = _resolve(root, args.out)
    vectors_dir = _resolve(root, args.vectors_dir)
    question_limit = positive_or_all(args.num_questions)
    sentence_limit = positive_or_all(args.num_sentences)
    label = method_name(args.contrast, args.layers, args.alpha)
    _warn_commit()
    print(f"loading {args.model}")
    model, tokenizer = load_causal_lm(args.model, trust_remote_code=args.trust_remote_code)

    means = {}
    for code in needed:
        means[code] = _collect_language(
            model,
            tokenizer,
            io_utils,
            code,
            train_dir,
            vectors_dir,
            args.model,
            args.layers,
            sentence_limit,
            args.batch_size,
            args.max_length,
            args.recollect,
        )

    for index, (source, target) in enumerate(pairs, start=1):
        track = args.track or infer_track(source, target, languages)
        out_dir = generation_dir(out_root, args.model, track, target)
        json_path = out_dir / f"{source}_to_{target}.json"
        if json_path.exists() and not args.overwrite:
            print(f"[{index}/{len(pairs)}] {source}->{target}: exists, skipping")
            continue
        questions = io_utils.load_questions(questions_dir, [source], question_limit)[source]
        vectors = _pair_vectors(means, source, target, args.contrast, pool, args.layers)
        dead = [
            layer
            for layer, vector in vectors.items()
            if float(vector.detach().float().norm()) == 0.0
        ]
        if dead:
            print(f"warning: zero DiffMean vector at layers {dead}; those layers are unchanged")
        print(
            f"[{index}/{len(pairs)}] {source}->{target} track={track} "
            f"{len(questions)} questions -> {json_path}"
        )
        hook = SteeringHook(model, vectors, args.alpha)
        outputs = generate_answers(
            model,
            tokenizer,
            questions,
            hook,
            max_new_tokens=args.max_tokens,
            batch_size=args.batch_size,
        )
        record = build_record(
            source,
            target,
            args.model,
            label,
            track,
            questions,
            outputs,
        )
        io_utils.write_json(json_path, record)
    print("done")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = project_root()
    if args.dry_run:
        return dry_run(args, root)
    return run_steering(args, root)


if __name__ == "__main__":
    raise SystemExit(main())

"""Stdlib checks for the scaffold. Vector math is skipped when torch is absent."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from clas_steer.generate import build_record  # noqa: E402
from clas_steer.official import (  # noqa: E402
    EVALUATOR_RESULT_KEYS,
    EVALUATOR_TOP_KEYS,
    OFFICIAL_COMMIT,
    checkout_commit,
    import_official,
    official_root,
)
from clas_steer.pairs import (  # noqa: E402
    DEFAULT_PAIRS,
    generation_dir,
    infer_track,
    method_name,
    parse_pairs,
)


class PairTests(unittest.TestCase):
    def test_default_slice_tracks(self) -> None:
        languages, _ = import_official()
        self.assertEqual(parse_pairs(None), DEFAULT_PAIRS)
        self.assertEqual(parse_pairs("  "), DEFAULT_PAIRS)
        tracks = [infer_track(source, target, languages) for source, target in DEFAULT_PAIRS]
        self.assertEqual(tracks, ["t1", "t2", "t1", "t2", "t2", "t2"])

    def test_parse_both_separators(self) -> None:
        self.assertEqual(parse_pairs("en:de, ar-en"), [("en", "de"), ("ar", "en")])

    def test_same_language_is_rejected(self) -> None:
        languages, _ = import_official()
        with self.assertRaises(ValueError):
            infer_track("en", "en", languages)

    def test_method_and_output_dir(self) -> None:
        self.assertEqual(method_name("pair", [28], 1.0), "diffmean:pair:L28:a1")
        self.assertEqual(
            method_name("others", [8, 28], 0.5),
            "diffmean:others:L8+28:a0.5",
        )
        out = generation_dir(Path("generations"), "meta-llama/Llama-3.1-8B-Instruct", "t2", "tr")
        self.assertEqual(
            out.as_posix(),
            "generations/Llama-3.1-8B-Instruct/t2/diffmean",
        )
        t3 = generation_dir(Path("generations"), "org/Model", "t3", "de")
        self.assertEqual(t3.as_posix(), "generations/Model/t3/de/diffmean")


class RecordTests(unittest.TestCase):
    def test_record_matches_evaluator_keys(self) -> None:
        question = "How are you?"
        record = build_record(
            "en",
            "tr",
            "meta-llama/Llama-3.1-8B-Instruct",
            "diffmean:pair:L28:a1",
            "t2",
            [question],
            ["İyiyim."],
        )
        for key in EVALUATOR_TOP_KEYS:
            self.assertIn(key, record)
        row = record["results"][0]
        for key in EVALUATOR_RESULT_KEYS:
            self.assertIn(key, row)
        self.assertEqual(row["input"], question)
        self.assertEqual(row["original_prompt"], question)
        self.assertNotIn("Respond in", row["input"])
        self.assertEqual(record["source_language"], "en")
        self.assertEqual(record["target_language"], "tr")

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_record("en", "de", "m", "method", "t1", ["q"], [])


class PinTests(unittest.TestCase):
    def test_checkout_matches_pin(self) -> None:
        self.assertEqual(checkout_commit(official_root()), OFFICIAL_COMMIT)


class DryRunTests(unittest.TestCase):
    def test_dry_run_prints_slice_without_torch(self) -> None:
        script = ROOT / "src" / "clas_steer" / "run_slice.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--dry-run"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = completed.stdout
        self.assertIn("en -> de", output)
        self.assertIn("en -> tr", output)
        self.assertIn("track=t1", output)
        self.assertIn("track=t2", output)
        self.assertIn("diffmean:pair:L28:a1", output)
        self.assertIn(OFFICIAL_COMMIT, output)
        self.assertIn("no model was loaded", output)
        self.assertNotIn("import torch", output.lower())


class TorchTests(unittest.TestCase):
    def test_diffmean_and_hook(self) -> None:
        try:
            import torch
            from torch import nn
        except ImportError:
            self.skipTest("torch is not installed")
        from clas_steer.collect import last_real_index
        from clas_steer.hooks import SteeringHook
        from clas_steer.vectors import average_means, diffmean, load_vectors, save_vectors

        vector = diffmean(torch.tensor([3.0, 0.0]), torch.tensor([1.0, 0.0]))
        self.assertTrue(torch.allclose(vector, torch.tensor([1.0, 0.0])))
        self.assertAlmostEqual(float(torch.linalg.vector_norm(vector)), 1.0)
        zero = diffmean(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))
        self.assertTrue(torch.allclose(zero, torch.zeros(2)))
        pooled = average_means(
            [torch.tensor([0.0, 2.0]), torch.tensor([2.0, 0.0])]
        )
        self.assertTrue(torch.allclose(pooled, torch.tensor([1.0, 1.0])))

        right = last_real_index(torch.tensor([[1, 1, 1, 0, 0]]))
        left = last_real_index(torch.tensor([[0, 0, 1, 1, 1]]))
        self.assertEqual(int(right[0]), 2)
        self.assertEqual(int(left[0]), 4)

        class Block(nn.Module):
            def forward(self, hidden):
                return hidden

        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([Block(), Block()])

        toy = Toy()
        hidden = torch.zeros(2, 3, 2)
        with SteeringHook(toy, {0: vector}, alpha=3.0):
            shifted = toy.model.layers[0](hidden)
            untouched = toy.model.layers[1](hidden)
        self.assertTrue(torch.allclose(shifted, torch.tensor([3.0, 0.0]).expand_as(hidden)))
        self.assertTrue(torch.allclose(untouched, hidden))
        self.assertEqual(len(list(toy.model.layers[0]._forward_hooks)), 0)

        path = ROOT / "vectors" / "_test" / "unit.pt"
        try:
            save_vectors(path, {28: vector}, {"language": "tr", "layers": [28]})
            loaded, meta = load_vectors(path)
        finally:
            if path.exists():
                path.unlink()
            parent = path.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        self.assertTrue(torch.allclose(loaded[28], vector))
        self.assertEqual(meta["language"], "tr")


if __name__ == "__main__":
    unittest.main()

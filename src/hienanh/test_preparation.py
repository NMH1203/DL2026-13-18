"""Regression checks for annotation geometry and pretrained model compatibility."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

from prepare_dataset import convert_labels
from src.enhancement import load_model


ROOT = Path(__file__).resolve().parents[1]


class LabelConversionTests(unittest.TestCase):
    def test_rotated_box_uses_all_four_corners(self):
        text, classes, issues = convert_labels("7 0.5 0.1 0.9 0.5 0.5 0.9 0.1 0.5", 12)
        self.assertEqual(classes, [7])
        np.testing.assert_allclose(list(map(float, text.split()[1:])), [0.5, 0.5, 0.8, 0.8])
        self.assertEqual(issues, [])

    def test_clipping_is_logged_and_degenerate_boxes_are_dropped(self):
        text, classes, issues = convert_labels("2 -0.1 0.2 0.3 0.2 0.3 0.6 -0.1 0.6\n1 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2", 12)
        np.testing.assert_allclose(list(map(float, text.split()[1:])), [0.15, 0.4, 0.3, 0.4])
        self.assertEqual(classes, [2])
        self.assertEqual([item["action"] for item in issues], ["clip_to_image", "drop_zero_area_box"])

    def test_invalid_labels_fail(self):
        for text in ("12 0.5 0.5 0.1 0.1", "1.5 0.5 0.5 0.1 0.1", "0 nan 0.5 0.1 0.1", "0 1 2", "0 0.5 0.5 -0.1 0.1"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                convert_labels(text, 12)

    def test_empty_and_duplicate_annotations(self):
        self.assertEqual(convert_labels("\n", 12), ("", [], []))
        text, classes, issues = convert_labels("3 0.5 0.5 0.2 0.4\n3 0.5 0.5 0.2 0.4", 12)
        self.assertEqual(len(text.splitlines()), 1)
        self.assertEqual(classes, [3])
        self.assertEqual(issues[0]["action"], "drop_duplicate_box")


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_outputs_match_official_implementations(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(42)
        image = torch.rand(1, 3, 32, 48, generator=generator) * 0.4
        for variant, scale in [("zerodce", 1), ("zerodcepp", 1), ("zerodcepp", 4)]:
            with self.subTest(variant=variant, scale=scale):
                reference_path = ROOT / "references" / f"official_{variant}_model.py"
                spec = importlib.util.spec_from_file_location("official_model", reference_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                checkpoint = ROOT / "Results" / "weights" / f"{variant}_Epoch99.pth"
                official = module.enhance_net_nopool() if variant == "zerodce" else module.enhance_net_nopool(scale)
                official.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
                official.eval()
                actual = load_model(checkpoint, variant, scale)
                with torch.inference_mode():
                    expected = official(image)[1 if variant == "zerodce" else 0]
                    result = actual(image)
                torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)
                self.assertGreaterEqual(float(result.min()), 0)
                self.assertLessEqual(float(result.max()), 1)


if __name__ == "__main__":
    unittest.main()

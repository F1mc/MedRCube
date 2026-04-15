"""
MedRCube evaluator.

This module is a **library** — import ``MedRCubeEvaluator`` from your
own script or use ``run_eval.py`` as the CLI entry point.
"""

import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from models import ModelAdapter, SampleMessage


class MedRCubeEvaluator:
    """Load the benchmark dataset, run inference, and produce a three-tier report."""

    def __init__(
        self,
        model: ModelAdapter,
        dataset_path: str,
        output_path: str = "eval_results",
        strict: bool = False,
    ):
        self.model = model
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.strict = strict
        self.samples: List[Dict[str, Any]] = []
        os.makedirs(output_path, exist_ok=True)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self) -> List[Dict[str, Any]]:
        print(f"Loading data from {self.dataset_path}")
        skipped = {
            "missing_image": 0,
            "missing_required_fields": 0,
            "missing_options": 0,
        }
        for root, _, files in os.walk(self.dataset_path):
            if "test.json" not in files:
                continue
            with open(os.path.join(root, "test.json"), "r", encoding="utf-8") as f:
                for item in json.load(f):
                    img = item.get("image_path")
                    if img and not os.path.isabs(img):
                        img = os.path.normpath(os.path.join(root, img))
                        item["image_path"] = img

                    # Basic schema checks for evaluatable samples
                    if not item.get("id") or not item.get("question") or not item.get("gt_answer"):
                        skipped["missing_required_fields"] += 1
                        if self.strict:
                            raise ValueError(f"Invalid sample (missing required fields): {item}")
                        continue

                    has_any_option = any(f"option_{opt}" in item for opt in ("A", "B", "C", "D"))
                    if not has_any_option:
                        skipped["missing_options"] += 1
                        if self.strict:
                            raise ValueError(f"Invalid sample (missing options): {item.get('id')}")
                        continue

                    # Image existence check: required for vision evaluation; default is skip.
                    if img and not os.path.exists(img):
                        skipped["missing_image"] += 1
                        if self.strict:
                            raise FileNotFoundError(f"Missing image for sample id={item.get('id')}: {img}")
                        continue

                    self.samples.append(self._build_prompt(item))
        if not self.strict and any(skipped.values()):
            parts = ", ".join(f"{k}={v}" for k, v in skipped.items() if v)
            print(f"Skipped samples due to sample issues (default behavior): {parts}")
        print(f"Total samples loaded: {len(self.samples)}")
        return self.samples

    # ------------------------------------------------------------------
    # Inference + scoring
    # ------------------------------------------------------------------

    def run_eval(self, batch_size: int = 4) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        print(f"Starting inference for {len(self.samples)} samples ...")

        for i in tqdm(range(0, len(self.samples), batch_size), desc="Inferring"):
            batch = self.samples[i : i + batch_size]
            msgs = [
                SampleMessage(prompt=s["_prompt"], image_path=s.get("image_path"))
                for s in batch
            ]
            responses = self.model.generate(msgs)

            for sample, resp in zip(batch, responses):
                sample["model_response"] = resp
                sample["correct"] = self._judge(resp, sample.get("_gt_letter", ""))
                results.append(sample)

        report = self._build_report(results)
        self._save(report)
        return report

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _save(self, report: Dict[str, Any]) -> None:
        with open(os.path.join(self.output_path, "results.json"), "w", encoding="utf-8") as f:
            json.dump(report["details"], f, indent=2, ensure_ascii=False)

        summary = {k: v for k, v in report.items() if k != "details"}
        with open(os.path.join(self.output_path, "metrics_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(item: Dict[str, Any]) -> Dict[str, Any]:
        choices = []
        for opt in ("A", "B", "C", "D"):
            key = f"option_{opt}"
            if key in item:
                val = item[key]
                if item.get("gt_answer") == val:
                    item["_gt_letter"] = opt
                choices.append(f"{opt}. {val}")

        item["_prompt"] = (
            f"Question: {item['question']}\nOptions:\n"
            + "\n".join(choices)
            + "\nAnswer with the option's letter from the given choices directly."
        )
        return item

    @staticmethod
    def _judge(response: str, gt_letter: str) -> bool:
        match = re.search(r"\b([A-D])\b", str(response).upper())
        return (match.group(1) if match else "") == gt_letter

    def _build_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(results)
        nc = sum(1 for r in results if r.get("correct"))

        def _acc(records, keys):
            stats = defaultdict(lambda: {"n": 0, "correct": 0})
            for r in records:
                gid = "|".join(f"{k}={r.get(k)}" for k in keys)
                stats[gid]["n"] += 1
                if r.get("correct"):
                    stats[gid]["correct"] += 1
            return {
                gid: {"n": s["n"], "correct": s["correct"], "acc": s["correct"] / s["n"]}
                for gid, s in stats.items()
            }

        return {
            "global": {"n": n, "correct": nc, "acc": nc / n if n else 0.0},
            "slice_results": {
                "by_task": _acc(results, ["task"]),
                "by_modality": _acc(results, ["modality"]),
                "by_parts": _acc(results, ["parts"]),
                "by_dataset": _acc(results, ["dataset"]),
            },
            "voxel_results": {
                "by_task_modality_parts": _acc(results, ["task", "modality", "parts"]),
            },
            "details": results,
        }

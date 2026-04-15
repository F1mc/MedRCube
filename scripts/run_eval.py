#!/usr/bin/env python3
"""
MedRCube evaluation — single CLI entry point.

    cd scripts
    python run_eval.py --help
"""

import argparse
import json

from eval import MedRCubeEvaluator
from models import ModelAdapter


def build_model(args) -> ModelAdapter:
    if args.model_type == "local":
        from models.hf_vlm import HfVLM

        return HfVLM(args.model_path, max_new_tokens=args.max_new_tokens)

    if args.model_type == "api":
        from models.openai_api import OpenAIAPI

        return OpenAIAPI(
            model_name=args.model_name,
            api_key=args.api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
        )

    raise ValueError(f"Unknown model_type: {args.model_type}")


def main():
    p = argparse.ArgumentParser(description="MedRCube Evaluation")

    p.add_argument("--dataset_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--batch_size", type=int, default=20)
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast on any invalid sample (missing fields/options/images). Default: skip problematic samples.",
    )

    p.add_argument("--model_type", required=True, choices=["local", "api"])
    p.add_argument("--model_path", default=None, help="(local) Path to model weights")
    p.add_argument("--model_name", default=None, help="(api) Model name, e.g. gpt-4o")
    p.add_argument("--api_key", default=None)
    p.add_argument("--base_url", default=None)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max_new_tokens", type=int, default=1024)

    args = p.parse_args()
    model = build_model(args)

    evaluator = MedRCubeEvaluator(
        model=model,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        strict=args.strict,
    )
    evaluator.load_data()
    report = evaluator.run_eval(batch_size=args.batch_size)

    print("\n" + "=" * 40)
    print(json.dumps(report["global"], indent=2, ensure_ascii=False))
    print("=" * 40)


if __name__ == "__main__":
    main()

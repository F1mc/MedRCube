"""
Local model adapter — reference implementation using Qwen2.5-VL.

This file is a **working example**, not a universal adapter.
Adapt ``__init__`` and ``_infer_single`` to match your own model's
loading and chat-template conventions.

Standalone check (run from ``scripts/``):
    python -m models.hf_vlm
"""

from __future__ import annotations

from typing import List, Sequence

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from models import SampleMessage

# ── Qwen-specific import; replace with your model's utilities ──
from qwen_vl_utils import process_vision_info


class HfVLM:

    def __init__(self, model_path: str, max_new_tokens: int = 512):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_new_tokens = max_new_tokens
        print(f"Loading model from {model_path} → {self.device} ...")

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

    # ── ModelAdapter interface ──────────────────────────────────

    def generate(self, messages: Sequence[SampleMessage]) -> List[str]:
        """
        Each ``msg.prompt`` is a fully formatted string (question +
        options + instruction).  ``msg.image_path`` is an absolute
        path.  Return one free-form answer string per message; the
        evaluator extracts the option letter (A–D) via regex.
        """
        return [self._infer_single(m) for m in messages]

    # ── Internals (model-specific — edit below) ─────────────────

    def _infer_single(self, msg: SampleMessage) -> str:
        chat = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{msg.image_path}"},
                    {"type": "text", "text": msg.prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(chat)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        gen_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = [
            out[len(inp) :] for inp, out in zip(inputs.input_ids, gen_ids)
        ]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


# ── Standalone sanity check with a real MedRCube sample ─────────
if __name__ == "__main__":
    import sys

    WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else "/path/to/your/weights"
    SAMPLE_PROMPT = (
        "Question: Please observe the ultrasound image. "
        "What kind of area is the red mask area in the image?\n"
        "Options:\n"
        "A. Malignant\n"
        "B. Benign\n"
        "C. Undetermined\n"
        "D. Normal\n"
        "Answer with the option's letter from the given choices directly."
    )
    SAMPLE_IMAGE = "../MedRCube/BUSI/pictures/benign_277.png"

    model = HfVLM(WEIGHTS)
    out = model.generate([SampleMessage(prompt=SAMPLE_PROMPT, image_path=SAMPLE_IMAGE)])
    print(f"Model output: {out}")

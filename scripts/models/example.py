from __future__ import annotations

from typing import List, Sequence

from . import SampleMessage


class ExampleModel:
    """
    Minimal template for integrating a model into MedRCube evaluation.

    Implement `generate(messages)` and return a list of strings.
    Each string should contain one of: A / B / C / D (case-insensitive).
    """

    def generate(self, messages: Sequence[SampleMessage]) -> List[str]:
        # Replace this with your own model inference.
        # This placeholder always returns "A".
        return ["A" for _ in messages]

"""
Minimal template for integrating your own model.

Copy this file, rename it, fill in ``generate``, and import your class
in ``run_eval.py``.

Standalone check:
    cd scripts
    python -m models.example
"""

from __future__ import annotations

from typing import List, Sequence

from models import SampleMessage


class MyCustomModel:
    """Replace this with your own model class."""

    def __init__(self, **kwargs):
        # TODO: load your model / connect to your service here.
        pass

    def generate(self, messages: Sequence[SampleMessage]) -> List[str]:
        """
        Run inference on a batch of samples.

        Each ``msg.prompt`` is a fully formatted string that already
        contains the question, the four options (A/B/C/D), and an
        instruction asking the model to answer.  ``msg.image_path``
        is the absolute path to the corresponding image.

        Return one free-form answer string per message.  The evaluator
        will extract an option letter (A–D) from it via regex, so
        returning just the letter or a sentence containing it both work.
        """
        responses = []
        for msg in messages:
            # TODO: replace with your real inference logic.
            responses.append("A")
        return responses


# ------------------------------------------------------------------
# Standalone sanity check with a real MedRCube sample
# ------------------------------------------------------------------
if __name__ == "__main__":
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

    model = MyCustomModel()
    out = model.generate([
        SampleMessage(prompt=SAMPLE_PROMPT, image_path=SAMPLE_IMAGE),
    ])
    print(f"Input prompt:\n{SAMPLE_PROMPT}\n")
    print(f"Model output: {out}")

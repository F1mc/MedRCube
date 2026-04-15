"""
MedRCube model integration layer.

To add your own model, implement the ``ModelAdapter`` protocol:

    class MyModel:
        def generate(self, messages: list[SampleMessage]) -> list[str]:
            ...

See ``models/example.py`` for a minimal template.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence


@dataclass(frozen=True)
class SampleMessage:
    """One evaluation sample handed to the model."""

    prompt: str
    image_path: Optional[str] = None


class ModelAdapter(Protocol):
    """
    Stable public contract.

    Any object whose ``generate`` method accepts a sequence of
    ``SampleMessage`` and returns one answer string per message
    satisfies this protocol.
    """

    def generate(self, messages: Sequence[SampleMessage]) -> List[str]: ...

"""
OpenAI-compatible API model adapter.

Works out-of-the-box with any provider that exposes the
``/chat/completions`` endpoint (OpenAI, Azure, DeepSeek, vLLM, etc.).

Standalone check (run from ``scripts/``):
    python -m models.openai_api
"""

from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import json
import os
import threading
import time
from io import BytesIO
from typing import List, Sequence

from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from models import SampleMessage


class OpenAIAPI:
    """Concurrent, cache-enabled API client."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        max_workers: int = 10,
    ):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_workers = max_workers

        safe = model_name.replace("/", "_").replace(" ", "_").replace(":", "_").strip()
        self.cache_file = f"api_cache_{safe}.jsonl"
        self._lock = threading.Lock()
        self._cache = self._load_cache()

        print(f"[Init] API Model: {model_name} | Cache: {self.cache_file}")

    # ── ModelAdapter interface ──────────────────────────────────

    def generate(self, messages: Sequence[SampleMessage]) -> List[str]:
        """
        Each ``msg.prompt`` is a fully formatted string (question +
        options + instruction).  ``msg.image_path`` is an absolute
        path.  Return one free-form answer string per message; the
        evaluator extracts the option letter (A–D) via regex.
        """
        results: List[str | None] = [None] * len(messages)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._request, m): i for i, m in enumerate(messages)}
            for fut in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(messages),
                desc=f"API Infer: {self.model_name}",
            ):
                results[futures[fut]] = fut.result()
        return results  # type: ignore[return-value]

    # ── Internals ───────────────────────────────────────────────

    def _request(self, msg: SampleMessage) -> str:
        key = self._hash(msg)
        if key in self._cache:
            return self._cache[key]

        content = self._build_content(msg)
        api_msgs = [{"role": "user", "content": content}]

        for _ in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=api_msgs,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=120,
                )
                text = resp.choices[0].message.content
                self._save(key, text)
                return text
            except Exception as e:
                err = str(e).lower()
                if any(kw in err for kw in ("balance", "quota", "insufficient")):
                    print("\n[CRITICAL] Insufficient balance — waiting for input")
                    input()
                    continue
                time.sleep(5)
        return "Error: Request Failed"

    @staticmethod
    def _hash(msg: SampleMessage) -> str:
        raw = json.dumps({"p": msg.prompt, "i": msg.image_path}, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    @staticmethod
    def _build_content(msg: SampleMessage) -> list:
        parts: list = []
        if msg.prompt:
            parts.append({"type": "text", "text": msg.prompt})
        if msg.image_path and os.path.exists(msg.image_path):
            try:
                img = Image.open(msg.image_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=95)
                b64 = base64.b64encode(buf.getvalue()).decode()
                parts.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                )
            except Exception as e:
                print(f"[Warn] Failed to encode image {msg.image_path}: {e}")
        return parts

    def _load_cache(self) -> dict:
        data: dict = {}
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data.update(json.loads(line))
                    except Exception:
                        continue
        print(f"[API Cache] Loaded {len(data)} records.")
        return data

    def _save(self, key: str, value: str) -> None:
        with self._lock:
            self._cache[key] = value
            with open(self.cache_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({key: value}, ensure_ascii=False) + "\n")


# ── Standalone sanity check with a real MedRCube sample ─────────
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

    model = OpenAIAPI(
        model_name="gpt-4o",
        api_key="YOUR_KEY",
        base_url="https://api.openai.com/v1",
    )
    out = model.generate([SampleMessage(prompt=SAMPLE_PROMPT, image_path=SAMPLE_IMAGE)])
    print(f"Model output: {out}")

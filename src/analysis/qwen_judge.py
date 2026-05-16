"""
Qwen2.5-Omni judge: scores an audio file on a 0-10 musicality scale.

The model is given the raw waveform (16 kHz mono) plus a short prompt
that asks for an integer rating. We parse the first 0-10 number that
appears in the reply and treat that as the reward signal.

Loaded once per process via `get_judge(...)`. The same instance is
reused across the RLAIF training loop so the model is not reloaded for
every sample.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch


SYSTEM_PROMPT = (
    "You are an expert music critic. You will listen to a short audio "
    "clip and rate its overall musical quality on an integer scale from "
    "0 (random noise, no musical structure) to 10 (highly coherent, "
    "tonal, well-formed). Reply with ONLY the integer rating on the first "
    "line, then one short sentence of justification on the second line."
)

USER_PROMPT = (
    "Rate this audio clip from 0 to 10 on musicality. "
    "Output the integer on the first line."
)

_SCORE_RE = re.compile(r"\b(10|[0-9])\b")


@dataclass
class JudgeResult:
    path: str
    score: Optional[float]
    raw_text: str


def _load_audio(path: str | Path, target_sr: int = 16000) -> np.ndarray:
    """Read a WAV, downmix to mono, resample to `target_sr`."""
    import librosa
    audio, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return audio.astype(np.float32)


class QwenJudge:
    """Lazy wrapper around Qwen2.5-Omni thinker. One model, many calls."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-Omni-3B",
                 device: str = "cpu", dtype: str = "bfloat16",
                 max_new_tokens: int = 48):
        from transformers import (
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )
        torch_dtype = getattr(torch, dtype)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.device = device
        if device != "cpu":
            self.model.to(device)
        self.max_new_tokens = max_new_tokens
        # Cache the audio sample rate the Whisper feature extractor expects
        self.audio_sr = getattr(self.processor.feature_extractor,
                                "sampling_rate", 16000)

    @torch.no_grad()
    def score_audio(self, audio_path: str | Path,
                    waveform: Optional[np.ndarray] = None) -> JudgeResult:
        if waveform is None:
            waveform = _load_audio(audio_path, target_sr=self.audio_sr)
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": SYSTEM_PROMPT},
            ]},
            {"role": "user", "content": [
                {"type": "audio", "audio": waveform},
                {"type": "text", "text": USER_PROMPT},
            ]},
        ]
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        inputs = self.processor(
            text=text,
            audio=[waveform],
            return_tensors="pt",
            padding=True,
            sampling_rate=self.audio_sr,
        )
        if self.device != "cpu":
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v
                      for k, v in inputs.items()}
        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
        # Strip the prompt tokens, decode only the new generation.
        in_len = inputs["input_ids"].shape[1]
        gen = out_ids[:, in_len:]
        raw = self.processor.batch_decode(
            gen, skip_special_tokens=True
        )[0].strip()
        score = _parse_score(raw)
        return JudgeResult(path=str(audio_path), score=score, raw_text=raw)


def _parse_score(text: str) -> Optional[float]:
    """Pull the first 0-10 integer out of the reply."""
    for line in text.splitlines():
        m = _SCORE_RE.search(line)
        if m:
            return float(m.group(1))
    m = _SCORE_RE.search(text)
    return float(m.group(1)) if m else None


@lru_cache(maxsize=4)
def get_judge(model_id: str = "Qwen/Qwen2.5-Omni-3B",
              device: str = "cpu", dtype: str = "bfloat16") -> QwenJudge:
    return QwenJudge(model_id=model_id, device=device, dtype=dtype)


def score_directory(audio_dir: str | Path, judge: QwenJudge,
                    limit: Optional[int] = None) -> List[JudgeResult]:
    files = sorted(Path(audio_dir).glob("*.wav"))
    if limit is not None:
        files = files[:limit]
    out: List[JudgeResult] = []
    for p in files:
        r = judge.score_audio(p)
        print(f"  {p.name}: score={r.score}  raw={r.raw_text!r}", flush=True)
        out.append(r)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()

    print(f"Loading judge from {args.model} on {args.device}...", flush=True)
    judge = QwenJudge(model_id=args.model, device=args.device,
                      dtype=args.dtype, max_new_tokens=args.max_new_tokens)
    print("Judge ready.", flush=True)

    results = score_directory(args.audio_dir, judge, limit=args.limit)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            [
                {"path": r.path, "score": r.score, "raw": r.raw_text}
                for r in results
            ],
            f, indent=2,
        )
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

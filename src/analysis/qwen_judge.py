"""
Qwen2.5-Omni-7B as a music critic — runs locally on a MacBook Pro
(64 GB) and *actually listens to the audio*, unlike the feature-based
Ollama judge in scripts/judge_with_ollama.py.

Usage on the laptop (one-time install):

    pip install transformers accelerate soundfile librosa
    pip install qwen-omni-utils  # optional helper for chat-template prep

Then, for batch judging:

    python -m src.analysis.qwen_judge \\
        --audio-dir results/audio \\
        --out results/QWEN_REVIEW.json

For RLAIF, import `QwenAudioJudge.score(wav_array, sample_rate)` from
src/train/qwen_rlaif_train.py — it caches the model so each batch
only pays setup cost once.

We do NOT load the model at import time (it's ~14 GB in bf16) so this
file remains importable on CI runners that don't have it installed.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Standard prompt — phrased so the model emits a parseable score.
SYSTEM_PROMPT = (
    "You are a music critic. The user will play you a short audio clip "
    "produced by an experimental tabula-rasa music generator. The "
    "generator was trained against psychoacoustic physics rewards only "
    "(Sethares dissonance, Terhardt virtual pitch, Large-Kolen rhythmic "
    "entrainment) — it has never heard real music. Your job is to "
    "(a) describe what you actually hear in 1–2 sentences, "
    "(b) say what specifically makes it feel robotic vs alive, "
    "(c) give two concrete suggestions for the next iteration "
    "(synth choice, tempo, voicing, key), and "
    "(d) give an overall score from 1 (unlistenable) to 10 (compelling "
    "music). End with the literal line: SCORE: <integer>."
)


@dataclass
class QwenJudgeResult:
    file: str
    score: Optional[int]
    response: str

    def to_dict(self):
        return {"file": self.file, "score": self.score,
                "response": self.response}


def _parse_score(text: str) -> Optional[int]:
    m = re.search(r"SCORE:\s*(\d+)", text, re.IGNORECASE)
    if not m:
        return None
    return max(1, min(10, int(m.group(1))))


class QwenAudioJudge:
    """Stateful wrapper that loads the model once and scores many clips.

    Designed to run on Apple Silicon ("mps"), CUDA, or CPU. On a
    64 GB MBP the bf16 model fits comfortably in unified memory; on
    a CPU-only machine you'll want a quantized GGUF instead.
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-Omni-7B"
    SAMPLE_RATE = 16000     # Qwen-Omni audio encoder native rate

    def __init__(self,
                 model_name: str = DEFAULT_MODEL,
                 device: str = "auto",
                 dtype: str = "bfloat16"):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self._model = None
        self._processor = None

    # -- lazy load ---------------------------------------------------

    def _load(self):
        if self._model is not None:
            return
        import torch
        # Apple Silicon path
        if self.device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

        # transformers >= 4.46 ships Qwen2.5-Omni
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        torch_dtype = getattr(torch, self.dtype)

        print(f"[qwen_judge] loading {self.model_name} on {self.device} "
              f"(dtype={self.dtype}) — first call only")
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForMultimodalLM.from_pretrained(
            self.model_name,
            dtype=torch_dtype,
            attn_implementation="sdpa",   # Mac-compatible (no flash_attn)
        )
        self._model = self._model.to(self.device)
        self._model.eval()

    # -- core scoring ------------------------------------------------

    def _resample_to_16k(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr == self.SAMPLE_RATE:
            return audio
        try:
            import librosa
            return librosa.resample(audio.astype(np.float32),
                                     orig_sr=sr, target_sr=self.SAMPLE_RATE)
        except ImportError:
            # Cheap linear resample
            new_len = int(len(audio) * self.SAMPLE_RATE / sr)
            xp = np.linspace(0, 1, len(audio))
            x = np.linspace(0, 1, new_len)
            return np.interp(x, xp, audio).astype(np.float32)

    def score(self, audio: np.ndarray, sample_rate: int,
              max_new_tokens: int = 220,
              temperature: float = 0.4,
              system_prompt: str = SYSTEM_PROMPT) -> QwenJudgeResult:
        """Score a numpy waveform; returns a QwenJudgeResult."""
        self._load()
        import torch

        audio = self._resample_to_16k(audio, sample_rate)

        messages = [
            {"role": "system",
             "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user",
             "content": [
                 {"type": "text",
                  "text": "Critique this audio."},
                 {"type": "audio", "audio": audio,
                  "sampling_rate": self.SAMPLE_RATE},
             ]},
        ]
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_tensors="pt", return_dict=True,
        )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
            )
        # Strip the prompt prefix
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = generated[:, prompt_len:]
        text = self._processor.batch_decode(
            new_tokens, skip_special_tokens=True)[0]
        return QwenJudgeResult(
            file="<inline>",
            score=_parse_score(text),
            response=text.strip(),
        )

    def score_wav(self, path: str | Path) -> QwenJudgeResult:
        """Convenience: read a WAV from disk and score it."""
        from src.analysis.audio_features import _read_wav
        audio, sr = _read_wav(Path(path))
        result = self.score(audio.astype(np.float32), sr)
        result.file = str(path)
        return result


# -- CLI ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", default="results/audio")
    parser.add_argument("--model", default=QwenAudioJudge.DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--out", default="results/QWEN_REVIEW.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    wavs = sorted(audio_dir.glob("*.wav"))
    if args.limit:
        wavs = wavs[: args.limit]
    print(f"[qwen_judge] scoring {len(wavs)} WAVs from {audio_dir}")

    judge = QwenAudioJudge(model_name=args.model, device=args.device,
                            dtype=args.dtype)
    results = []
    for wav in wavs:
        print(f"  {wav.name}...", flush=True)
        try:
            r = judge.score_wav(wav)
        except Exception as e:    # noqa: BLE001
            print(f"    ERROR: {e}")
            continue
        print(f"    score={r.score}")
        results.append(r.to_dict())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

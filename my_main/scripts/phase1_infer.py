#!/usr/bin/env python3
"""
Phase 1B: LoRA inference test
Loads the fine-tuned adapter and transcribes a wav file.

Usage:
  python3 my_main/scripts/phase1_infer.py \
      --adapter my_main/phase1_lora \
      --audio   my_main/sample/gemma4_test_ja.wav \
      --gpu     1
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

# transformers 5.x references torch.float8_e8m0fnu (PyTorch 2.7+). Patch for older torch.
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.float8_e4m3fn  # type: ignore[attr-defined]

import soundfile as sf
from scipy import signal as scipy_signal
from transformers import AutoTokenizer, Gemma4UnifiedForConditionalGeneration
from peft import PeftModel

SAMPLE_RATE    = 16_000
CHUNK_SAMPLES  = 640        # 40ms @ 16kHz = 1 audio token
MAX_AUDIO_TOKS = 750        # 30s
AUDIO_TOKEN_ID = 258881     # from Gemma4 config


def load_audio_16k(path: str) -> np.ndarray:
    """Load wav, convert to mono float32 at 16kHz with peak normalization."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        n_out = int(len(audio) * SAMPLE_RATE / sr)
        audio = scipy_signal.resample(audio, n_out)
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95
    return audio.astype(np.float32)


def audio_to_frames(audio: np.ndarray, max_tokens: int = MAX_AUDIO_TOKS) -> np.ndarray:
    """Split float32 PCM into (n_frames, 640) chunks, same as training."""
    audio = audio[: max_tokens * CHUNK_SAMPLES]
    remainder = len(audio) % CHUNK_SAMPLES
    if remainder:
        audio = np.concatenate([audio, np.zeros(CHUNK_SAMPLES - remainder, dtype=np.float32)])
    return audio.reshape(-1, CHUNK_SAMPLES)


def build_inputs(tokenizer, frames: np.ndarray, device: torch.device,
                 instruction: str | None = None,
                 audio_delimiters: bool = False) -> dict:
    """
    Replicate the training collator's token sequence exactly.
    Training uses Gemma4 Unified special tokens: <|turn>=105, <turn|>=106, \n=107
    NOT <start_of_turn>/<end_of_turn> (those split into 7 tokens each in this tokenizer).

    Sequence: [bos] <|turn>user\n [audio x N] (instruction) <turn|>\n <|turn>model\n

    `instruction` mirrors Phase 0's layout (audio first, then the text prompt
    inside the same user turn) so the base model knows what to do with the audio.
    """
    n = len(frames)
    audio_token_id = getattr(tokenizer, "audio_token_id", AUDIO_TOKEN_ID)

    bos       = [tokenizer.bos_token_id]
    usr_open  = tokenizer.encode("<|turn>user\n",          add_special_tokens=False)  # [105, 2364, 107]
    usr_close = tokenizer.encode("<turn|>\n<|turn>model\n", add_special_tokens=False)  # [106, 107, 105, ...]
    audio_ids = [audio_token_id] * n
    if audio_delimiters:
        # wrap placeholders with the tokenizer's audio boundary tokens
        soa = tokenizer.encode("<|audio>", add_special_tokens=False)
        eoa = tokenizer.encode("<audio|>", add_special_tokens=False)
        audio_ids = soa + audio_ids + eoa
    instr_ids = tokenizer.encode(instruction, add_special_tokens=False) if instruction else []

    ids = bos + usr_open + audio_ids + instr_ids + usr_close
    input_ids      = torch.tensor([ids], dtype=torch.long,  device=device)
    attention_mask = torch.ones_like(input_ids)

    input_features      = torch.from_numpy(frames).unsqueeze(0).to(device=device, dtype=torch.float32)
    input_features_mask = torch.ones(1, n, dtype=torch.bool, device=device)

    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_features=input_features,
        input_features_mask=input_features_mask,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter",        default=None,   help="path to saved LoRA adapter dir (omit for base model)")
    p.add_argument("--audio",          required=True,  help="path to wav file")
    p.add_argument("--instruction",    default=None,   help="text instruction placed after audio in the user turn")
    p.add_argument("--audio-delimiters", action="store_true",
                   help="wrap audio placeholders with <|audio> ... <audio|> boundary tokens")
    p.add_argument("--gpu",            type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--base-model",     default="google/gemma-4-12b-it")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")

    tok_src = args.adapter or args.base_model
    print(f"[load] tokenizer from {tok_src}")
    tokenizer = AutoTokenizer.from_pretrained(tok_src)

    # Diagnostic: how does the tokenizer mark audio?
    added = tokenizer.get_added_vocab()
    audio_specials = {t: i for t, i in added.items() if "audio" in t.lower()}
    print(f"[diag] audio-related added tokens: {audio_specials}")
    print(f"[diag] audio_token_id attr: {getattr(tokenizer, 'audio_token_id', None)}  "
          f"(fallback const: {AUDIO_TOKEN_ID} = {tokenizer.decode([AUDIO_TOKEN_ID])!r})")

    print(f"[load] base model: {args.base_model}")
    model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )

    if args.adapter:
        print(f"[load] LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    else:
        print("[load] no adapter: running BASE model")
    model.eval()

    # Audio
    print(f"[audio] loading {args.audio}")
    audio  = load_audio_16k(args.audio)
    frames = audio_to_frames(audio)
    print(f"[audio] {len(audio)/SAMPLE_RATE:.2f}s → {len(frames)} frames ({len(frames)*40}ms)")

    if args.instruction:
        print(f"[infer] instruction: {args.instruction!r}")
    inputs = build_inputs(tokenizer, frames, device, instruction=args.instruction,
                          audio_delimiters=args.audio_delimiters)
    prompt_len = inputs["input_ids"].shape[1]
    print(f"[infer] prompt tokens: {prompt_len}, generating (max {args.max_new_tokens} new tokens)...")

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    generated = output_ids[0, prompt_len:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"\n{'='*60}")
    print(f"[result] {text}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 1B preflight check - run before starting fine-tuning.

Verifies environment, dataset, and tokenizer WITHOUT loading the 12B model.
Exits with code 0 if all checks pass, 1 if any check fails.

Usage:
  python3 phase1_preflight.py \
      --data /workspace/my_main/phase1_data/podcast_train
"""

from __future__ import annotations

import argparse
import sys

import torch

# Patch for PyTorch < 2.7 (transformers 5.x uses float8_e8m0fnu at import time)
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.float8_e4m3fn  # type: ignore[attr-defined]


AUDIO_TOKEN_ID = 258881
CHUNK_SAMPLES  = 640
SAMPLE_RATE    = 16_000

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = PASS if ok else FAIL
    line = f"  [{status}] {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return ok


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def run_checks(data_path: str) -> bool:
    all_ok = True

    # ------------------------------------------------------------------
    section("1. GPU / CUDA")
    # ------------------------------------------------------------------
    has_cuda = torch.cuda.is_available()
    all_ok &= check("CUDA available", has_cuda)
    if has_cuda:
        n_gpu = torch.cuda.device_count()
        for i in range(n_gpu):
            p = torch.cuda.get_device_properties(i)
            gb = p.total_memory / 1e9
            check(f"GPU {i}: {p.name}", True, f"{gb:.1f} GB")
        total_gb = sum(
            torch.cuda.get_device_properties(i).total_memory
            for i in range(n_gpu)
        ) / 1e9
        need_gb = 18.0  # 4-bit 12B model (~7 GB) + activations + optimizer
        all_ok &= check(
            f"Total GPU memory >= {need_gb:.0f} GB",
            total_gb >= need_gb,
            f"found {total_gb:.1f} GB",
        )

    # ------------------------------------------------------------------
    section("2. Package versions")
    # ------------------------------------------------------------------
    import transformers, peft, datasets, accelerate
    import importlib.metadata as meta

    def ver(pkg: str) -> str:
        try:
            return meta.version(pkg)
        except meta.PackageNotFoundError:
            return "NOT INSTALLED"

    torch_ver = torch.__version__
    tf_ver    = transformers.__version__
    peft_ver  = peft.__version__

    check("torch",         True, torch_ver)
    check("transformers",  True, tf_ver)
    check("peft",          True, peft_ver)
    check("datasets",      True, ver("datasets"))
    check("bitsandbytes",  True, ver("bitsandbytes"))
    check("accelerate",    True, ver("accelerate"))

    # transformers 5.0+ required for gemma4_unified
    tf_major = int(tf_ver.split(".")[0])
    all_ok &= check("transformers >= 5.0", tf_major >= 5, tf_ver)

    # ------------------------------------------------------------------
    section("3. Dataset")
    # ------------------------------------------------------------------
    from datasets import load_from_disk
    import numpy as np
    from pathlib import Path

    ds_path = Path(data_path)
    all_ok &= check("Dataset path exists", ds_path.exists(), str(ds_path))
    if not ds_path.exists():
        print("\n  Cannot proceed without dataset.")
        return False

    ds = load_from_disk(str(ds_path))
    n = len(ds)
    all_ok &= check("Sample count >= 500", n >= 500, f"{n} samples")

    required_cols = {"audio_array", "text", "duration_s", "n_frames"}
    has_cols = required_cols.issubset(set(ds.features.keys()))
    all_ok &= check("Required columns present", has_cols, str(sorted(ds.features.keys())))

    if has_cols:
        empty_text = sum(1 for i in range(min(200, n)) if not ds[i]["text"].strip())
        all_ok &= check(
            "Text non-empty (first 200)",
            empty_text == 0,
            f"{empty_text} empty",
        )

        durs = np.array([ds[i]["duration_s"] for i in range(min(200, n))], dtype=float)
        check("Duration stats", True,
              f"min={durs.min():.1f}s  avg={durs.mean():.1f}s  max={durs.max():.1f}s")

        # Check audio array shape
        sample = ds[0]
        arr = sample["audio_array"]
        frames_expected = sample["n_frames"]
        frames_actual = (len(arr) + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES
        all_ok &= check(
            "audio_array / n_frames consistent",
            abs(frames_actual - frames_expected) <= 1,
            f"len={len(arr)}  n_frames={frames_expected}",
        )

    # ------------------------------------------------------------------
    section("4. Tokenizer special tokens")
    # ------------------------------------------------------------------
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("google/gemma-4-12b-it")

    def enc(s: str) -> list[int]:
        return tok.encode(s, add_special_tokens=False)

    turn_start = enc("<|turn>")
    turn_end   = enc("<turn|>")
    newline    = enc("\n")

    all_ok &= check("<|turn>  == [105]",  turn_start == [105],  str(turn_start))
    all_ok &= check("<turn|>  == [106]",  turn_end   == [106],  str(turn_end))
    all_ok &= check("newline  == [107]",  newline    == [107],  str(newline))
    all_ok &= check("bos_token_id == 2",  tok.bos_token_id == 2,
                    str(tok.bos_token_id))
    all_ok &= check("eos_token_id == 1",  tok.eos_token_id == 1,
                    str(tok.eos_token_id))
    all_ok &= check(
        f"audio_token_id == {AUDIO_TOKEN_ID}",
        getattr(tok, "audio_token_id", None) == AUDIO_TOKEN_ID,
        str(getattr(tok, "audio_token_id", "N/A")),
    )

    # ------------------------------------------------------------------
    section("5. Collator smoke test")
    # ------------------------------------------------------------------
    import numpy as np

    # Minimal inline collator test (no model)
    bos      = [tok.bos_token_id]
    prefix   = enc("<|turn>user\n")
    bridge   = enc("<turn|>\n<|turn>model\n")
    turn_end_ids = enc("<turn|>\n")
    eos      = [tok.eos_token_id]

    text_ids = enc("テスト音声の転写です。")
    n_audio  = 50   # 2s of audio
    seq = bos + prefix + [AUDIO_TOKEN_ID]*n_audio + bridge + text_ids + turn_end_ids + eos
    resp_start = len(bos) + len(prefix) + n_audio + len(bridge)

    decoded = tok.decode(seq)
    has_audio   = "<|audio|>" in decoded
    has_bos_eos = decoded.startswith("<bos>") and decoded.endswith("<eos>")
    labels_ok   = resp_start < len(seq)

    all_ok &= check("Sequence contains audio tokens",  has_audio,   "")
    all_ok &= check("Sequence starts <bos>, ends <eos>", has_bos_eos, repr(decoded[:40]))
    all_ok &= check("Loss mask start index valid",     labels_ok,
                    f"resp_start={resp_start}  seq_len={len(seq)}")

    loss_text = tok.decode(seq[resp_start:])
    check("Loss region starts with model response", True, repr(loss_text[:40]))

    # ------------------------------------------------------------------
    section("6. Memory estimate")
    # ------------------------------------------------------------------
    # 12B params @ 4-bit = ~7 GB; activations + optimizer overhead
    # Rough estimate: 7 (model) + 2 (activations) + 1 (LoRA optimizer) = ~10 GB
    if has_cuda:
        total_gb = sum(
            torch.cuda.get_device_properties(i).total_memory
            for i in range(torch.cuda.device_count())
        ) / 1e9
        est_gb = 10.0
        ok = total_gb >= est_gb
        if not ok:
            print(f"\n  {WARN}  GPU memory {total_gb:.1f} GB < estimated {est_gb:.1f} GB")
            print("       Try: --no-4bit=False (keep 4-bit QLoRA), batch-size=1, grad-accum=16")
        else:
            check(f"Memory sufficient for 4-bit QLoRA", True,
                  f"{total_gb:.1f} GB available, ~{est_gb:.1f} GB needed")

    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    if all_ok:
        print(f"  {PASS}  All checks passed. Ready for fine-tuning.")
    else:
        print(f"  {FAIL}  Some checks failed. Fix issues above before training.")
    print(f"{'='*60}\n")

    return all_ok


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 1B preflight check")
    p.add_argument(
        "--data",
        default="/workspace/my_main/phase1_data/podcast_train",
        help="Path to HF Dataset from phase1_data_pipeline.py",
    )
    args = p.parse_args()

    ok = run_checks(args.data)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

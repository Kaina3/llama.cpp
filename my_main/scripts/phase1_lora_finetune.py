#!/usr/bin/env python3
"""
Phase 1B: LoRA fine-tuning for Gemma4 UA audio transcription (J-CHAT)

Verified model structure (google/gemma-4-12b-it, Gemma4UnifiedForConditionalGeneration):

  model.audio_tower              - Gemma4UnifiedAudioModel (num_layers=0 in unified)
  model.embed_audio              - Gemma4MultimodalEmbedder
    .embedding_pre_projection_norm  - RMSNorm (no learned scale, eps=1e-6)
    .embedding_projection           - nn.Linear(640, 3840, bias=False)  <- TRAIN FULLY
  model.language_model           - Gemma4TextModel (48 layers)
    .layers[i].self_attn.q_proj  - nn.Linear (16 heads * 512 head_dim = 8192)
    .layers[i].self_attn.v_proj  - nn.Linear (exists on sliding layers, None on full)
  lm_head                        - tied to language_model.embed_tokens.weight

  GGUF counterpart: mm.a.input_projection.weight  shape=(3840, 640)

Audio pipeline (unified):
  raw PCM (float32, 16kHz) -> split into 640-sample frames
  -> audio_tower (subsample bypass + output_proj) -> (n_frames, 640)
  -> embed_audio (RMSNorm + embedding_projection) -> (n_frames, 3840)
  -> injected at audio_token_id positions in token sequence

Training format:
  <bos><start_of_turn>user\n<audio_tok x N><end_of_turn>\n
  <start_of_turn>model\n{text}<end_of_turn><eos>

  Loss: only on text tokens in model turn (audio + user turn masked to -100)

Usage:
  # Inspect model structure (no training)
  python3 phase1_lora_finetune.py --model google/gemma-4-12b-it --inspect --gpu 0

  # Small overfit test (500 samples)
  python3 phase1_lora_finetune.py \\
      --model   google/gemma-4-12b-it \\
      --data    /workspace/llama.cpp/my_main/phase1_data/podcast_train \\
      --output  /workspace/llama.cpp/my_main/phase1_lora \\
      --gpu     0 \\
      --samples 500 \\
      --epochs  3

  # Full run (5000 samples, multi-GPU)
  python3 phase1_lora_finetune.py \\
      --model   google/gemma-4-12b-it \\
      --data    /workspace/llama.cpp/my_main/phase1_data/podcast_train \\
      --output  /workspace/llama.cpp/my_main/phase1_lora \\
      --gpu     0,1 \\
      --epochs  3

Docker notes:
  - Set HF_TOKEN env var for model download (HF gated model)
  - Mount workspace: -v /path/to/llama.cpp:/workspace/llama.cpp
  - GPU: --gpus '"device=0,1"' (docker flag) + --gpu 0,1 (this script flag)
  - Cache dir: HF_HOME=/workspace/.cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE    = 16_000
CHUNK_SAMPLES  = 640        # 40ms @ 16kHz
MAX_AUDIO_TOKS = 750        # 30s max = 480000 samples / 640
AUDIO_TOKEN_ID = 258881     # from Gemma4 config.json audio_token_id
N_DECODER_LAYERS = 48       # Gemma4 12B text decoder layers


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1B: Gemma4 UA LoRA fine-tuning")

    # Required
    p.add_argument("--model", default="google/gemma-4-12b-it",
                   help="HF model ID or local path (default: google/gemma-4-12b-it)")
    p.add_argument("--data", default=None,
                   help="Path to HF Dataset saved by phase1_data_pipeline.py")
    p.add_argument("--output", default=None,
                   help="Output directory for LoRA adapter")

    # GPU
    p.add_argument("--gpu", default="0",
                   help="GPU device(s), comma-separated (e.g. 0  or  0,1). "
                        "Sets CUDA_VISIBLE_DEVICES before loading anything.")

    # LoRA hyperparams
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--finetune-decoder-layers", type=int, default=4,
                   help="Number of final decoder layers to apply LoRA to")

    # Training hyperparams
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1,
                   help="Per-device batch size (keep 1 for 24GB GPU)")
    p.add_argument("--grad-accum", type=int, default=16,
                   help="Gradient accumulation steps (effective batch = batch*accum)")
    p.add_argument("--lr-audio", type=float, default=1e-5,
                   help="Learning rate for embed_audio.embedding_projection")
    p.add_argument("--lr-lora", type=float, default=2e-6,
                   help="Learning rate for decoder LoRA weights")
    p.add_argument("--max-audio-tokens", type=int, default=MAX_AUDIO_TOKS,
                   help="Max audio tokens per sample (truncates long audio)")
    p.add_argument("--samples", type=int, default=None,
                   help="Limit dataset to first N samples (default: all)")

    # Precision
    p.add_argument("--no-4bit", action="store_true",
                   help="Disable 4-bit quantization (uses bfloat16 instead)")
    p.add_argument("--fp16", action="store_true",
                   help="Use fp16 instead of bf16")

    # Utility
    p.add_argument("--inspect", action="store_true",
                   help="Print model structure then exit (no training)")
    p.add_argument("--resume", default=None,
                   help="Path to a checkpoint to resume from")
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--log-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ---------------------------------------------------------------------------
# GPU setup
# ---------------------------------------------------------------------------

def setup_gpu(gpu_str: str) -> None:
    """Set CUDA_VISIBLE_DEVICES based on --gpu flag before importing torch."""
    if gpu_str:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        print(f"[gpu] CUDA_VISIBLE_DEVICES={gpu_str}")
    n = torch.cuda.device_count()
    if n == 0:
        print("[gpu] WARNING: no CUDA devices visible; running on CPU")
    else:
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            gb = props.total_memory / 1e9
            print(f"[gpu] device {i}: {props.name}  ({gb:.1f} GB)")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_processor(args: argparse.Namespace):
    from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"\n[model] Loading processor from {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False)

    compute_dtype = torch.float16 if args.fp16 else torch.bfloat16

    if args.no_4bit:
        print(f"[model] Loading model in {compute_dtype} (no quantization)")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=compute_dtype,
            device_map="auto",
            trust_remote_code=False,
        )
    else:
        print("[model] Loading model in 4-bit QLoRA mode")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=False,
        )

    print(f"[model] Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B total")
    return model, processor


# ---------------------------------------------------------------------------
# Inspect model structure
# ---------------------------------------------------------------------------

def inspect_model(model, processor) -> None:
    """Print all named modules with their types and parameter counts."""
    print("\n" + "=" * 70)
    print("MODEL STRUCTURE INSPECTION")
    print("=" * 70)

    # Named modules
    audio_related = []
    decoder_layers = []
    other = []

    for name, module in model.named_modules():
        n_params = sum(p.numel() for p in module.parameters(recurse=False))
        if n_params == 0 and not any(True for _ in module.children()):
            continue  # skip empty containers with no direct params

        entry = f"  {name:<70s} {type(module).__name__:<30s} {n_params:>10,}"
        if "audio" in name or "embed_audio" in name:
            audio_related.append(entry)
        elif "language_model.layers" in name and "self_attn" in name:
            decoder_layers.append(entry)
        elif n_params > 0:
            other.append(entry)

    print("\n[Audio-related modules]")
    for e in audio_related:
        print(e)

    print("\n[Decoder self-attention modules (first 2 and last 6 layers)]")
    shown = set()
    for e in decoder_layers:
        # show first 2 layers and last 6 layers
        import re
        m = re.search(r"layers\.(\d+)\.", e)
        if m:
            idx = int(m.group(1))
            if idx < 2 or idx >= N_DECODER_LAYERS - 6:
                if idx not in shown:
                    shown.add(idx)
                    print(f"\n  [layer {idx}]")
                print(e)

    print("\n[Key training targets]")
    for name, module in model.named_modules():
        if name in ("model.embed_audio.embedding_projection",
                    "model.embed_audio.embedding_pre_projection_norm"):
            n_params = sum(p.numel() for p in module.parameters())
            print(f"  {name}  ->  {type(module).__name__}  ({n_params:,} params)")

    # Check audio projection weight exists and its shape
    print("\n[Audio projection weight check]")
    for name, param in model.named_parameters():
        if "embed_audio.embedding_projection" in name:
            print(f"  {name}  shape={tuple(param.shape)}  dtype={param.dtype}")

    # Tokenizer info
    print("\n[Tokenizer special tokens]")
    tok = processor.tokenizer
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        print(f"  {attr}: {getattr(tok, attr, 'N/A')}")
    audio_id = getattr(processor.tokenizer, "audio_token_id",
                       getattr(model.config, "audio_token_id", AUDIO_TOKEN_ID))
    print(f"  audio_token_id: {audio_id}")

    # Test chat template encoding
    print("\n[Chat template test (text only)]")
    dummy_text = [
        {"role": "user", "content": "こんにちは"},
        {"role": "model", "content": "こんにちは、元気ですか？"},
    ]
    try:
        encoded = tok.apply_chat_template(dummy_text, tokenize=True,
                                          add_generation_prompt=False)
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded
        print(f"  token count: {len(ids)}")
        for i, tid in enumerate(ids[:20]):
            print(f"    [{i:2d}] id={tid:8d}  {repr(tok.decode([tid]))}")
    except Exception as exc:
        print(f"  WARNING: chat template test failed: {exc}")

    # コレーター用テンプレートトークン確認
    print("\n[Collator template tokens]")
    for label, s in [
        ("turn_start",      "<|turn>"),
        ("turn_end",        "<turn|>"),
        ("user_role",       "user"),
        ("model_role",      "model"),
        ("newline",         "\n"),
        ("audio_placeholder", tok.decode([audio_id])),
    ]:
        ids_t = tok.encode(s, add_special_tokens=False)
        print(f"  {label:20s}: {ids_t}  ({repr(s)})")

    print("\n" + "=" * 70)
    print("Inspection done. Re-run without --inspect to start training.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------

def apply_lora(model, args: argparse.Namespace):
    """Apply PEFT LoRA to decoder layers and keep embed_audio unfrozen."""
    from peft import LoraConfig, get_peft_model

    # Decoder layer indices to apply LoRA to (last N layers)
    start = N_DECODER_LAYERS - args.finetune_decoder_layers
    lora_layers = list(range(start, N_DECODER_LAYERS))
    print(f"\n[lora] Applying LoRA to decoder layers: {lora_layers}")
    print(f"[lora] rank={args.lora_rank}  alpha={args.lora_alpha}  dropout={args.lora_dropout}")

    # Note: full_attention layers (indices 5,11,...,47) have no v_proj
    # (attention_k_eq_v=True => v_proj=None for full layers). PEFT skips
    # modules that don't exist, so listing "v_proj" here is safe.
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        layers_to_transform=lora_layers,
        # embed_audio.embedding_projection is kept trainable via modules_to_save
        modules_to_save=["embed_audio.embedding_projection"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Audio preprocessing helpers
# ---------------------------------------------------------------------------

def audio_to_frames(audio_array: np.ndarray, max_tokens: int) -> np.ndarray:
    """
    Split raw float32 PCM (16kHz) into (n_tokens, 640) chunks.
    Truncates to max_tokens frames. Pads last partial frame with zeros.
    Returns: float32 array of shape (n_tokens, 640)
    """
    total_samples = len(audio_array)
    max_samples = max_tokens * CHUNK_SAMPLES

    if total_samples > max_samples:
        audio_array = audio_array[:max_samples]
        total_samples = max_samples

    # Pad to a multiple of CHUNK_SAMPLES
    remainder = total_samples % CHUNK_SAMPLES
    if remainder != 0:
        pad = CHUNK_SAMPLES - remainder
        audio_array = np.concatenate([audio_array, np.zeros(pad, dtype=np.float32)])

    n_tokens = len(audio_array) // CHUNK_SAMPLES
    return audio_array.reshape(n_tokens, CHUNK_SAMPLES).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset and collator
# ---------------------------------------------------------------------------

class Phase1Dataset(torch.utils.data.Dataset):
    """Wraps the HF Dataset produced by phase1_data_pipeline.py."""

    def __init__(self, hf_dataset, max_tokens: int, limit: int | None = None):
        if limit is not None:
            hf_dataset = hf_dataset.select(range(min(limit, len(hf_dataset))))
        self.ds = hf_dataset
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx) -> dict[str, Any]:
        sample = self.ds[idx]
        audio_array = np.asarray(sample["audio_array"], dtype=np.float32)
        frames = audio_to_frames(audio_array, self.max_tokens)
        return {
            "frames":  frames,            # (n_tokens, 640)
            "n_tokens": frames.shape[0],
            "text": sample["text"],
        }


class AudioTextCollator:
    """
    Collates (frames, text) into Gemma4 Unified model inputs with proper loss masking.

    Builds:
      input_ids         (B, seq_len)       - text tokens + audio placeholders
      attention_mask    (B, seq_len)       - 1=valid
      input_features    (B, max_n_tok, 640) - raw audio frames (padded)
      input_features_mask (B, max_n_tok)   - 1=valid audio frame
      labels            (B, seq_len)       - -100 except model response tokens

    Sequence layout (Gemma4 Unified uses <|turn>/<turn|> tokens):
      [bos][<|turn>][user][\n][<|audio|> x N][<turn|>][\n]
      [<|turn>][model][\n][text_tokens][<turn|>][\n][eos]

    Loss is computed only on: text_tokens + <turn|> + \n + eos
    """

    def __init__(self, tokenizer, audio_token_id: int):
        self.tok = tokenizer
        self.audio_token_id = audio_token_id
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        # Pre-encode template fragments using Gemma4 Unified special tokens
        # <|turn> = 105, <turn|> = 106, \n = 107
        self._prefix_ids = self._encode_no_special("<|turn>user\n")
        self._bridge_ids = self._encode_no_special("<turn|>\n<|turn>model\n")
        self._turn_end_ids = self._encode_no_special("<turn|>\n")

        eos = tokenizer.eos_token_id
        self._eos_ids = [eos] if isinstance(eos, int) else list(eos)[:1]

        bos = tokenizer.bos_token_id
        self._bos_ids = [bos] if bos is not None else []

        # Verify template tokens are correct
        turn_start = self._encode_no_special("<|turn>")
        assert turn_start == [105], f"Expected <|turn>==[105], got {turn_start}"
        turn_end = self._encode_no_special("<turn|>")
        assert turn_end == [106], f"Expected <turn|>==[106], got {turn_end}"

    def _encode_no_special(self, text: str) -> list[int]:
        """Tokenize text without adding BOS/EOS."""
        return self.tok.encode(text, add_special_tokens=False)

    def _build_sequence(self, n_audio_tokens: int, text: str
                        ) -> tuple[list[int], int]:
        """
        Returns (full_sequence_ids, model_response_start_idx).
        model_response_start_idx points to the first token of the model response.
        """
        audio_ids = [self.audio_token_id] * n_audio_tokens
        text_ids  = self._encode_no_special(text)

        seq = (
            self._bos_ids
            + self._prefix_ids
            + audio_ids
            + self._bridge_ids
            + text_ids
            + self._turn_end_ids
            + self._eos_ids
        )

        # model response starts after bridge
        resp_start = (len(self._bos_ids)
                      + len(self._prefix_ids)
                      + n_audio_tokens
                      + len(self._bridge_ids))

        return seq, resp_start

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        seqs, resp_starts = [], []
        for item in batch:
            s, r = self._build_sequence(item["n_tokens"], item["text"])
            seqs.append(s)
            resp_starts.append(r)

        # Pad sequences (right-pad)
        max_seq = max(len(s) for s in seqs)
        input_ids  = torch.full((len(batch), max_seq), self.pad_id, dtype=torch.long)
        attn_mask  = torch.zeros(len(batch), max_seq, dtype=torch.long)
        labels     = torch.full((len(batch), max_seq), -100, dtype=torch.long)

        for i, (seq, r) in enumerate(zip(seqs, resp_starts)):
            n = len(seq)
            input_ids[i, :n] = torch.tensor(seq, dtype=torch.long)
            attn_mask[i, :n] = 1
            # Loss only on model response portion (index r onwards, inclusive)
            labels[i, r:n] = torch.tensor(seq[r:], dtype=torch.long)

        # Pad audio frames
        max_n_tok = max(item["n_tokens"] for item in batch)
        input_features = torch.zeros(len(batch), max_n_tok, CHUNK_SAMPLES, dtype=torch.float32)
        input_features_mask = torch.zeros(len(batch), max_n_tok, dtype=torch.bool)

        for i, item in enumerate(batch):
            n = item["n_tokens"]
            input_features[i, :n] = torch.from_numpy(item["frames"])
            input_features_mask[i, :n] = True

        return {
            "input_ids":           input_ids,
            "attention_mask":      attn_mask,
            "input_features":      input_features,
            "input_features_mask": input_features_mask,
            "labels":              labels,
        }


# ---------------------------------------------------------------------------
# Optimizer with per-group learning rates
# ---------------------------------------------------------------------------

def build_optimizer(model, args: argparse.Namespace) -> torch.optim.Optimizer:
    """
    Two parameter groups:
      - embed_audio.embedding_projection  : lr = args.lr_audio
      - LoRA weights (lora_A, lora_B)     : lr = args.lr_lora
    All other trainable params: lr = args.lr_lora (fallback)
    """
    audio_proj_params = []
    lora_params       = []
    other_params      = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "embedding_projection" in name and "embed_audio" in name:
            audio_proj_params.append(param)
        elif "lora_" in name:
            lora_params.append(param)
        else:
            other_params.append(param)

    n_audio = sum(p.numel() for p in audio_proj_params)
    n_lora  = sum(p.numel() for p in lora_params)
    n_other = sum(p.numel() for p in other_params)
    print(f"\n[optimizer] audio_proj: {n_audio:,} params  lr={args.lr_audio}")
    print(f"[optimizer] lora:       {n_lora:,} params  lr={args.lr_lora}")
    if n_other:
        print(f"[optimizer] other:      {n_other:,} params  lr={args.lr_lora} (fallback)")

    param_groups = []
    if audio_proj_params:
        param_groups.append({"params": audio_proj_params, "lr": args.lr_audio})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": args.lr_lora})
    if other_params:
        param_groups.append({"params": other_params, "lr": args.lr_lora})

    # Use 8-bit Adam if bitsandbytes is available (saves ~75% optimizer memory)
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(param_groups, betas=(0.9, 0.999), eps=1e-8)
        print("[optimizer] using AdamW8bit (bitsandbytes)")
    except ImportError:
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
        print("[optimizer] using AdamW (bitsandbytes not available)")

    return optimizer


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_loop(model, dataloader, optimizer, scheduler,
               args: argparse.Namespace, output_dir: Path) -> None:
    """Simple custom training loop with gradient accumulation."""
    device = next(model.parameters()).device
    model.train()

    total_steps = len(dataloader) * args.epochs
    accum_steps = 0
    global_step = 0
    running_loss = 0.0

    print(f"\n[train] Starting training: {args.epochs} epochs, "
          f"{len(dataloader)} steps/epoch, "
          f"effective batch={args.batch_size * args.grad_accum}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            running_loss += loss.item() * args.grad_accum
            accum_steps += 1

            if accum_steps == args.grad_accum:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                accum_steps = 0
                global_step += 1

                if global_step % args.log_steps == 0:
                    avg_loss = running_loss / (args.log_steps * args.grad_accum)
                    lr_lora  = optimizer.param_groups[-1]["lr"]
                    print(f"  epoch={epoch+1} step={global_step}/{total_steps // args.grad_accum}"
                          f"  loss={avg_loss:.4f}  lr={lr_lora:.2e}")
                    running_loss = 0.0

                if global_step % args.save_steps == 0:
                    ckpt = output_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(str(ckpt))
                    print(f"  [save] checkpoint saved to {ckpt}")

    # Final save
    print(f"\n[train] Saving final adapter to {output_dir}")
    model.save_pretrained(str(output_dir))
    print("[train] Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Set CUDA_VISIBLE_DEVICES BEFORE importing torch (already imported above
    # for type hints, so we patch here; in Docker the env var matters more)
    setup_gpu(args.gpu)

    torch.manual_seed(args.seed)

    # --inspect mode: load model and print structure, then exit
    if args.inspect:
        model, processor = load_model_and_processor(args)
        inspect_model(model, processor)
        return

    # Validate required args for training
    if args.data is None or args.output is None:
        print("ERROR: --data and --output are required for training.")
        print("       Use --inspect to examine model structure without training.")
        sys.exit(1)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: dataset not found at {data_path}")
        sys.exit(1)

    output_dir = Path(args.output)

    # Load dataset
    print(f"\n[data] Loading dataset from {data_path}")
    from datasets import load_from_disk
    raw_ds = load_from_disk(str(data_path))
    print(f"[data] {len(raw_ds)} samples available")

    # Load model + processor
    model, processor = load_model_and_processor(args)
    tokenizer = processor.tokenizer

    # Resolve audio_token_id from model config (fallback to hardcoded)
    audio_token_id = getattr(model.config, "audio_token_id", AUDIO_TOKEN_ID)
    print(f"[model] audio_token_id = {audio_token_id}")

    # Apply LoRA
    model = apply_lora(model, args)

    # Enable gradient checkpointing to reduce activation memory
    model.gradient_checkpointing_enable()

    # Dataset + collator
    dataset  = Phase1Dataset(raw_ds, max_tokens=args.max_audio_tokens,
                             limit=args.samples)
    collator = AudioTextCollator(tokenizer, audio_token_id)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,     # 0 to avoid multiprocess issues in Docker
        pin_memory=True,
    )

    print(f"[data] Training on {len(dataset)} samples")

    # Optimizer + scheduler
    optimizer = build_optimizer(model, args)

    total_update_steps = (len(dataloader) * args.epochs) // args.grad_accum
    warmup_steps = min(100, total_update_steps // 10)
    from torch.optim.lr_scheduler import LinearLR
    scheduler = LinearLR(optimizer,
                         start_factor=0.1, end_factor=1.0,
                         total_iters=warmup_steps)

    # Resume from checkpoint
    if args.resume:
        print(f"\n[resume] Loading checkpoint from {args.resume}")
        from peft import PeftModel
        model.load_adapter(args.resume)

    # Train
    train_loop(model, dataloader, optimizer, scheduler, args, output_dir)

    # Save processor alongside adapter
    processor.save_pretrained(str(output_dir))
    print(f"[save] Processor saved to {output_dir}")


if __name__ == "__main__":
    main()

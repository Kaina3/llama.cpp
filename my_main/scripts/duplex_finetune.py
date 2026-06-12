#!/usr/bin/env python3
"""
Duplex Fine-tuning: 時刻整列ストリーミング形式での Gemma 4 Unified LoRA 学習

設計書: my_main/research/weekly/202606W1/0612/1000_full_duplex_redesign.md

phase1_lora_finetune.py からの主な変更点:
  - 学習フォーマットを時刻整列インターリーブに変更:
      [BOS] <|turn>{asr|chat}\n <|audio> [A_0][S_0][A_1][S_1]... <audio|> <turn|>\n
    A_t = 音声フレーム (audio placeholder)、S_t = テキストスロット (トークン or PAD)
  - embed_audio.embedding_projection は凍結（事前学習済みを保護）
  - LoRA を全 decoder 層の q/k/v/o + MLP に拡大
  - PAD スロットも loss 対象（「黙るべき時に黙る」を学習）
  - メトリクス (train/val の loss, loss_text, loss_pad, acc, gnorm, lr) を
    output/metrics.jsonl に逐次記録 -> plot_metrics.py で学習曲線を描画

Usage:
  python3 my_main/scripts/duplex_finetune.py \\
      --data   my_main/duplex_data/asr_train \\
      --output my_main/duplex_lora \\
      --gpu    1 \\
      --epochs 3

  # 学習曲線の描画 (学習中でも実行可)
  python3 my_main/scripts/plot_metrics.py my_main/duplex_lora/metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

# torch.float8_e8m0fnu was added in PyTorch 2.7; transformers 5.x uses it at
# import time. Patch for older torch BEFORE importing transformers/peft.
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.float8_e4m3fn  # type: ignore[attr-defined]

SAMPLE_RATE      = 16_000
CHUNK_SAMPLES    = 640
FRAME_S          = 0.04
AUDIO_TOKEN_ID   = 258881
N_DECODER_LAYERS = 48


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Duplex time-aligned LoRA fine-tuning")
    p.add_argument("--model",  default="google/gemma-4-12b-it")
    p.add_argument("--data",   required=True, help="duplex_data_pipeline.py の出力 dir")
    p.add_argument("--output", required=True, help="adapter / metrics の出力 dir")
    p.add_argument("--gpu",    default="0", help="CUDA_VISIBLE_DEVICES に設定")

    # LoRA
    p.add_argument("--lora-rank",    type=int,   default=16)
    p.add_argument("--lora-alpha",   type=int,   default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--attn-only", action="store_true",
                   help="LoRA を attention のみに限定 (デフォルトは MLP も含む)")

    # 学習
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--batch-size", type=int,   default=1)
    p.add_argument("--grad-accum", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=1e-4, help="LoRA の学習率")
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--samples",    type=int,   default=None, help="train サンプル数上限")
    p.add_argument("--seed",       type=int,   default=42)

    # フォーマット
    p.add_argument("--max-frames", type=int, default=750, help="最大音声フレーム数 (750=30s)")
    p.add_argument("--text-delay-frames", type=int, default=3,
                   help="トークン音声終了からテキスト配置までの遅延 [frames]")
    p.add_argument("--pad-token", default="<mask>",
                   help="テキストスロットの PAD に使う未使用トークン "
                        "(Gemma 4 vocab に <unusedN> は無いため <mask>=4 を転用)")
    p.add_argument("--pad-loss-weight", type=float, default=1.0,
                   help="PAD スロットの loss 重み (text スロットは常に 1.0)")

    # 評価・ログ
    p.add_argument("--val-samples", type=int, default=64)
    p.add_argument("--eval-steps",  type=int, default=50)
    p.add_argument("--save-steps",  type=int, default=200)
    p.add_argument("--log-steps",   type=int, default=10)

    # その他
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--resume",  default=None, help="adapter checkpoint dir から再開")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading & LoRA (phase1_lora_finetune.py のパターンを踏襲)
# ---------------------------------------------------------------------------

def load_model_and_processor(args):
    from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"[model] loading processor: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False)

    if args.no_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=False)
    else:
        print("[model] loading in 4-bit QLoRA mode")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb, device_map="auto",
            trust_remote_code=False)
    return model, processor


def dequantize_audio_projection_frozen(model) -> None:
    """
    embed_audio.embedding_projection を fp32 nn.Linear に置換する（凍結のまま）。
    目的は学習ではなく dtype 整合: prepare_model_for_kbit_training 後の
    masked_scatter_ (audio embeds -> inputs_embeds) で Float/BFloat16 不一致を防ぐ。
    """
    try:
        from bitsandbytes.nn import Linear4bit
    except ImportError:
        return
    target = model.model.embed_audio.embedding_projection
    if not isinstance(target, Linear4bit):
        print("[lora] audio projection not quantized; skip dequantize")
        return
    import bitsandbytes.functional as bnbF
    device = next(target.parameters()).device
    qs = target.weight.quant_state
    if qs is not None:
        w = bnbF.dequantize_4bit(target.weight.data, qs).to(torch.float32)
    else:
        w = target.weight.data.reshape(target.out_features, target.in_features).to(torch.float32)
    new = torch.nn.Linear(target.in_features, target.out_features,
                          bias=target.bias is not None,
                          dtype=torch.float32, device=device)
    new.weight = torch.nn.Parameter(w, requires_grad=False)
    if target.bias is not None:
        new.bias = torch.nn.Parameter(target.bias.data.to(torch.float32),
                                      requires_grad=False)
    model.model.embed_audio.embedding_projection = new
    print("[lora] audio projection: dequantized NF4->fp32, FROZEN")


def apply_lora(model, args):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if not args.no_4bit:
        dequantize_audio_projection_frozen(model)
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})

    # 全 decoder 層を対象 (language_model 配下のみ。vision/audio tower は除外)
    mods = "q_proj|k_proj|v_proj|o_proj" if args.attn_only \
        else "q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
    pattern = rf".*language_model.*\.({mods})$"
    print(f"[lora] target pattern: {pattern}")
    print(f"[lora] rank={args.lora_rank} alpha={args.lora_alpha} dropout={args.lora_dropout}")

    cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=pattern, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)

    # projection が誤って学習対象になっていないことを保証
    for name, param in model.named_parameters():
        if "embed_audio" in name and "lora_" not in name:
            param.requires_grad = False

    model.print_trainable_parameters()
    if args.no_4bit:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


# ---------------------------------------------------------------------------
# Dataset & time-aligned collator
# ---------------------------------------------------------------------------

def audio_to_frames(audio: np.ndarray, max_frames: int) -> np.ndarray:
    audio = audio[: max_frames * CHUNK_SAMPLES]
    rem = len(audio) % CHUNK_SAMPLES
    if rem:
        audio = np.concatenate([audio, np.zeros(CHUNK_SAMPLES - rem, dtype=np.float32)])
    return audio.reshape(-1, CHUNK_SAMPLES).astype(np.float32)


class DuplexDataset(torch.utils.data.Dataset):
    """duplex_data_pipeline.py の出力をラップする。"""

    def __init__(self, hf_dataset, max_frames: int, indices: list[int]):
        self.ds = hf_dataset
        self.max_frames = max_frames
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i) -> dict[str, Any]:
        s = self.ds[self.indices[i]]
        audio = np.asarray(s["audio_array"], dtype=np.float32)
        frames = audio_to_frames(audio, self.max_frames)
        limit_s = frames.shape[0] * FRAME_S
        # 音声を切り詰めた場合、範囲外のトークンは捨てる
        toks = [(txt, st, en) for txt, st, en
                in zip(s["token_texts"], s["token_starts"], s["token_ends"])
                if en <= limit_s]
        return {
            "frames": frames,
            "n_frames": frames.shape[0],
            "tokens": toks,
            "mode": s["mode"],
        }


class DuplexCollator:
    """
    時刻整列インターリーブ形式を組み立てる。

      [BOS] <|turn>{role}\n <|audio> [A_0][S_0][A_1][S_1]...[A_n][S_n] <audio|> <turn|>\n

    S_t はフレーム t 終了時点でモデルが発するべきトークン。
    トークンのテキストは「音声終了フレーム + delay」以降の空きスロットに
    FIFO で詰める。スロットが埋まらない時刻は PAD。
    Loss はスロット位置のみ (PAD 含む)。
    """

    def __init__(self, tokenizer, audio_token_id: int, pad_token: str,
                 delay_frames: int):
        self.tok = tokenizer
        self.audio_id = audio_token_id
        self.delay = delay_frames
        self.seq_pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
        pad_ids = enc(pad_token)
        assert len(pad_ids) == 1, f"--pad-token must be a single token, got {pad_ids}"
        self.slot_pad_id = pad_ids[0]

        soa, eoa = enc("<|audio>"), enc("<audio|>")
        assert len(soa) == 1 and len(eoa) == 1, f"audio delimiters not single tokens: {soa} {eoa}"
        self.soa_id, self.eoa_id = soa[0], eoa[0]

        bos = tokenizer.bos_token_id
        self._bos = [bos] if bos is not None else []
        self._prefix = {
            "asr":    enc("<|turn>asr\n"),
            "dialog": enc("<|turn>chat\n"),
        }
        self._closing = enc("<turn|>\n")
        self.overflow_tokens = 0  # スロット不足で落ちたトークン数 (累計)

    def _text_slots(self, tokens: list[tuple], n_frames: int) -> list[int]:
        events = []
        for txt, _st, en in tokens:
            ids = self.tok.encode(txt, add_special_tokens=False)
            if not ids:
                continue
            nat = int(en / FRAME_S)          # 音声終了フレーム (これより早くは置けない)
            if nat >= n_frames:
                self.overflow_tokens += len(ids)
                continue
            # delay 加算が末尾を超える場合は因果性を保てる範囲でクランプ
            f = min(nat + self.delay, n_frames - 1)
            events.append((f, ids))
        events.sort(key=lambda e: e[0])

        slots, queue, ei = [], deque(), 0
        for t in range(n_frames):
            while ei < len(events) and events[ei][0] <= t:
                queue.extend(events[ei][1])
                ei += 1
            slots.append(queue.popleft() if queue else self.slot_pad_id)
        self.overflow_tokens += len(queue) + sum(len(e[1]) for e in events[ei:])
        return slots

    def _build(self, item: dict) -> tuple[list[int], list[int]]:
        n = item["n_frames"]
        slots = self._text_slots(item["tokens"], n)
        prefix = self._bos + self._prefix[item["mode"]] + [self.soa_id]

        ids = list(prefix)
        labels = [-100] * len(prefix)
        for t in range(n):
            ids += [self.audio_id, slots[t]]
            labels += [-100, slots[t]]
        ids += [self.eoa_id] + self._closing
        labels += [-100] * (1 + len(self._closing))
        return ids, labels

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        built = [self._build(it) for it in batch]
        max_len = max(len(ids) for ids, _ in built)
        B = len(batch)

        input_ids = torch.full((B, max_len), self.seq_pad_id, dtype=torch.long)
        attn      = torch.zeros(B, max_len, dtype=torch.long)
        labels    = torch.full((B, max_len), -100, dtype=torch.long)
        for i, (ids, lab) in enumerate(built):
            input_ids[i, :len(ids)] = torch.tensor(ids)
            attn[i, :len(ids)] = 1
            labels[i, :len(lab)] = torch.tensor(lab)

        max_f = max(it["n_frames"] for it in batch)
        feats = torch.zeros(B, max_f, CHUNK_SAMPLES, dtype=torch.float32)
        fmask = torch.zeros(B, max_f, dtype=torch.bool)
        for i, it in enumerate(batch):
            feats[i, :it["n_frames"]] = torch.from_numpy(it["frames"])
            fmask[i, :it["n_frames"]] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "input_features": feats,
            "input_features_mask": fmask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Loss (PAD / text 分離メトリクス付き)
# ---------------------------------------------------------------------------

def compute_loss(logits: torch.Tensor, labels: torch.Tensor,
                 slot_pad_id: int, pad_weight: float) -> dict:
    """
    シフト済み CE を手計算し、PAD スロットと text スロットを分離して返す。
    戻り値の "loss" が backward 対象。
    """
    import torch.nn.functional as F

    pred = logits[:, :-1]
    tgt  = labels[:, 1:]
    mask = tgt != -100
    sel_logits = pred[mask].float()
    sel_tgt    = tgt[mask]

    ce = F.cross_entropy(sel_logits, sel_tgt, reduction="none")
    is_pad = sel_tgt == slot_pad_id
    n_pad, n_text = int(is_pad.sum()), int((~is_pad).sum())

    with torch.no_grad():
        correct = sel_logits.argmax(-1) == sel_tgt
        loss_pad  = float(ce[is_pad].mean())  if n_pad  else 0.0
        loss_text = float(ce[~is_pad].mean()) if n_text else 0.0
        acc_pad   = float(correct[is_pad].float().mean())  if n_pad  else 0.0
        acc_text  = float(correct[~is_pad].float().mean()) if n_text else 0.0

    w = torch.where(is_pad, pad_weight, 1.0)
    loss = (ce * w).sum() / w.sum().clamp(min=1.0)

    return {"loss": loss, "loss_pad": loss_pad, "loss_text": loss_text,
            "acc_pad": acc_pad, "acc_text": acc_text,
            "n_pad": n_pad, "n_text": n_text}


# ---------------------------------------------------------------------------
# Metrics logger
# ---------------------------------------------------------------------------

class MetricsLogger:
    """output/metrics.jsonl に 1 行 1 レコードで逐次追記する。"""

    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "a", buffering=1)

    def log(self, **rec) -> None:
        rec["ts"] = time.time()
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, slot_pad_id, pad_weight) -> dict:
    model.eval()
    agg = {"loss": 0.0, "loss_pad": 0.0, "loss_text": 0.0,
           "acc_pad": 0.0, "acc_text": 0.0}
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        out = model(**batch)
        m = compute_loss(out.logits, labels, slot_pad_id, pad_weight)
        agg["loss"] += float(m["loss"])
        for k in ("loss_pad", "loss_text", "acc_pad", "acc_text"):
            agg[k] += m[k]
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in agg.items()}


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def train(model, train_loader, val_loader, optimizer, scheduler, collator,
          args, output_dir: Path) -> None:
    device = next(model.parameters()).device
    metrics = MetricsLogger(output_dir / "metrics.jsonl")
    model.train()

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_updates = steps_per_epoch * args.epochs
    print(f"\n[train] {args.epochs} epochs x {len(train_loader)} batches "
          f"(grad_accum={args.grad_accum}, {total_updates} updates total)")

    global_step, accum, t0 = 0, 0, time.monotonic()
    run = {"loss": 0.0, "loss_pad": 0.0, "loss_text": 0.0,
           "acc_pad": 0.0, "acc_text": 0.0, "n": 0}

    for epoch in range(args.epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            out = model(**batch)
            m = compute_loss(out.logits, labels, collator.slot_pad_id,
                             args.pad_loss_weight)
            (m["loss"] / args.grad_accum).backward()

            run["loss"] += float(m["loss"]); run["n"] += 1
            for k in ("loss_pad", "loss_text", "acc_pad", "acc_text"):
                run[k] += m[k]
            accum += 1

            if accum < args.grad_accum:
                continue
            accum = 0

            gnorm = 0.0
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    gnorm += float(p.grad.norm()) ** 2
            gnorm = gnorm ** 0.5
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % args.log_steps == 0:
                n = max(run["n"], 1)
                lr = optimizer.param_groups[0]["lr"]
                print(f"  ep={epoch+1} step={global_step}/{total_updates}"
                      f"  loss={run['loss']/n:.4f}"
                      f"  text={run['loss_text']/n:.4f} pad={run['loss_pad']/n:.4f}"
                      f"  acc_t={run['acc_text']/n:.3f}"
                      f"  gnorm={gnorm:.2e} lr={lr:.2e}"
                      f"  ovf={collator.overflow_tokens}")
                metrics.log(split="train", epoch=epoch + 1, step=global_step,
                            loss=run["loss"]/n, loss_text=run["loss_text"]/n,
                            loss_pad=run["loss_pad"]/n,
                            acc_text=run["acc_text"]/n, acc_pad=run["acc_pad"]/n,
                            gnorm=gnorm, lr=lr,
                            overflow_tokens=collator.overflow_tokens,
                            elapsed_s=time.monotonic() - t0)
                run = {k: 0.0 for k in run}

            if val_loader and global_step % args.eval_steps == 0:
                ev = evaluate(model, val_loader, device,
                              collator.slot_pad_id, args.pad_loss_weight)
                print(f"  [val] step={global_step}  loss={ev['loss']:.4f}"
                      f"  text={ev['loss_text']:.4f} pad={ev['loss_pad']:.4f}"
                      f"  acc_t={ev['acc_text']:.3f} acc_p={ev['acc_pad']:.3f}")
                metrics.log(split="val", epoch=epoch + 1, step=global_step, **ev)

            if global_step % args.save_steps == 0:
                ckpt = output_dir / f"checkpoint-{global_step}"
                model.save_pretrained(str(ckpt))
                print(f"  [save] {ckpt}")

    print(f"\n[train] saving final adapter to {output_dir}")
    model.save_pretrained(str(output_dir))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"[gpu] CUDA_VISIBLE_DEVICES={args.gpu}")
    torch.manual_seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    from datasets import load_from_disk
    raw = load_from_disk(args.data)
    print(f"[data] {len(raw)} samples from {args.data}")

    # train/val split (seed 固定)
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(raw)).tolist()
    n_val = min(args.val_samples, len(raw) // 10)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    if args.samples:
        train_idx = train_idx[:args.samples]
    print(f"[data] train={len(train_idx)}  val={len(val_idx)}")

    model, processor = load_model_and_processor(args)
    tokenizer = processor.tokenizer
    audio_token_id = getattr(model.config, "audio_token_id", AUDIO_TOKEN_ID)
    model = apply_lora(model, args)

    if args.resume:
        print(f"[resume] loading adapter from {args.resume}")
        model.load_adapter(args.resume, adapter_name="default")

    collator = DuplexCollator(tokenizer, audio_token_id,
                              args.pad_token, args.text_delay_frames)
    print(f"[collator] slot_pad={collator.slot_pad_id} ({args.pad_token})"
          f"  soa={collator.soa_id} eoa={collator.eoa_id}"
          f"  delay={args.text_delay_frames} frames")

    mk_loader = lambda indices, shuffle: torch.utils.data.DataLoader(
        DuplexDataset(raw, args.max_frames, indices),
        batch_size=args.batch_size, shuffle=shuffle,
        collate_fn=collator, num_workers=0, pin_memory=True)
    train_loader = mk_loader(train_idx, True)
    val_loader   = mk_loader(val_idx, False) if val_idx else None

    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "lora_" in n]
    n_lora = sum(p.numel() for p in lora_params)
    print(f"[optimizer] lora params: {n_lora:,}  lr={args.lr:.2e}")
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(lora_params, lr=args.lr,
                                        betas=(0.9, 0.999), eps=1e-8)
        print("[optimizer] AdamW8bit")
    except ImportError:
        optimizer = torch.optim.AdamW(lora_params, lr=args.lr,
                                      betas=(0.9, 0.999), eps=1e-8)
        print("[optimizer] AdamW")

    total_updates = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=min(args.warmup_steps, total_updates // 10 + 1),
        num_training_steps=total_updates)

    train(model, train_loader, val_loader, optimizer, scheduler, collator,
          args, output_dir)

    processor.save_pretrained(str(output_dir))
    print(f"[save] processor saved to {output_dir}")
    print(f"[done] metrics: {output_dir / 'metrics.jsonl'}")


if __name__ == "__main__":
    main()

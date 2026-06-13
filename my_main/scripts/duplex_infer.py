#!/usr/bin/env python3
"""
Duplex inference test: teacher-forced evaluation + optional streaming mode.

学習フォーマット（DuplexCollator と同一）でシーケンスを組み立て、
テキストスロット位置の logit を読んで「いつ・何のトークンを出すか」を確認する。

2 つのモード:
  batch   : 音声全体を 1 回の forward で処理。スロット位置の argmax を表示。
            学習が効いているか確認する最初のテストはこれ。
  stream  : KV キャッシュを使い 1 フレーム × 1 forward で逐次処理。
            推論ループ設計書 §5 の実装。

Usage:
  # batch モード（デフォルト）
  python3 my_main/scripts/duplex_infer.py \\
      --adapter my_main/duplex_lora \\
      --audio   my_main/sample/gemma4_test_ja.wav \\
      --gpu 1

  # adapter なし（ベースモデル単体で確認）
  python3 my_main/scripts/duplex_infer.py \\
      --audio my_main/sample/gemma4_test_ja.wav \\
      --gpu 1

  # streaming モード
  python3 my_main/scripts/duplex_infer.py \\
      --adapter my_main/duplex_lora \\
      --audio   my_main/sample/gemma4_test_ja.wav \\
      --mode stream --gpu 1
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import soundfile as sf
from scipy import signal as scipy_signal

if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.float8_e4m3fn  # type: ignore[attr-defined]

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

SAMPLE_RATE   = 16_000
CHUNK_SAMPLES = 640
FRAME_S       = 0.04          # 40ms
AUDIO_TOKEN_ID = 258881
PAD_TOKEN     = "<mask>"      # 学習時と同じ PAD トークン


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_audio_16k(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = scipy_signal.resample(audio, int(len(audio) * SAMPLE_RATE / sr))
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95
    return audio.astype(np.float32)


def audio_to_frames(audio: np.ndarray, max_frames: int = 750) -> np.ndarray:
    audio = audio[: max_frames * CHUNK_SAMPLES]
    rem = len(audio) % CHUNK_SAMPLES
    if rem:
        audio = np.concatenate([audio, np.zeros(CHUNK_SAMPLES - rem, dtype=np.float32)])
    return audio.reshape(-1, CHUNK_SAMPLES).astype(np.float32)


# ---------------------------------------------------------------------------
# Sequence builder (DuplexCollator と同一フォーマット)
# ---------------------------------------------------------------------------

def build_duplex_inputs(tokenizer, frames: np.ndarray, device: torch.device,
                        pad_id: int, mode: str = "asr") -> dict:
    """
    [BOS] <|turn>{mode}\n <|audio> [A_0][PAD][A_1][PAD]...[A_n][PAD] <audio|> <turn|>\n

    テキストスロットはすべて PAD で埋める（推論時の初期状態）。
    slot_positions[t] = input_ids 内のスロット t のインデックス。
    """
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)

    bos     = [tokenizer.bos_token_id]
    prefix  = enc(f"<|turn>{mode}\n")
    soa     = enc("<|audio>")
    eoa     = enc("<audio|>")
    closing = enc("<turn|>\n")

    assert len(soa) == 1 and len(eoa) == 1, f"境界トークンが単一でない: soa={soa} eoa={eoa}"
    audio_id = getattr(tokenizer, "audio_token_id", AUDIO_TOKEN_ID)

    n = len(frames)
    head = bos + prefix + soa    # [BOS] <|turn>asr\n <|audio>
    body = []
    slot_positions = []
    for t in range(n):
        pos = len(head) + len(body) + 1   # +1 because audio token comes first
        body.append(audio_id)             # A_t
        body.append(pad_id)               # S_t (PAD placeholder)
        slot_positions.append(pos)        # position of S_t in full sequence

    tail = eoa + closing

    ids = head + body + tail
    input_ids   = torch.tensor([ids], dtype=torch.long,  device=device)
    attn        = torch.ones_like(input_ids)
    input_features      = torch.from_numpy(frames).unsqueeze(0).to(device=device, dtype=torch.float32)
    input_features_mask = torch.ones(1, n, dtype=torch.bool, device=device)

    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "input_features": input_features,
        "input_features_mask": input_features_mask,
        "slot_positions": slot_positions,   # 呼び出し元用（モデルには渡さない）
    }


# ---------------------------------------------------------------------------
# Batch mode: 1 forward pass でスロット全読み出し
# ---------------------------------------------------------------------------

@torch.inference_mode()
def infer_batch(model, tokenizer, frames: np.ndarray,
                pad_id: int, device: torch.device,
                top_k: int = 5, temperature: float = 1.0,
                mode: str = "asr") -> None:
    inp = build_duplex_inputs(tokenizer, frames, device, pad_id, mode)
    slot_positions = inp.pop("slot_positions")

    out = model(
        input_ids=inp["input_ids"],
        attention_mask=inp["attention_mask"],
        input_features=inp["input_features"],
        input_features_mask=inp["input_features_mask"],
    )
    logits = out.logits[0]   # (seq_len, vocab_size)

    print(f"\n{'='*60}")
    print(f"  batch mode:  {len(frames)} frames = {len(frames)*FRAME_S:.2f}s")
    print(f"  seq_len:     {inp['input_ids'].shape[1]}")
    print(f"  slot count:  {len(slot_positions)}")
    print(f"{'='*60}")
    print(f"  time(s)  pred_token          top-{top_k}")
    print(f"  {'—'*54}")

    non_pad = 0
    for t, pos in enumerate(slot_positions):
        if pos >= logits.shape[0]:
            break
        # logit at pos-1 predicts pos (shifted by 1 in causal LM)
        logit = logits[pos - 1]
        probs = torch.softmax(logit.float() / max(temperature, 1e-8), dim=-1)
        top = torch.topk(probs, top_k)
        pred_id = top.indices[0].item()
        pred_tok = tokenizer.decode([pred_id])
        ts = t * FRAME_S

        is_pad = pred_id == pad_id
        marker = "   " if is_pad else ">> "
        if not is_pad:
            non_pad += 1
        top_str = "  ".join(
            f"{tokenizer.decode([i.item()])!r}({v.item():.3f})"
            for i, v in zip(top.indices, top.values)
        )
        print(f"  {marker}{ts:5.2f}s  {pred_tok!r:20s}  {top_str}")

    print(f"{'='*60}")
    print(f"  非PADスロット: {non_pad} / {len(slot_positions)}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Stream mode: 1 frame × 1 forward（KV キャッシュ）
# ---------------------------------------------------------------------------

@torch.inference_mode()
def infer_stream(model, tokenizer, frames: np.ndarray,
                 pad_id: int, device: torch.device,
                 temperature: float = 0.8, mode: str = "asr") -> None:
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    bos    = [tokenizer.bos_token_id]
    prefix = enc(f"<|turn>{mode}\n")
    soa    = enc("<|audio>")
    eoa    = enc("<audio|>")
    closing = enc("<turn|>\n")
    audio_id = getattr(tokenizer, "audio_token_id", AUDIO_TOKEN_ID)

    n = len(frames)
    all_input_features      = torch.from_numpy(frames).unsqueeze(0).to(device=device, dtype=torch.float32)
    all_input_features_mask = torch.ones(1, n, dtype=torch.bool, device=device)

    print(f"\n{'='*60}")
    print(f"  stream mode: {n} frames = {n*FRAME_S:.2f}s")
    print(f"{'='*60}")

    # プレフィックスを一括 forward してキャッシュ確立
    head_ids = bos + prefix + soa
    input_ids = torch.tensor([head_ids], dtype=torch.long, device=device)
    attn      = torch.ones_like(input_ids)

    out = model(
        input_ids=input_ids,
        attention_mask=attn,
        input_features=all_input_features,
        input_features_mask=all_input_features_mask,
        use_cache=True,
    )
    past = out.past_key_values

    output_tokens = []
    for t in range(n):
        ts = t * FRAME_S

        # --- ステップ 1: 音声フレーム A_t を追加 ---
        a_ids = torch.tensor([[audio_id]], dtype=torch.long, device=device)
        a_attn = torch.ones(1, 1, dtype=torch.long, device=device)
        out = model(input_ids=a_ids, attention_mask=a_attn,
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        # 音声フレーム後の logit でテキストスロット S_t を予測
        logit = out.logits[0, -1]

        # --- ステップ 2: S_t をサンプル ---
        if temperature > 0:
            probs = torch.softmax(logit.float() / temperature, dim=-1)
            pred_id = torch.multinomial(probs, 1).item()
        else:
            pred_id = logit.argmax().item()

        pred_tok = tokenizer.decode([pred_id])
        is_pad = pred_id == pad_id

        # S_t を next input として追加
        s_ids = torch.tensor([[pred_id]], dtype=torch.long, device=device)
        s_attn = torch.ones(1, 1, dtype=torch.long, device=device)
        out = model(input_ids=s_ids, attention_mask=s_attn,
                    past_key_values=past, use_cache=True)
        past = out.past_key_values

        if not is_pad:
            output_tokens.append((ts, pred_tok))
            print(f"  >> {ts:5.2f}s  {pred_tok!r}")

    print(f"\n  --- 生成テキスト ---")
    text = "".join(tok for _, tok in output_tokens)
    print(f"  {text}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Duplex model inference test")
    p.add_argument("--adapter",      default=None,   help="duplex_lora adapter dir (省略でベースモデル)")
    p.add_argument("--audio",        required=True,  help="wav ファイルパス")
    p.add_argument("--mode",         default="batch", choices=["batch", "stream"],
                   help="batch: 1 forward 全スロット読み / stream: KV キャッシュ逐次")
    p.add_argument("--task",         default="asr",  choices=["asr", "chat"],
                   help="<|turn>asr or chat (学習時の mode と合わせる)")
    p.add_argument("--gpu",          default="1",    help="CUDA_VISIBLE_DEVICES")
    p.add_argument("--temperature",  type=float, default=0.0,
                   help="0=argmax, >0=サンプリング (stream モードのみ有効)")
    p.add_argument("--top-k",        type=int, default=5,
                   help="batch モードで表示する top-k 候補数")
    p.add_argument("--base-model",   default="google/gemma-4-12b-it")
    p.add_argument("--max-frames",   type=int, default=750)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0")

    tok_src = args.adapter or args.base_model
    print(f"[load] tokenizer: {tok_src}")
    tokenizer = AutoTokenizer.from_pretrained(tok_src)

    pad_ids = tokenizer.encode(PAD_TOKEN, add_special_tokens=False)
    assert len(pad_ids) == 1, f"PAD トークン '{PAD_TOKEN}' が複数トークンになっています: {pad_ids}"
    pad_id = pad_ids[0]
    print(f"[load] PAD token: '{PAD_TOKEN}' = id {pad_id}")

    print(f"[load] base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    if args.adapter:
        print(f"[load] adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    else:
        print("[load] adapter なし（ベースモデル）")
    model.eval()

    print(f"[audio] {args.audio}")
    audio  = load_audio_16k(args.audio)
    frames = audio_to_frames(audio, args.max_frames)
    print(f"[audio] {len(audio)/SAMPLE_RATE:.2f}s → {len(frames)} frames")

    if args.mode == "batch":
        infer_batch(model, tokenizer, frames, pad_id, device,
                    top_k=args.top_k, temperature=args.temperature,
                    mode=args.task)
    else:
        infer_stream(model, tokenizer, frames, pad_id, device,
                     temperature=max(args.temperature, 0.8),
                     mode=args.task)


if __name__ == "__main__":
    main()

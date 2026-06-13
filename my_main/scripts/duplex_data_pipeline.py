#!/usr/bin/env python3
"""
Duplex Data Pipeline: J-CHAT -> 時刻整列ストリーミング学習用データセット

設計書: my_main/research/weekly/202606W1/0612/1000_full_duplex_redesign.md

J-CHAT shar の supervision 2 種（話者セグメント / トークン単位時刻付き転写）を
保持したままデータセット化する。phase1_data_pipeline.py との違い:
  - テキストを " ".join で潰さず、トークンごとの (text, start, end) を保持
  - 話者セグメントとの時刻照合でトークンに話者を割り当てる
  - asr / dialog の 2 モードを同一スキーマで出力する

Modes:
  asr    : 話者ターン単位に切り出し。テキストチャンネル = そのターンの全トークン
           (Phase 1': 時刻整列ストリーミング ASR 用)
  dialog : 2 話者・低オーバーラップの cut を丸ごと使用。
           テキストチャンネル = モデル役話者のトークンのみ（話者ごとに 1 サンプル）
           (Phase 2: full-duplex 対話用)

Output Dataset schema (両モード共通):
  id            str       サンプル ID (cut_id + suffix)
  mode          str       "asr" | "dialog"
  audio_array   f32[]     16kHz モノラル PCM
  n_samples     int32
  n_frames      int32     40ms フレーム数
  duration_s    f32
  token_texts   str[]     テキストチャンネルのトークン（時刻順）
  token_starts  f32[]     秒。audio_array 先頭基準
  token_ends    f32[]
  speaker       str       モデル役話者 (asr ではターンの話者)
  n_speakers    int32     元 cut の話者数
  overlap_ratio f32       元 cut の発話オーバーラップ率
  source_cut    str       元 cut ID

Usage:
  # Phase 1' 用 (streaming ASR)
  python3 my_main/scripts/duplex_data_pipeline.py \\
      --mode asr \\
      --manifest /workspace/jchat_data/transcribed_jchat/podcast_train.json \\
      --shards 8 --max-samples 5000 \\
      --output-dir my_main/duplex_data/asr_train

  # Phase 2 用 (full-duplex dialog)
  python3 my_main/scripts/duplex_data_pipeline.py \\
      --mode dialog \\
      --manifest /workspace/jchat_data/transcribed_jchat/podcast_train.json \\
      --shards 16 --max-samples 2000 \\
      --output-dir my_main/duplex_data/dialog_train
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

SAMPLE_RATE   = 16_000
CHUNK_SAMPLES = 640          # 40ms @ 16kHz
FRAME_S       = 0.04

# asr モード
ASR_MIN_DUR    = 1.0         # ターンの最短長 [s]
ASR_MAX_DUR    = 30.0        # ターンの最長長 [s]（超過はスキップ）
ASR_MIN_TOKENS = 2
ASR_PAD_S      = 0.16        # ターン前後に残す文脈 [s]
TURN_MERGE_GAP = 1.0         # 同一話者セグメントをターンに統合する最大ギャップ [s]

# dialog モード
DLG_N_SPEAKERS  = 2
DLG_MAX_OVERLAP = 0.05
DLG_MIN_DUR     = 10.0
DLG_MIN_TOKENS  = 5

TOKEN_ASSIGN_TOL = 1.0       # セグメント外トークンを最近傍に割り当てる許容距離 [s]


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == SAMPLE_RATE:
        return audio.astype(np.float32)
    from scipy.signal import resample_poly
    g  = math.gcd(SAMPLE_RATE, orig_sr)
    return resample_poly(audio, SAMPLE_RATE // g, orig_sr // g).astype(np.float32)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.abs(audio).max()
    if peak < 1e-8:
        return audio
    return (audio / peak * 0.95).astype(np.float32)


# ---------------------------------------------------------------------------
# Supervision parsing
# ---------------------------------------------------------------------------

@dataclass
class Seg:
    speaker: str
    start: float
    end: float


@dataclass
class Tok:
    text: str
    start: float
    end: float
    speaker: str = ""


def parse_supervisions(cut) -> tuple[list[Seg], list[Tok]]:
    """cut.supervisions を話者セグメントとテキストトークンに分離する。"""
    segs, toks = [], []
    for s in getattr(cut, "supervisions", []) or []:
        text = getattr(s, "text", None)
        speaker = getattr(s, "speaker", None)
        start = float(s.start)
        end   = start + float(s.duration)
        if text:
            toks.append(Tok(text=text, start=start, end=end))
        elif speaker:
            segs.append(Seg(speaker=speaker, start=start, end=end))
    segs.sort(key=lambda x: x.start)
    toks.sort(key=lambda x: x.start)
    return segs, toks


def assign_token_speakers(segs: list[Seg], toks: list[Tok]) -> list[Tok]:
    """
    各トークンを時刻照合で話者セグメントに割り当てる。
    セグメント間のギャップに落ちたトークンは最近傍セグメント
    (距離 TOKEN_ASSIGN_TOL 以内) に割り当て、それ以遠は破棄する。
    """
    out = []
    for t in toks:
        mid = (t.start + t.end) / 2
        best, best_dist = None, float("inf")
        for g in segs:
            if g.start <= mid < g.end:
                best, best_dist = g, 0.0
                break
            dist = max(g.start - mid, mid - g.end)
            if dist < best_dist:
                best, best_dist = g, dist
        if best is not None and best_dist <= TOKEN_ASSIGN_TOL:
            out.append(Tok(text=t.text, start=t.start, end=t.end, speaker=best.speaker))
    return out


def merge_turns(segs: list[Seg]) -> list[Seg]:
    """連続する同一話者セグメントをギャップ TURN_MERGE_GAP 以内で 1 ターンに統合。"""
    turns: list[Seg] = []
    for g in segs:
        if turns and turns[-1].speaker == g.speaker and g.start - turns[-1].end <= TURN_MERGE_GAP:
            turns[-1].end = max(turns[-1].end, g.end)
        else:
            turns.append(Seg(speaker=g.speaker, start=g.start, end=g.end))
    return turns


def overlap_ratio(segs: list[Seg], duration: float) -> float:
    """隣接セグメント間の重なり時間の割合。"""
    if duration <= 0 or len(segs) < 2:
        return 0.0
    ivs = sorted((g.start, g.end) for g in segs)
    ov = sum(max(0.0, min(ivs[i][1], ivs[i + 1][1]) - ivs[i + 1][0])
             for i in range(len(ivs) - 1))
    return ov / duration


# ---------------------------------------------------------------------------
# Sample builders
# ---------------------------------------------------------------------------

def _make_record(sample_id: str, mode: str, audio: np.ndarray,
                 toks: list[Tok], speaker: str, n_speakers: int,
                 ov: float, source_cut: str) -> dict:
    return {
        "id":            sample_id,
        "mode":          mode,
        "audio_array":   audio.tolist(),
        "n_samples":     len(audio),
        "n_frames":      math.ceil(len(audio) / CHUNK_SAMPLES),
        "duration_s":    len(audio) / SAMPLE_RATE,
        "token_texts":   [t.text for t in toks],
        "token_starts":  [t.start for t in toks],
        "token_ends":    [t.end for t in toks],
        "speaker":       speaker,
        "n_speakers":    n_speakers,
        "overlap_ratio": ov,
        "source_cut":    source_cut,
    }


def asr_samples_from_cut(cut, audio16k: np.ndarray) -> Iterator[dict]:
    """話者ターン単位の streaming ASR サンプルを生成する。"""
    segs, toks = parse_supervisions(cut)
    if not segs or not toks:
        return
    toks = assign_token_speakers(segs, toks)
    turns = merge_turns(segs)
    n_speakers = len({g.speaker for g in segs})
    ov = overlap_ratio(segs, float(cut.duration))
    total_s = len(audio16k) / SAMPLE_RATE

    for i, turn in enumerate(turns):
        dur = turn.end - turn.start
        if not (ASR_MIN_DUR <= dur <= ASR_MAX_DUR):
            continue
        turn_toks = [t for t in toks
                     if t.speaker == turn.speaker
                     and turn.start - ASR_PAD_S <= (t.start + t.end) / 2 < turn.end + ASR_PAD_S]
        if len(turn_toks) < ASR_MIN_TOKENS:
            continue

        t0 = max(0.0, turn.start - ASR_PAD_S)
        t1 = min(total_s, turn.end + ASR_PAD_S)
        s0, s1 = int(t0 * SAMPLE_RATE), int(t1 * SAMPLE_RATE)
        audio = normalize_audio(audio16k[s0:s1])
        if len(audio) < CHUNK_SAMPLES:
            continue

        rebased = [Tok(text=t.text,
                       start=max(0.0, t.start - t0),
                       end=min(t1 - t0, max(0.0, t.end - t0)),
                       speaker=t.speaker)
                   for t in turn_toks]
        yield _make_record(f"{cut.id}_turn{i}", "asr", audio, rebased,
                           turn.speaker, n_speakers, ov, cut.id)


def dialog_samples_from_cut(cut, audio16k: np.ndarray,
                            max_dur: float) -> Iterator[dict]:
    """2 話者・低オーバーラップ cut から full-duplex 対話サンプルを生成する。"""
    segs, toks = parse_supervisions(cut)
    if not segs or not toks:
        return
    speakers = sorted({g.speaker for g in segs})
    if len(speakers) != DLG_N_SPEAKERS:
        return
    ov = overlap_ratio(segs, float(cut.duration))
    if ov > DLG_MAX_OVERLAP:
        return
    if float(cut.duration) < DLG_MIN_DUR:
        return

    toks = assign_token_speakers(segs, toks)

    # max_dur 秒に切り詰め（音声・トークンとも）
    n_keep = min(len(audio16k), int(max_dur * SAMPLE_RATE))
    audio = normalize_audio(audio16k[:n_keep])
    limit_s = n_keep / SAMPLE_RATE

    for spk in speakers:
        spk_toks = [Tok(text=t.text, start=t.start, end=min(t.end, limit_s), speaker=spk)
                    for t in toks if t.speaker == spk and t.start < limit_s]
        if len(spk_toks) < DLG_MIN_TOKENS:
            continue
        yield _make_record(f"{cut.id}_{spk}", "dialog", audio, spk_toks,
                           spk, len(speakers), ov, cut.id)


# ---------------------------------------------------------------------------
# J-CHAT shar iteration
# ---------------------------------------------------------------------------

def iter_samples(manifest_path: str, mode: str, shards: int,
                 max_dialog_dur: float) -> Iterator[dict]:
    try:
        import lhotse
    except ImportError:
        sys.exit("ERROR: lhotse not found. Install: pip install lhotse webdataset smart-open")

    print(f"[jchat] manifest: {manifest_path}")
    with open(manifest_path) as f:
        fields = json.load(f)

    n_avail = len(fields["cuts"])
    use = min(shards, n_avail)
    fields = {k: v[:use] for k, v in fields.items() if k in ("cuts", "recording")}
    print(f"[jchat] using {use}/{n_avail} shards (~{use * 1000} cuts)")

    cuts = lhotse.CutSet.from_shar(fields=fields)

    n_cuts = 0
    for cut in cuts:
        n_cuts += 1
        try:
            raw = cut.load_audio()
            if raw.ndim == 2:
                raw = raw.mean(axis=0)
            audio16k = resample_to_16k(raw.astype(np.float32), int(cut.sampling_rate))
        except Exception as e:
            print(f"  [skip] {cut.id}: audio load failed: {e}", file=sys.stderr)
            continue

        if mode == "asr":
            yield from asr_samples_from_cut(cut, audio16k)
        else:
            yield from dialog_samples_from_cut(cut, audio16k, max_dialog_dur)

    print(f"[jchat] processed {n_cuts} cuts")


# ---------------------------------------------------------------------------
# Dataset build & verify
# ---------------------------------------------------------------------------

def build_and_save(samples: Iterator[dict], output_dir: str,
                   max_samples: Optional[int]) -> None:
    from datasets import Dataset, Features, Value, Sequence, concatenate_datasets

    features = Features({
        "id":            Value("string"),
        "mode":          Value("string"),
        "audio_array":   Sequence(Value("float32")),
        "n_samples":     Value("int32"),
        "n_frames":      Value("int32"),
        "duration_s":    Value("float32"),
        "token_texts":   Sequence(Value("string")),
        "token_starts":  Sequence(Value("float32")),
        "token_ends":    Sequence(Value("float32")),
        "speaker":       Value("string"),
        "n_speakers":    Value("int32"),
        "overlap_ratio": Value("float32"),
        "source_cut":    Value("string"),
    })

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    batches, buf, total = [], [], 0
    t0 = time.monotonic()
    for s in samples:
        buf.append(s)
        total += 1
        if len(buf) >= 500:
            batches.append(Dataset.from_dict({k: [x[k] for x in buf] for k in buf[0]},
                                             features=features))
            print(f"  [batch] {total:>6d} samples  elapsed={time.monotonic()-t0:.0f}s")
            buf.clear()
        if max_samples and total >= max_samples:
            break
    if buf:
        batches.append(Dataset.from_dict({k: [x[k] for x in buf] for k in buf[0]},
                                         features=features))
    if not batches:
        sys.exit("[error] no samples produced")

    ds = concatenate_datasets(batches)
    durs = ds["duration_s"]
    n_toks = [len(t) for t in ds["token_texts"]]
    print(f"\n[dataset] {len(ds)} samples  total={sum(durs)/3600:.2f}h  "
          f"avg={sum(durs)/len(durs):.1f}s  avg_tokens={sum(n_toks)/len(n_toks):.0f}")
    ds.save_to_disk(str(out))

    manifest = {
        "n_samples":      len(ds),
        "mode":           ds[0]["mode"],
        "total_hours":    round(sum(durs) / 3600, 3),
        "avg_duration_s": round(sum(durs) / len(durs), 2),
        "avg_tokens":     round(sum(n_toks) / len(n_toks), 1),
        "output_dir":     str(out),
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[dataset] saved to {out}")


def verify_dataset(output_dir: str) -> None:
    """先頭サンプルの時刻整合性を確認する。"""
    from datasets import load_from_disk
    ds = load_from_disk(output_dir)
    print(f"\n[verify] {len(ds)} samples")
    for i in range(min(3, len(ds))):
        s = ds[i]
        dur = s["duration_s"]
        starts, ends = s["token_starts"], s["token_ends"]
        assert all(0 <= a <= dur + 0.5 for a in starts), f"token start out of range: {s['id']}"
        assert all(a <= b + 1e-6 for a, b in zip(starts, ends)), f"start>end: {s['id']}"
        assert sorted(starts) == starts, f"not chronological: {s['id']}"
        text = "".join(s["token_texts"])
        print(f"  [{i}] {s['id']}  dur={dur:.1f}s  frames={s['n_frames']}  "
              f"tokens={len(starts)}  spk={s['speaker']}")
        print(f"      text: {text[:80]}")
    print("[verify] OK")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode",        required=True, choices=["asr", "dialog"])
    p.add_argument("--manifest",    default="/workspace/jchat_data/transcribed_jchat/podcast_train.json")
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--shards",      type=int, default=4,
                   help="使用する shar シャード数 (1 shard ≈ 1000 cuts ≈ 14h)")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-dialog-dur", type=float, default=60.0,
                   help="dialog モードの最大 cut 長 [s] (長い cut は先頭を切り出し)")
    p.add_argument("--verify",      action="store_true")
    args = p.parse_args()

    samples = iter_samples(args.manifest, args.mode, args.shards, args.max_dialog_dur)
    build_and_save(samples, args.output_dir, args.max_samples)
    if args.verify:
        verify_dataset(args.output_dir)


if __name__ == "__main__":
    main()

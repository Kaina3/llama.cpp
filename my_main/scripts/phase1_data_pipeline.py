#!/usr/bin/env python3
"""
Phase 1: J-CHAT Data Pipeline for Gemma 4 UA Incremental Audio Fine-tuning

J-CHAT コーパスから Gemma 4 UA の LoRA ファインチューニング用訓練データを生成する。

Pipeline:
  1. J-CHAT CutSet を lhotse でロード（shar/webdataset 形式）
  2. 各 Cut を 16kHz モノラルにリサンプリング
  3. 発話区間を 40ms チャンク (640 samples) に分割
  4. 転写テキストを取得
  5. HuggingFace Dataset (.arrow) として保存

Output Dataset schema:
  - id          : str    -- J-CHAT のカット ID
  - audio_array : list[float32]  -- 16kHz モノラル PCM (float32, -1.0〜1.0)
  - n_samples   : int   -- audio_array の長さ
  - n_frames    : int   -- 40ms フレーム数 (= ceil(n_samples / 640))
  - duration_s  : float -- 発話時間 [秒]
  - text        : str   -- 転写テキスト
  - speaker     : str   -- 話者 ID (あれば)
  - source      : str   -- "podcast" | "youtube"

Training format (how to use this dataset):
  [BOS] [<start_of_turn>user\n]
  [audio_embed_0] ... [audio_embed_N]   <- project audio_array at training time
  [<end_of_turn>\n<start_of_turn>model\n]
  [text tokens]                          <- loss computed here
  [<end_of_turn>][EOS]

HuggingFace J-CHAT アクセス手順:
  1. https://huggingface.co/datasets/sarulab-speech/J-CHAT でライセンス同意
  2. huggingface-cli login  (または HF_TOKEN 環境変数を設定)
  3. 本スクリプトを実行

Usage:
  # J-CHAT (shar + 転写あり) から生成
  python3 phase1_data_pipeline.py \\
      --jchat-json  /path/to/transcribed_jchat/podcast_train.json \\
      --source      podcast \\
      --output-dir  /workspace/phase1_data/podcast_train \\
      --max-samples 5000

  # J-CHAT を HuggingFace Hub から直接ストリーミング (要認証)
  python3 phase1_data_pipeline.py \\
      --hf-streaming podcast_train \\
      --output-dir   /workspace/phase1_data/podcast_train \\
      --max-samples  5000

  # テストモード: 既存サンプル WAV でパイプラインを検証
  python3 phase1_data_pipeline.py \\
      --test-mode \\
      --wav-dir   /workspace/llama.cpp/my_main/sample \\
      --output-dir /workspace/phase1_data/test
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  Constants (Gemma 4 UA)
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16_000   # 16kHz
CHUNK_SAMPLES = 640      # 40ms @ 16kHz
MIN_DURATION  = 0.5      # 秒未満の発話はスキップ
MAX_DURATION  = 30.0     # 秒超の発話はトリミング

# ─────────────────────────────────────────────────────────────────────────────
#  Audio helpers
# ─────────────────────────────────────────────────────────────────────────────
def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """
    任意のサンプルレートの float32 モノラル配列を 16kHz に変換する。
    scipy.signal.resample_poly を使用（高品質・軽量）。
    """
    if orig_sr == SAMPLE_RATE:
        return audio
    from scipy.signal import resample_poly
    from math import gcd
    g   = gcd(SAMPLE_RATE, orig_sr)
    up  = SAMPLE_RATE // g
    dn  = orig_sr // g
    return resample_poly(audio, up, dn).astype(np.float32)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """ピーク正規化。無音の場合はそのまま返す。"""
    peak = np.abs(audio).max()
    if peak < 1e-8:
        return audio
    return audio / peak


def audio_to_chunks(audio: np.ndarray) -> list[np.ndarray]:
    """1D float32 配列を 640-sample チャンクのリストに分割（末尾ゼロパディング）。"""
    n = math.ceil(len(audio) / CHUNK_SAMPLES)
    chunks = []
    for i in range(n):
        c = audio[i * CHUNK_SAMPLES : (i + 1) * CHUNK_SAMPLES]
        if len(c) < CHUNK_SAMPLES:
            c = np.pad(c, (0, CHUNK_SAMPLES - len(c)))
        chunks.append(c)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
#  J-CHAT loader (lhotse shar 形式)
# ─────────────────────────────────────────────────────────────────────────────
def _iter_jchat_shar(json_path: str, source: str) -> Iterator[dict]:
    """
    J-CHAT shar 形式の JSON を読み込み、サンプル辞書のイテレータを返す。

    各サンプル:
      {id, audio_array (float32 1-D), duration_s, text, speaker, source}
    """
    try:
        import lhotse
    except ImportError:
        sys.exit("ERROR: lhotse not found. Install: pip install lhotse webdataset smart-open")

    print(f"[jchat] Loading shar manifest: {json_path}")
    with open(json_path) as f:
        fields = json.load(f)
    cuts = lhotse.CutSet.from_shar(fields=fields)

    for cut in cuts:
        # 転写テキストを取得
        sups = getattr(cut, "supervisions", []) or []
        text = " ".join(s.text for s in sups if getattr(s, "text", None))
        if not text.strip():
            continue

        # 時間フィルタ
        dur = float(cut.duration)
        if dur < MIN_DURATION:
            continue
        if dur > MAX_DURATION:
            # 長すぎる場合は前半 MAX_DURATION 秒だけ使う
            cut = cut.truncate(duration=MAX_DURATION)
            dur = MAX_DURATION

        # 音声ロード + 16kHz リサンプリング
        try:
            # lhotse は (channels, samples) の float32 ndarray を返す
            raw = cut.load_audio()  # (C, T)
            if raw.ndim == 2:
                raw = raw.mean(axis=0)  # モノラルに
            audio = resample_to_16k(raw.astype(np.float32),
                                    int(cut.sampling_rate))
            audio = normalize_audio(audio)
        except Exception as e:
            print(f"  [skip] {cut.id}: audio load failed: {e}", file=sys.stderr)
            continue

        # 話者 ID
        speaker = sups[0].speaker if sups and getattr(sups[0], "speaker", None) else ""

        yield {
            "id":          cut.id,
            "audio_array": audio.tolist(),
            "n_samples":   len(audio),
            "n_frames":    math.ceil(len(audio) / CHUNK_SAMPLES),
            "duration_s":  dur,
            "text":        text.strip(),
            "speaker":     speaker or "",
            "source":      source,
        }


def _iter_jchat_hf_streaming(split: str) -> Iterator[dict]:
    """
    HuggingFace Hub から J-CHAT を直接ストリーミングロード（認証が必要）。
    lhotse の代わりに HF datasets を使用するフォールバック実装。
    ※ J-CHAT は通常 shar 形式のため、実際には shar 経由を推奨。
    """
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("ERROR: datasets not found. Install: pip install datasets")

    print(f"[jchat] HF streaming: sarulab-speech/J-CHAT / {split}")
    ds = load_dataset("sarulab-speech/J-CHAT", split=split,
                      streaming=True, trust_remote_code=True)
    for sample in ds:
        audio_info = sample.get("audio") or {}
        raw   = np.array(audio_info.get("array", []), dtype=np.float32)
        sr    = audio_info.get("sampling_rate", SAMPLE_RATE)
        text  = sample.get("transcription") or sample.get("text") or ""
        if not text.strip() or len(raw) == 0:
            continue

        audio = resample_to_16k(raw, sr)
        audio = normalize_audio(audio)
        dur   = len(audio) / SAMPLE_RATE
        if dur < MIN_DURATION or dur > MAX_DURATION:
            continue

        yield {
            "id":          sample.get("id", ""),
            "audio_array": audio.tolist(),
            "n_samples":   len(audio),
            "n_frames":    math.ceil(len(audio) / CHUNK_SAMPLES),
            "duration_s":  dur,
            "text":        text.strip(),
            "speaker":     sample.get("speaker", ""),
            "source":      split,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  テストモード: ローカル WAV ファイルを使用
# ─────────────────────────────────────────────────────────────────────────────
def _iter_local_wav(wav_dir: str) -> Iterator[dict]:
    """
    ローカル WAV ディレクトリからサンプルを生成（開発・検証用）。
    転写テキストは空文字。
    """
    try:
        import soundfile as sf
    except ImportError:
        try:
            import scipy.io.wavfile as wavfile
            _sf = None
        except ImportError:
            sys.exit("ERROR: soundfile or scipy not found.")
        _sf = None
    else:
        _sf = sf

    wav_dir = Path(wav_dir)
    wavs = sorted(wav_dir.glob("*.wav"))
    if not wavs:
        sys.exit(f"ERROR: No .wav files found in {wav_dir}")

    print(f"[test] Found {len(wavs)} WAV files in {wav_dir}")

    for wav_path in wavs:
        try:
            if _sf is not None:
                raw, sr = _sf.read(str(wav_path), dtype="float32", always_2d=False)
            else:
                import scipy.io.wavfile as wf
                sr, raw = wf.read(str(wav_path))
                if raw.dtype == np.int16:
                    raw = raw.astype(np.float32) / 32768.0
                elif raw.dtype == np.int32:
                    raw = raw.astype(np.float32) / 2147483648.0
                raw = raw.astype(np.float32)

            if raw.ndim == 2:
                raw = raw.mean(axis=1)
            audio = resample_to_16k(raw, int(sr))
            audio = normalize_audio(audio)
            dur   = len(audio) / SAMPLE_RATE
        except Exception as e:
            print(f"  [skip] {wav_path.name}: {e}", file=sys.stderr)
            continue

        yield {
            "id":          wav_path.stem,
            "audio_array": audio.tolist(),
            "n_samples":   len(audio),
            "n_frames":    math.ceil(len(audio) / CHUNK_SAMPLES),
            "duration_s":  dur,
            "text":        "",   # テストモードでは転写なし
            "speaker":     "",
            "source":      "local",
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset ビルダー
# ─────────────────────────────────────────────────────────────────────────────
def build_and_save(
    samples: Iterable[dict],
    output_dir: str,
    max_samples: Optional[int] = None,
    batch_size: int = 500,
) -> None:
    """
    サンプルイテレータから HuggingFace Dataset を構築し、output_dir に保存する。

    保存形式: Arrow (datasets.Dataset.save_to_disk)
    ロード方法:
      from datasets import load_from_disk
      ds = load_from_disk('/path/to/output_dir')
    """
    try:
        from datasets import Dataset, Features, Value, Sequence
    except ImportError:
        sys.exit("ERROR: datasets not found. Install: pip install datasets")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = Features({
        "id":          Value("string"),
        "audio_array": Sequence(Value("float32")),
        "n_samples":   Value("int32"),
        "n_frames":    Value("int32"),
        "duration_s":  Value("float32"),
        "text":        Value("string"),
        "speaker":     Value("string"),
        "source":      Value("string"),
    })

    # バッチ単位で収集して Dataset を構築
    all_batches: list[Dataset] = []
    buf: list[dict] = []
    total = 0
    t0 = time.monotonic()

    for sample in samples:
        buf.append(sample)
        total += 1

        if len(buf) >= batch_size:
            batch_dict = {k: [s[k] for s in buf] for k in buf[0]}
            all_batches.append(Dataset.from_dict(batch_dict, features=features))
            n = len(buf)
            elapsed = time.monotonic() - t0
            avg_dur = sum(s["duration_s"] for s in buf) / n
            print(f"  [batch] {total:>6d} samples  avg_dur={avg_dur:.1f}s  "
                  f"elapsed={elapsed:.1f}s")
            buf.clear()

        if max_samples and total >= max_samples:
            break

    # 残りバッファを保存
    if buf:
        batch_dict = {k: [s[k] for s in buf] for k in buf[0]}
        all_batches.append(Dataset.from_dict(batch_dict, features=features))

    if not all_batches:
        print("[warn] No samples were processed.", file=sys.stderr)
        return

    # バッチを連結
    from datasets import concatenate_datasets
    dataset = concatenate_datasets(all_batches)

    # 統計情報を表示
    durations = dataset["duration_s"]
    total_h   = sum(durations) / 3600.0
    print(f"\n[dataset] {len(dataset)} samples  "
          f"total={total_h:.2f}h  "
          f"avg={sum(durations)/len(durations):.2f}s  "
          f"min={min(durations):.2f}s  max={max(durations):.2f}s")

    # 保存
    dataset.save_to_disk(str(output_dir))
    print(f"[dataset] Saved to {output_dir}")

    # マニフェスト JSON（人間可読）
    manifest = {
        "n_samples":  len(dataset),
        "total_hours": round(total_h, 3),
        "avg_duration_s": round(sum(durations) / len(durations), 3),
        "min_duration_s": round(float(min(durations)), 3),
        "max_duration_s": round(float(max(durations)), 3),
        "output_dir": str(output_dir),
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[dataset] Manifest written: {output_dir / 'manifest.json'}")


# ─────────────────────────────────────────────────────────────────────────────
#  データセット検証
# ─────────────────────────────────────────────────────────────────────────────
def verify_dataset(dataset_dir: str, mmproj_path: Optional[str] = None) -> None:
    """
    保存された Dataset を読み込み、音声投影が正常に動作するか検証する。

    --mmproj が指定されている場合は Gemma4UAProjector を実行して
    埋め込みベクトルに NaN が含まれないことを確認する。
    """
    try:
        from datasets import load_from_disk
    except ImportError:
        sys.exit("ERROR: datasets not found.")

    ds = load_from_disk(dataset_dir)
    print(f"[verify] Dataset: {len(ds)} samples")

    # 先頭3サンプルを検証
    for i in range(min(3, len(ds))):
        s = ds[i]
        audio = np.array(s["audio_array"], dtype=np.float32)
        chunks = audio_to_chunks(audio)
        assert len(chunks) == s["n_frames"], \
            f"n_frames mismatch: {len(chunks)} != {s['n_frames']}"

        print(f"  [{i}] id={s['id']}  dur={s['duration_s']:.2f}s  "
              f"frames={s['n_frames']}  text={s['text'][:40]!r}")

        if mmproj_path:
            # Gemma4UAProjector を使って投影テスト
            # phase0_streaming.py の Gemma4UAProjector をインポート
            scripts_dir = Path(__file__).parent
            sys.path.insert(0, str(scripts_dir))
            from phase0_streaming import Gemma4UAProjector, RMS_EPS
            proj = Gemma4UAProjector(mmproj_path)
            embeds = [proj.project(c) for c in chunks]
            embed_arr = np.stack(embeds)
            has_nan = bool(np.isnan(embed_arr).any())
            print(f"       embed shape={embed_arr.shape}  nan={has_nan}")
            assert not has_nan, "NaN found in embeddings!"

    print("[verify] OK")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1: J-CHAT → Gemma 4 UA training dataset builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # データソース（いずれか1つ必須）
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--jchat-json", metavar="JSON",
        help="J-CHAT transcribed shar manifest JSON (例: transcribed_jchat/podcast_train.json)",
    )
    src.add_argument(
        "--hf-streaming", metavar="SPLIT",
        help="HuggingFace Hub から直接ストリーミング (例: podcast_train). 認証が必要.",
    )
    src.add_argument(
        "--test-mode", action="store_true",
        help="ローカル WAV ファイルを使用したテストモード (--wav-dir と組み合わせて使用)",
    )

    # オプション
    parser.add_argument("--source",      default="podcast",
                        help="ソース識別子 (podcast | youtube). --jchat-json 時に使用")
    parser.add_argument("--wav-dir",     default=None,
                        help="テストモード時のローカル WAV ディレクトリ")
    parser.add_argument("--output-dir",  required=True,
                        help="出力 Dataset ディレクトリ")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="処理するサンプルの上限 (デフォルト: 無制限)")
    parser.add_argument("--verify",      action="store_true",
                        help="保存後に Dataset を検証する")
    parser.add_argument("--mmproj",      default=None,
                        help="--verify 時に使用する mmproj GGUF パス")
    parser.add_argument("--batch-size",  type=int, default=500,
                        help="バッチあたりのサンプル数 (デフォルト: 500)")

    args = parser.parse_args()

    # ソースを選択
    if args.jchat_json:
        samples = _iter_jchat_shar(args.jchat_json, args.source)
    elif args.hf_streaming:
        samples = _iter_jchat_hf_streaming(args.hf_streaming)
    else:  # test-mode
        wav_dir = args.wav_dir or str(
            Path(__file__).parent.parent / "sample"
        )
        samples = _iter_local_wav(wav_dir)

    # データセット構築 + 保存
    build_and_save(
        samples     = samples,
        output_dir  = args.output_dir,
        max_samples = args.max_samples,
        batch_size  = args.batch_size,
    )

    # 検証
    if args.verify:
        verify_dataset(args.output_dir, args.mmproj)


if __name__ == "__main__":
    main()

# Phase 1: J-CHAT データパイプライン 実装ログ

**日付**: 2026-06-08  
**担当**: Anderson  
**ブランチ**: `feat/phase0-server-improvements`  
**スクリプト**: `my_main/scripts/phase1_data_pipeline.py`

---

## 概要

Phase 0 で「40ms チャンクの逐次注入 → テキスト生成」の配管が機能することを確認した。
Phase 1 では Gemma 4 12B UA を J-CHAT コーパスでファインチューニングし、
音声入力への応答精度とタイミング制御を改善する。

本ドキュメントはその **第一歩: データパイプラインの実装** のログである。

---

## 実装内容

### スクリプト: `my_main/scripts/phase1_data_pipeline.py`

J-CHAT コーパスから Gemma 4 UA 学習用データセットを生成するパイプライン。

```
J-CHAT (lhotse CutSet / HF Hub)
    ↓ 発話単位で切り出し
    ↓ 16kHz モノラルにリサンプリング
    ↓ ピーク正規化
    ↓ 0.5〜30秒でフィルタリング
    ↓ HuggingFace Dataset (.arrow) として保存
/workspace/phase1_data/{split}/
```

#### 出力 Dataset スキーマ

| カラム | 型 | 説明 |
|--------|-----|------|
| `id` | string | J-CHAT の Cut ID |
| `audio_array` | float32[] | 16kHz モノラル PCM (-1.0〜1.0) |
| `n_samples` | int32 | サンプル数 |
| `n_frames` | int32 | 40ms フレーム数 (= ⌈n_samples / 640⌉) |
| `duration_s` | float32 | 発話時間 [秒] |
| `text` | string | 転写テキスト |
| `speaker` | string | 話者 ID |
| `source` | string | "podcast" / "youtube" |

#### 学習時のシーケンス形式

```
[BOS]
[<start_of_turn>user\n]
[audio_embed_0] [audio_embed_1] ... [audio_embed_N]   ← 学習時に project
[<end_of_turn>\n<start_of_turn>model\n]
[text tokens]                                          ← ここに Loss を計算
[<end_of_turn>] [EOS]
```

#### 3つのデータソース

| モード | フラグ | 用途 |
|--------|--------|------|
| J-CHAT shar + 転写 | `--jchat-json` | 本番 (要 HF 認証) |
| HF Hub ストリーミング | `--hf-streaming` | 本番フォールバック |
| ローカル WAV | `--test-mode` | 開発・検証用 |

---

## 動作確認結果

```
[test] Found 3 WAV files in /workspace/llama.cpp/my_main/sample

[dataset] 3 samples  total=0.01h  avg=7.56s  min=3.85s  max=10.00s
[dataset] Saved to /workspace/phase1_data/test
[dataset] Manifest written: /workspace/phase1_data/test/manifest.json

[verify] Dataset: 3 samples
  [0] id=gemma4_audio_qa_input  dur=10.00s  frames=250  text=''
       embed shape=(250, 3840)  nan=False
  [1] id=gemma4_own_voice       dur=8.84s   frames=221  text=''
       embed shape=(221, 3840)  nan=False
  [2] id=gemma4_test_ja         dur=3.85s   frames=97   text=''
       embed shape=(97, 3840)   nan=False
[verify] OK
```

- 音声 → 40ms チャンク分割: ✅  
- Gemma4UAProjector 投影 (NaN なし): ✅  
- HuggingFace Dataset 形式での保存: ✅

---

## J-CHAT へのアクセス手順

J-CHAT は CC-BY-NC 4.0 ライセンス（研究・非商用のみ）。
HuggingFace でのライセンス同意が必要。

```bash
# 1. ブラウザで同意
#    https://huggingface.co/datasets/sarulab-speech/J-CHAT

# 2. コンテナ内で HF 認証
docker exec -it kaina-llama-dev bash
huggingface-cli login   # → HF_TOKEN を入力

# 3. メタデータ JSON をダウンロード
huggingface-cli download sarulab-speech/J-CHAT \
    transcribed_jchat/podcast_train.json \
    --repo-type dataset \
    --local-dir /workspace/jchat_data/

# 4. データパイプラインを実行
python3 /workspace/llama.cpp/my_main/scripts/phase1_data_pipeline.py \
    --jchat-json  /workspace/jchat_data/transcribed_jchat/podcast_train.json \
    --source      podcast \
    --output-dir  /workspace/phase1_data/podcast_train \
    --max-samples 5000 \
    --verify
```

---

## 次のステップ: Phase 1A → Phase 1B

```
Phase 1A (完了) ──────────────────────────────────────────────
  [✅] データパイプライン実装 (phase1_data_pipeline.py)
  [✅] サンプル音声での動作確認
  [  ] J-CHAT (5,000〜10,000 サンプル) での本番実行
       → llama.cpp/my_main/research/weekly/202606W1/0608/
         1100_phase1a_jchat_prep.md で記録予定

Phase 1B (次に実装) ───────────────────────────────────────────
  LoRA ファインチューニングスクリプト
  → llama.cpp/my_main/scripts/phase1_lora_finetune.py

Phase 2 (その後) ──────────────────────────────────────────────
  ステレオ対話ファインチューニング (full-duplex)
```

詳細は [1001_phase1_next_steps.md](1001_phase1_next_steps.md) を参照。

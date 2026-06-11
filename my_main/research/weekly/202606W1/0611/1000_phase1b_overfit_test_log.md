# Phase 1B: LoRA オーバーフィットテスト 実施ログ

**日付**: 2026-06-11  
**スクリプト**: `my_main/scripts/phase1_lora_finetune.py`  
**データ**: `my_main/phase1_data/podcast_train`（5000 サンプル中 500 件使用）  
**GPU**: NVIDIA A100 PCIE 40GB × 1（device 1）

---

## 目的

本番学習（5000 サンプル）の前に、以下を検証する「パイプライン正常性確認」。

1. 音声フレームがモデルに正しく入力されているか
2. 勾配が `embed_audio.embedding_projection` と LoRA 両方に流れているか
3. loss が下降するか（過学習できるか）

---

## 発見 1: モデルアーキテクチャの誤認識（修正済）

スクリプトの冒頭コメントに `audio_tower - Gemma4UnifiedAudioModel (num_layers=0)` と記載していたが、実際に読み込まれるモデルは異なるクラスだった。

| 項目 | 誤（旧認識） | 正（実際） |
|---|---|---|
| model_type | `gemma4` | **`gemma4_unified`** |
| 最上位クラス | `Gemma4ForConditionalGeneration` | **`Gemma4UnifiedForConditionalGeneration`** |
| 音声処理 | `Gemma4AudioModel`（Conv2d サブサンプリング 4x 含む） | **`Gemma4UnifiedMultimodalEmbedder`（サブサンプリングなし）** |

### Unified モデルの音声パイプライン

```
raw PCM (float32, 16kHz)
  └─ 640 サンプル = 1 フレーム = 1 audio token
       └─ embed_audio.embedding_pre_projection_norm  (RMSNorm, 学習パラメータなし)
       └─ embed_audio.embedding_projection           (Linear 640→3840, 学習対象)
            └─ inputs_embeds へ masked_scatter で注入
```

- **Conv2d サブサンプリングは存在しない**。1 フレーム = 1 トークンの 1:1 対応
- コレーターが `n_frames` 個の `audio_token_id` を `input_ids` に埋め込む設計は正しい
- `input_features` / `input_features_mask` のキー名・形状 `(B, T, 640)` も正しい

---

## 発見 2: 第 1 回オーバーフィットテストで loss が全く下がらなかった

### 症状

500 サンプル × 3 epoch、すべての loss が 10.2〜11.2 台で推移。ランダム基準（ln(260000) ≈ 12.5）に近く、実質ゼロ学習。

### 根本原因: LR が小さすぎた

| パラメータ | 値 | 問題 |
|---|---|---|
| lr_lora | 2e-6 | 本番向けの保守値 |
| lr_audio | 1e-5 | 同上 |
| grad_accum | 16 | 500 サンプル → 31 更新/epoch |
| 3 epoch 合計 | **93 回の重み更新** | 重みがほぼ動かない |

AdamW の実効ステップ幅 ≈ lr × grad ≈ 2e-6 × 1.0 = **2e-6**。93 ステップでは重みの変化量が初期化スケールの 0.01% 未満。

### 修正内容

`phase1_lora_finetune.py` に `--lr-scale <float>` 引数を追加。全 LR に乗数をかけられるようにした。

```bash
--lr-scale 100  # lr_audio=1e-3, lr_lora=2e-4（オーバーフィットテスト用）
--lr-scale 10   # lr_audio=1e-4, lr_lora=2e-5（本番用推奨）
--lr-scale 1    # デフォルト（旧来値、低すぎる）
```

また、重み更新ごとに grad norm を表示するようログを強化した：

```
loss=4.0897  lr=2.00e-04  gnorm_audio=3.09e+01  gnorm_lora=2.61e+00
```

---

## 第 2 回オーバーフィットテスト結果

### 実行コマンド

```bash
python3 my_main/scripts/phase1_lora_finetune.py \
    --model  google/gemma-4-12b-it \
    --data   my_main/phase1_data/podcast_train \
    --output my_main/phase1_lora_overfit \
    --gpu 1 --samples 500 --epochs 5 --lr-scale 100
```

### ハイパーパラメータ

| パラメータ | 値 |
|---|---|
| lr_lora | 2e-4（2e-6 × 100） |
| lr_audio | 1e-3（1e-5 × 100） |
| lora_rank / alpha | 16 / 32 |
| LoRA 対象レイヤー | decoder 最終 4 層（44〜47） |
| grad_accum | 16 |
| 総更新回数 | 500 / 16 × 5 ≈ 156 回 |

### loss ログ

| epoch | step | loss | gnorm_audio | gnorm_lora |
|-------|------|------|-------------|------------|
| 1 | 10 | 10.5657 | 7.61e+02 | 1.04e+01 |
| 1 | 20 | 7.4910 | 6.56e+03 | 1.02e+01 |
| 1 | 30 | 6.2232 | 2.81e+02 | 5.72e+00 |
| 2 | 40 | 5.7098 | 2.75e+02 | 3.43e+00 |
| 2 | 50 | 5.3644 | 5.32e+01 | 2.38e+00 |
| 2 | 60 | 5.0292 | 4.25e+02 | 2.43e+00 |
| 3 | 70 | 5.3363 | 1.77e+02 | 3.24e+00 |
| 3 | 80 | 5.2093 | 1.99e+02 | 1.97e+00 |
| 3 | 90 | 4.7114 | 7.60e+01 | 2.20e+00 |
| 4 | 100 | 4.5536 | 4.28e+01 | 1.81e+00 |
| 4 | 110 | 4.4618 | 1.47e+01 | 1.53e+00 |
| 4 | 120 | 4.3328 | 2.17e+01 | 2.66e+00 |
| 5 | 130 | 4.0897 | 3.09e+01 | 2.61e+00 |
| 5 | 140 | 4.0250 | 2.55e+02 | 2.08e+00 |
| 5 | 150 | 4.1093 | 7.65e+01 | 2.30e+00 |

### 評価

- **loss: 10.57 → 4.03（ランダム基準 12.5 から大幅低下）** ✅
- gnorm_audio と gnorm_lora ともに有意な値 → 音声投影・LoRA の両方に勾配が流れている ✅
- 5 epoch 後 loss が 3 台に届かなかった理由：156 更新では 500 サンプルの完全記憶には不足。lr_scale=100 による gnorm_audio の高騰（6000 超）が step 20 以降の振動に影響した可能性もある

**パイプラインの正常性は確認された。本番学習に進む。**

---

## 本番学習の設定

### コマンド

```bash
python3 my_main/scripts/phase1_lora_finetune.py \
    --model  google/gemma-4-12b-it \
    --data   my_main/phase1_data/podcast_train \
    --output my_main/phase1_lora \
    --gpu    1 \
    --epochs 3 \
    --lr-scale 10 \
    --save-steps 200
```

### 変更点と理由

| 項目 | オーバーフィット | 本番 | 理由 |
|---|---|---|---|
| samples | 500 | 5000（全件） | 全データ使用 |
| epochs | 5 | 3 | データが 10x 多いので 1 周の情報量が大きい |
| lr-scale | 100 | **10** | lr_scale=100 は gnorm_audio が過大（6000+）。10 で汎化性を確保 |
| save-steps | — | 200 | 約 0.6 epoch ごとにチェックポイント保存 |

### 期待される指標

- 1 epoch 終了時 loss: 5〜7 台（データが多い分 1 epoch での下降幅は小さい）
- 3 epoch 終了時 loss: 3〜5 台
- gnorm_audio: 10〜100 台（過大な場合は lr-scale を下げる）

---

## 残課題

- [ ] 本番学習完了後、LoRA adapter を base model にマージ
- [ ] マージ済みモデルを `convert_hf_to_gguf.py` で GGUF 変換
- [ ] llama.cpp で推論テスト（音声入力 → テキスト出力）
- [ ] Phase 2: ステレオ対話ファインチューニング（full-duplex）

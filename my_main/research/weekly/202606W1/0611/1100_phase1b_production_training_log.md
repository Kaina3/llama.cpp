# Phase 1B: 本番学習 実施ログ

**日付**: 2026-06-11  
**スクリプト**: `my_main/scripts/phase1_lora_finetune.py`  
**データ**: `my_main/phase1_data/podcast_train`（5000 サンプル全件）  
**GPU**: NVIDIA A100 PCIE 40GB × 1（device 1）  
**出力**: `my_main/phase1_lora/`

---

## 実行コマンド

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

## ハイパーパラメータ

| パラメータ | 値 |
|---|---|
| samples | 5000（全件） |
| epochs | 3 |
| lr_audio | 1e-4（1e-5 × lr-scale 10） |
| lr_lora | 2e-5（2e-6 × lr-scale 10） |
| lora_rank / alpha | 16 / 32 |
| LoRA 対象 | decoder 最終 4 層（q_proj, v_proj） |
| grad_accum | 16 |
| 総ステップ数 | 937 steps/epoch × 3 epoch |
| 有効バッチサイズ | batch × grad_accum（スクリプトデフォルト） |

---

## 結果

### 初期 loss

- **epoch 1 開始時**: ~11

### 終盤ログ（epoch 3 終了付近）

| epoch | step | loss | gnorm_audio | gnorm_lora |
|-------|------|------|-------------|------------|
| 3 | 900/937 | 3.4150 | 3.65e+02 | 3.12e+00 |
| 3 | 910/937 | 3.3076 | 9.34e+01 | 3.35e+00 |
| 3 | 920/937 | 3.4289 | 4.80e+01 | 3.57e+00 |
| 3 | 930/937 | 3.3737 | 1.99e+02 | 6.64e+00 |

※ 中間ログは未保存。終盤 4 点のみ記録。

### 最終 loss

- **epoch 3 終盤**: **3.31〜3.43**（平均 ~3.35）
- 初期比: 11 → 3.35（**約 70% 減少**）
- perplexity 換算: e^3.35 ≈ **28.5**

---

## 評価

### 良い点

- loss が 11 → 3.35 に大幅減少。学習パイプラインは正常に機能している。
- `gnorm_lora` は 3〜7 程度で安定。LoRA 側は収束傾向にある。
- アダプタの保存が正常に完了（`my_main/phase1_lora/`）。

### 懸念点

**① `gnorm_audio` が不安定**

```
step=900: 3.65e+02
step=910: 9.34e+01  （-75%）
step=920: 4.80e+01  （-49%）
step=930: 1.99e+02  （+315%）
```

1 ステップで 4〜7 倍の変動。`embed_audio.embedding_projection` の学習が完全には収束していない。`lr_audio=1e-4` が依然として高い可能性がある。

**② loss が 3.3〜3.4 で停滞**

epoch 3 終盤で loss の下降が止まっている。日本語音声転写タスクとして well-trained なモデルでは loss < 1.0 が期待値。perplexity ≈ 28 は「音声から次トークンをある程度絞り込めているが、まだ不確実性が高い」状態。

**③ 実転写精度が未検証**

loss は間接指標。実際に音声 → テキスト変換が機能するかは推論テストで確認が必要。

---

## 次フェーズへの判断

| 選択肢 | 条件 |
|---|---|
| **A. このまま GGUF 変換 → 推論テストへ進む** | loss 3.35 でも転写が実用レベルなら次フェーズへ |
| **B. 追加学習（epoch 延長 or lr 調整）** | 推論テストで CER/WER が高い場合。`lr_audio` を 5e-5 に下げて再実験 |

**推奨: まず推論テストを行い、転写精度を実測してから判断する。**

---

## 残課題

- [ ] llama.cpp 推論テスト（音声入力 → テキスト出力）で CER/WER を実測
- [ ] LoRA adapter を base model にマージ（`peft.merge_adapter`）
- [ ] マージ済みモデルを `convert_hf_to_gguf.py` で GGUF 変換
- [ ] llama.cpp に組み込んで Phase 0 ストリーミングパイプラインと結合テスト
- [ ] （必要なら）`lr_audio` を下げて追加学習（gnorm_audio 安定化）

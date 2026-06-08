# Phase 1 全体計画: J-CHAT × Gemma 4 UA インクリメンタル音声ファインチューニング

**日付**: 2026-06-08  
**担当**: Anderson

---

## 現在地

```
Phase 0 (完了) ─ 40ms チャンク逐次注入 → テキスト生成の配管を確認
Phase 1A (進行中) ─ 学習データパイプラインを実装
```

Phase 0 で検証できたこと:
- 92フレーム (3.68秒) の音声を KV キャッシュに逐次注入: ✅  
- Gemma 4 12B が音声内容（エベレスト）を正しく理解: ✅  
- 課題: 注入速度が実時間比 1.0 に近く、低遅延対話には最適化が必要

---

## Phase 1 の目的

### なぜファインチューニングが必要か？

Gemma 4 12B UA の事前学習では、音声は「**塊として一括入力**」されている。  
Phase 0 のようにストリーミングで注入しても動くのは、モデルが汎用的だから。  
しかし以下の能力は事前学習されていない:

| 未学習の能力 | 具体的な問題 |
|-------------|-------------|
| チャンク途中での早期応答 | 発話が終わる前に答えを生成し始めない |
| バックチャネル ("ええ"、"なるほど") | 相槌応答ができない |
| 途中割り込みの検出 | ユーザーが発話を止めたことに気づかない |
| 口語・話し言葉の理解 | 書き言葉との差異 (J-CHAT で補う) |

ファインチューニングにより「**音声がチャンク単位で来ても応答できる**」能力を付与する。

---

## Phase 1 全体構成 (2ステップ)

### Phase 1A: データパイプライン (今回実装済み)

```
J-CHAT corpus (76,000h) 
    → 発話単位 Cut
    → 16kHz リサンプリング + ピーク正規化
    → HuggingFace Dataset (.arrow) として保存
```

**スクリプト**: `my_main/scripts/phase1_data_pipeline.py`  
**出力形式**: HuggingFace Dataset  
**学習に使う規模目安**: 5,000〜10,000 サンプル (まず小規模で検証)

### Phase 1B: LoRA ファインチューニング (次に実装)

```
HuggingFace Dataset (from Phase 1A)
    → Gemma 4 12B UA (PyTorch, HF Transformers)
    → PEFT LoRA (音声投影層 + 最終4層)
    → LoRA アダプタ保存
    → convert_hf_to_gguf.py で GGUF に変換
    → llama.cpp で推論
```

**スクリプト**: `my_main/scripts/phase1_lora_finetune.py` (未実装)  
**依存**: `transformers`, `peft`, `trl`, `bitsandbytes`

---

## Phase 1B の実装設計 (次に着手すべき内容)

### LoRA 適用箇所

| レイヤー | 学習方針 | 理由 |
|---------|---------|------|
| `audio_tower.mm_input_projection` | フル学習 | チャンク入力への適応が最重要 |
| `model.layers[-4:]` (最終4層) | LoRA (rank=16) | 音声対話スタイルの調整 |
| `model.layers[:-4]` | 凍結 | 日本語・推論能力を保護 |

### 学習設定 (推奨値)

| パラメータ | 値 |
|-----------|-----|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| target_modules | `q_proj`, `v_proj`, `mm_input_projection` |
| 学習率 (投影層) | 1e-5 |
| 学習率 (Decoder LoRA) | 2e-6 |
| バッチサイズ | 4 (gradient_accumulation=16 → 実効64) |
| エポック数 | 3 |
| 最大シーケンス長 | 2048 トークン相当 |

### 学習シーケンスの構造

```python
# 入力シーケンス (audio embeds + テキスト)
[BOS] [<start_of_turn>user\n]
[audio_embed_0] ... [audio_embed_N]      # N = n_frames (最大750 @ 30秒)
[<end_of_turn>\n<start_of_turn>model\n]
[text tokens]                            # ← ここだけ Loss を計算
[<end_of_turn>] [EOS]
```

**Loss mask**: 音声埋め込み部分は Loss を計算しない  
(テキスト出力部分だけを学習対象にする)

### HuggingFace Gemma 4 UA モデルの確認事項

学習を開始する前に確認が必要な点:

```bash
# Gemma 4 12B IT の HF モデル ID を確認
# (audio 対応版は gemma-4-12b-it または gemma-4-12b-ua)
python3 -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('google/gemma-4-12b-it')
print(cfg.model_type)
print(dir(cfg))
"
```

- `model_type` が `gemma4` であることを確認
- 音声投影層のモジュール名を `cfg` から特定
- PEFT の `target_modules` に正確なモジュール名を指定

---

## Phase 2 以降の展望

### Phase 2: ステレオ対話ファインチューニング

Phase 1 が「音声 → テキスト転写」の改善なのに対し、  
Phase 2 では「話者 A の発話音声 → 話者 B のテキスト応答」を学習する。

```
入力: <USR> [audio_embed × N] </USR>  (話者 A の発話)
出力: <SYS> [テキスト応答]     </SYS>  (話者 B の応答)
```

J-Moshi の 344h ステレオデータに相当する対話コーパスで学習する。

### Phase 3 (任意): 音声出力との統合

- Voicevox / COEIROINK などの OSS TTS と組み合わせ
- テキスト応答を音声に変換してフル音声対話を実現

---

## 実装ロードマップ

```
Week 1 (0608〜0615)
  [✅] Phase 1A: データパイプライン (phase1_data_pipeline.py)
  [  ] Phase 1A: J-CHAT 本番実行 (HF 認証 + 5000 サンプル生成)

Week 2 (0615〜0622)
  [  ] Phase 1B: LoRA ファインチューニングスクリプト
       (phase1_lora_finetune.py)
  [  ] Phase 1B: 小規模実験 (500 サンプルで過学習確認)

Week 3〜4
  [  ] Phase 1B: 5000+ サンプルでの本番学習
  [  ] LoRA → GGUF 変換・Phase 0 パイプラインとの統合テスト

Month 2+
  [  ] Phase 2: ステレオ対話データパイプライン
  [  ] Phase 2: Full-duplex ファインチューニング
```

---

## 参考情報

| 項目 | 値/リンク |
|------|----------|
| J-CHAT データセット | https://huggingface.co/datasets/sarulab-speech/J-CHAT |
| J-CHAT 論文 | https://arxiv.org/abs/2407.15828 |
| J-Moshi 論文 | https://arxiv.org/abs/2506.02979 |
| Gemma 4 12B IT (HF) | google/gemma-4-12b-it |
| Phase 0 ログ | [../0606/1600_phase0_streaming_implementation.md](../0606/1600_phase0_streaming_implementation.md) |
| 改造計画全体 | [../0606/1500_gemma4_12b_audio_improvement.md](../0606/1500_gemma4_12b_audio_improvement.md) |

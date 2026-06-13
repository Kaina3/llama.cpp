# 2026-06-12 作業サマリー

**担当**: Anderson  
**ブランチ**: `feat/phase1b-torch-compat`

---

## 概要

Phase 1B（バッチ音声→転写 LoRA）の失敗を受け、full-duplex 実現に向けた学習フォーマットを
根本から再設計し、新スクリプト群の実装とオーバーフィットテストまで完了した。

---

## 1. 実施した作業

### 1-A. Full-Duplex 再設計（詳細: [1000_full_duplex_redesign.md](1000_full_duplex_redesign.md)）

#### 問題の特定

Phase 1B の学習結果（loss 3.35 の LoRA）が推論時にスペース + EOS しか出力しない原因を分析し、
以下の 3 つの複合バグを確定した。

| # | 問題 | 影響 |
|---|---|---|
| 1 | `embedding_projection` を lr=1e-4 でフル学習（gnorm 48〜365 で乱高下） | 事前学習済みの音声→埋め込み投影を破壊 |
| 2 | スペース区切りテキスト（旧パイプラインの `" ".join()` 由来） | 自然な日本語分布と衝突 |
| 3 | 「音声のみ→転写」と「音声のみ→応答」が同一入力に別出力を要求 | 競合学習 |

また、現行の Phase 0（VAD 起動）+ Phase 1（一括転写）は構造的に半二重であり、
Moshi の full-duplex の核心「毎フレーム PAD/テキストを選択する」メカニズムが存在しないことも確認。

#### 決定事項

- Phase 1B の LoRA アダプタは**破棄**（比較用に保管のみ）
- 新フォーマット「時刻整列ストリーミング」へ移行

#### 新フォーマット仕様

40ms ごとに「音声トークン + テキストスロット」を交互に並べる。

```
[BOS] <|turn>asr\n <|audio> [A_0][S_0][A_1][S_1]...[A_n][S_n] <audio|> <turn|>\n
```

- `A_t`: 40ms 音声フレーム
- `S_t`: そのフレームでモデルが出すべきトークン。無言なら `<mask>`（PAD）
- loss はテキストスロットのみ（PAD 含む）
- `embed_audio.embedding_projection` は**凍結**
- LoRA は全 48 decoder 層の q/k/v/o + MLP に拡大

#### J-CHAT データ形式調査

J-CHAT shar（先頭シャード 1000 cuts）を実地検証し、以下を確認した。

| 発見 | 設計への影響 |
|---|---|
| トークン単位（40〜640ms 粒度）の時刻が既に含まれている | WhisperX による強制アライメント**不要** |
| 話者セグメントが diarization 済みで含まれている | pyannote **不要** |
| 旧パイプラインが `" ".join()` していたことが原因でスペース混入 | 新パイプラインでスペースなし連結に修正 |
| 2 話者 + 低オーバーラップ（≤5%）の cut が全体の約 24%（推定 14,000h） | Phase 2 データは選別で十分確保可能 |

#### 実装完了スクリプト

| スクリプト | 役割 |
|---|---|
| `my_main/scripts/duplex_data_pipeline.py` | J-CHAT shar → 時刻整列 HuggingFace Dataset |
| `my_main/scripts/duplex_finetune.py` | 時刻整列インターリーブ形式 LoRA 学習（projection 凍結、PAD/text 分離メトリクス） |
| `my_main/scripts/plot_metrics.py` | metrics.jsonl → 学習曲線 PNG（学習中でも実行可） |

---

### 1-B. オーバーフィットテスト（詳細: [1100_phase1prime_overfit_analysis.md](1100_phase1prime_overfit_analysis.md)）

#### 事前実験でのPAD崩壊の発見

新スクリプトの初回実行（pad_weight=0.1 / 0.05）で新たな問題が発生した。

| 実験 | 設定 | 結果 |
|---|---|---|
| A | pad_weight=0.1, 100 samples | acc_t: 0.011 → **0**（崩壊） |
| B | pad_weight=0.05, 500 samples | acc_t: 0.017 → **0**（崩壊） |

**根本原因**: PAD スロットが 83〜89% を占めるため、`<mask>` の初期 loss が高い（18.8）状態から学習が始まると、
PAD 勾配がテキスト勾配を圧倒して「常時 PAD 出力」に収束する。LoRA パラメータを PAD/text で共有しているため、
PAD logit が上がると text ポジションでも PAD logit が argmax になる。

実験 B のステップ 40 でそれが確認された（gnorm=243 でパラメータが急激に更新→以降 acc_t=0 で固定）。

#### 解決策: カリキュラム学習（Stage A → B）

| Stage | 設定 | 目的 |
|---|---|---|
| **A** | `pad_loss_weight=0.0` | PAD 勾配をゼロにしてテキスト学習だけを先に確立 |
| **B** | `pad_loss_weight=0.1`（Stage A から再開） | テキスト logit の優位を保ちつつ PAD 学習を追加 |

#### Stage A オーバーフィットテスト結果

```
コマンド: --samples 1000 --epochs 10 --lr 5e-5 --pad-loss-weight 0.0
```

| step | train loss | train acc_t | val loss | val acc_t |
|------|-----------|------------|---------|----------|
| 150  | ~4.3      | 0.24       | 4.61    | 0.213    |
| 200  | 3.59      | 0.315      | 3.73    | 0.298    |
| **300** | **2.78** | **0.421** | **3.25** | **0.382** ← val 最小 |
| 400  | 1.62      | 0.620      | 3.68    | 0.374    |
| 500  | 1.13      | 0.731      | 4.12    | 0.362    |
| 600  | 0.87      | 0.803      | 4.66    | 0.351    |
| 620  | 0.78      | 0.827      | —       | —        |

**判定: Stage A 成功。**  
PAD 崩壊が完全に解消され、acc_t が 0.24 → 0.83 と単調増加。val loss は step 300 を底に
以降上昇（オーバーフィット）し、モデルがタスクを学習できることを実証した。

---

## 2. 結果サマリー

| 項目 | 状態 |
|---|---|
| Phase 1B 失敗原因の特定 | ✅ 確定（projection 破壊 + スペース + 競合学習の複合） |
| 新フォーマット（時刻整列）の設計 | ✅ 完了 |
| J-CHAT データ形式調査 | ✅ 完了（WhisperX/pyannote 不要を確認） |
| スクリプト 3 本の実装 | ✅ 完了 |
| PAD 崩壊問題の発見と根本原因特定 | ✅ 完了 |
| Stage A オーバーフィットテスト | ✅ 成功（acc_t 83%、val loss 最小 3.25 @ step 300） |
| Stage B（PAD 学習追加） | 🔲 未実施 |
| 本番学習（フルデータ）| 🔲 未実施 |

---

## 3. 次のアクション

### 直近: Stage B オーバーフィットテスト

Stage A checkpoint-400（val loss=3.68）から再開し、PAD 学習を追加する。

```bash
python3 my_main/scripts/duplex_finetune.py \
    --data    my_main/duplex_data/asr_train \
    --output  my_main/duplex_lora_stage_b \
    --resume  my_main/duplex_lora_stage_a/checkpoint-400 \
    --gpu     0,1 \
    --epochs  3 \
    --lr      1e-5 \
    --pad-loss-weight 0.1
```

確認すべき点:
- acc_t が 0 に崩壊しないこと（Stage A で確立した text logit の優位が保たれること）
- acc_p が上昇すること（PAD 学習が進行していること）

### その後: 本番学習

Stage B が成功したら、フルデータ（5000 samples 以上）で本番学習を実施。
val loss 最小チェックポイントの保存間隔は `--save-steps 50` に変更推奨（step 300 付近が
オーバーフィットテストでの最適点だったが、save-steps=200 のため保存されなかった）。

---

## 4. 参照ドキュメント

- [1000_full_duplex_redesign.md](1000_full_duplex_redesign.md) — 再設計の背景・仕様・実装
- [1100_phase1prime_overfit_analysis.md](1100_phase1prime_overfit_analysis.md) — PAD 崩壊の分析とカリキュラム学習の設計
- [../0611/1100_phase1b_production_training_log.md](../0611/1100_phase1b_production_training_log.md) — Phase 1B 本番学習ログ（破棄の経緯）

# Full-Duplex 再設計: 時刻整列ストリーミング学習への移行

**日付**: 2026-06-12
**担当**: Anderson
**前提ドキュメント**:
- [改造計画全体](../0606/1500_gemma4_12b_audio_improvement.md)
- [Phase 0 実装ログ](../0606/1600_phase0_streaming_implementation.md)
- [Phase 1B 本番学習ログ](../0611/1100_phase1b_production_training_log.md)

---

## 1. 設計レビューの結論

Phase 1B（一括 audio→転写 LoRA）の結果と Moshi/J-Moshi の構造を比較した結果、
**現行設計のままでは full-duplex に到達できない**と判断し、学習フォーマットを再設計する。

| 評価 | 内容 |
|---|---|
| OK | Gemma 4 Unified を Moshi の Temporal Transformer 相当として流用する発想。Phase 0 で 40ms 逐次注入の配管は実証済み |
| OK | 音声出力を捨ててテキスト出力 + TTS にするスコープ削減 |
| NG | full-duplex の核心「時刻同期テキストストリーム（PAD）」が現行 Phase 1 に存在しない |
| NG | 一括 audio→転写の学習は、事前学習済み音声理解と Phase 2 の応答タスクの両方と競合する |

### ギャップ 1: full-duplex の正体は「毎フレーム、話すか黙るかを選ぶ」こと

Moshi の full-duplex は VAD や割り込み検出ロジックではなく、以下の構造で実現されている:

- 時間が一定刻みで進み、毎ステップでモデルは自分のチャンネルに必ず何かを出す（沈黙 = PAD トークン）
- ユーザーチャンネルの音声は毎ステップ無条件に注入され続ける
- 「いつ話し始めるか」「相槌」「割り込まれたら黙る」は、すべて
  「PAD をやめてテキストを出す / テキストをやめて PAD に戻る」という単一の決定に統一される

現行の Phase 0（VAD 起動）+ Phase 1（一括転写）は「聞き終わってから応答する」構造であり、
これをいくら磨いても半二重のまま。

### ギャップ 2: Phase 1B は事前学習資産と衝突していた

Phase 0 でベースモデルが音声内容（エベレスト）を正しく理解・回答できたことから、
`embed_audio.embedding_projection` は事前学習済みで機能している。それにもかかわらず:

1. projection を lr=1e-4 でフル学習した（gnorm_audio が 48〜365 で乱高下 = 事前学習済み投影を破壊した可能性）
2. J-CHAT のスペース区切りテキストをそのまま教え、自然な日本語の事前分布と衝突した
3. 「音声のみ→転写」は Phase 2 の「音声のみ→応答」と同一入力に別出力を要求する競合学習だった

推論テストで転写が出ずスペース + EOS のみ生成されたのは、underfit ではなく上記の複合と推定。

---

## 2. 確定した方針（2026-06-11 議論）

| 論点 | 決定 |
|---|---|
| 現行 Phase 1B 成果物（loss 3.35 の LoRA） | **破棄して再設計**。アダプタは比較用に保管のみ |
| Phase 2 用の時刻付き対話データ | **J-CHAT を話者分離**（pyannote 等で diarization）して確保 |

---

## 3. 新フォーマット: 時刻整列ストリーミング

### 3.1 シーケンス構造

40ms ごとに「音声フレーム 1 トークン + テキストスロット 1 トークン」を交互に並べる。

```
[BOS] <mode>
[A_0][PAD][A_1][PAD][A_2][遠][A_3][い][A_4][PAD]...
```

- `A_t`: 40ms 音声フレーム（audio_token_id 位置に埋め込み注入、従来同様）
- テキストスロット: その時点でモデルが発するべきトークン。無言なら PAD
- 1 秒 = 50 トークン（音声 25 + テキスト 25）。5 分対話 = 15k トークン
- PAD は Gemma の未使用トークン（`<unused0>` 系）を転用
- `<mode>`: タスク識別トークン。Phase 1'（ASR）と Phase 2（応答）を系列先頭で区別し、競合を回避

### 3.2 テキスト配置ルール

- 単語のテキストトークンは「その単語の音声終了 + 遅延 k フレーム」の位置に置く（k=2〜5、ハイパラ）
- タイムスタンプは **J-CHAT shar に含まれるトークン単位の時刻をそのまま使用**（9 章参照）。
  WhisperX 等での強制アライメントは不要
- テキストはスペースなしで連結して正規化（旧パイプラインの空白 join をやめる）

### 3.3 Loss 設計

- loss はテキストスロットのみ（音声トークン位置は除外、従来同様）
- **PAD も loss 対象**。「黙るべき時に PAD を出す」こと自体が full-duplex の学習対象

### 3.4 Phase 1' と Phase 2 の関係

| | Phase 1'（streaming ASR） | Phase 2（full-duplex 対話） |
|---|---|---|
| 音声チャンネル | 任意の発話 | 話者 A（ユーザー役）の発話 |
| テキストチャンネル | 同じ発話の転写（時刻整列） | 話者 B（モデル役）の発話テキスト（実時刻配置） |
| データ | J-CHAT 発話単位（既存 5000 件を再加工） | J-CHAT diarization 済み 2 話者対話区間 |
| 目的 | 音声→テキストの時刻整列を獲得 | 発話タイミング・相槌・ターンテイキングを獲得 |

フォーマットが共通なので、コレーターと推論ループは両フェーズで使い回せる。
Phase 1' はそのまま Phase 2 のカリキュラム前段になる。

---

## 4. 学習設定の変更

| 項目 | 旧（破棄） | 新 |
|---|---|---|
| embedding_projection | フル学習 lr=1e-4 | **凍結**（事前学習済みを保護） |
| LoRA 範囲 | 最終 4 層 q_proj/v_proj | **全 48 層 q/k/v/o + MLP**、rank 16〜32 |
| テキスト | スペース区切り・一括転写 | 正規化済み・時刻整列・PAD 同期 |
| タイミング学習 | なし（VAD 任せ） | PAD/テキストの選択として学習 |

projection 凍結により gnorm_audio の不安定要因は消える。
LoRA 全層化でもパラメータは全体の ~0.5% 程度で、A100 40GB x 1 で学習可能。

---

## 5. 推論ループ（Phase 0 からの変更）

`phase0_streaming.py` をベースに以下を変更:

1. VAD 起動を廃止
2. 毎フレーム: 音声注入 → 同じ forward の logits からテキストスロットを 1 トークンサンプル
3. PAD なら次フレームを待つ、テキストなら即時出力（将来は TTS へ）

→ 1 forward / 40ms。Phase 0 実測 34ms/frame から、量子化モデルで実時間動作が狙える。
VAD・割り込み検出などの外部ロジックは不要になる。

---

## 6. 実行ロードマップ

```
Step 1: 診断（着手中）
  ベースモデル（adapter なし）+ <|turn> 形式 + テキスト指示で
  HF パイプラインの転写能力を確認
  → ベースで転写できれば、以降は「能力獲得」ではなく「フォーマット適応」の問題に帰着

Step 2: データ再構築（J-CHAT 形式調査済み、9 章参照）
  - 新データパイプライン: shar からトークン時刻 + 話者セグメントを保持したまま抽出
  - WhisperX / pyannote は不要（両方とも J-CHAT に含まれていた）
  - Dataset スキーマ: audio(16k) + tokens[{text, start, dur}] + speaker_segments[{spk, start, dur}]

Step 3: 新コレーター実装（時刻整列フォーマット、3.1〜3.3 仕様）

Step 4: Phase 1' 再学習（projection 凍結 + LoRA 全層）

Step 5: 推論ループ書き換え（毎フレーム 1 トークンサンプル方式）

Step 6: Phase 2 データ準備（2 話者 / 低オーバーラップの cut を選別、並行着手可）
```

---

## 7. Step 1 診断結果（2026-06-11 実施）

`phase1_infer.py` を拡張（`--adapter` 省略可、`--instruction`、`--audio-delimiters` 追加）し、
**ベースモデル（adapter なし）** で `gemma4_test_ja.wav`（3.85s、97 frames）を転写テストした。

| 条件 | 生成結果 |
|---|---|
| 指示あり（音声の後に「一字一句書き起こして」） | `こんにちは。これは禅問答の文字起こしテストです。` |
| 指示あり + 音声境界トークン | `こんにちは。これは、ゼンマ法の文字起こしテストです。` |
| 音声のみ（指示なし） | `こんにちは！禅問答の文字起こしテストですね。承知いたしました。準備はできています。どうぞ...`（+ 冒頭にゴミトークン） |

### 結論

1. **ベースモデルの音声理解は HF パイプラインでも健在**。指示があればほぼ正確に転写できる
   （「Gemma 4」を「禅問答 / ゼンマ法」と聞き違える程度の、固有名詞レベルの誤りのみ）。
   → Phase 1B adapter がスペース + EOS しか出さなかったのは**学習による劣化**で確定。
   projection 凍結方針の正しさを裏付ける。
2. **「LLM がどこが音声か分からない」という仮説は否定された**。指示なし（音声のみ）でも
   モデルは音声の内容を完全に理解して会話的に応答している。問題は「音声の位置が分からない」
   ことではなく「**そのタスク（転写）をやれと言われていない**」こと。
   → 新フォーマットで「音声→時刻整列テキスト」を教えること自体は成立する。
3. **未使用だった音声境界トークンを発見**: `<|audio>`=256000、`<audio|>`=258883。
   事前学習ではこれで音声区間が囲まれていた可能性が高い。
   新コレーターでは音声区間をこの境界トークンで囲む。
4. 全条件で冒頭に `thought`（チャンネルヘッダ）が混入する。新フォーマットの学習で
   モデルチャンネルを直接教えれば消える見込み。

---

## 8. 留意点

- **モノラル音声の限界**: J-CHAT は mono のため、話者 B（モデル役）の声も入力音声に混入する。
  推論時はマイク入力（ユーザーのみ）なので条件が一致しない。
  実測ではオーバーラップは平均 4.1%、76% の cut が 5% 未満（9 章）なので、
  低オーバーラップ cut の選別で実害は小さい。
  モデル自身の発話はテキストとして文脈に残るため、音声として聞こえる必要はない。
- **ライセンス**: J-CHAT は CC-BY-NC 4.0 + 日本著作権法 30 条の 4 目的限定（研究用途のみ）。
  成果物の配布に注意。
- **音声とテキストの区別について**: LLM 本体にとって入力はすべて埋め込み列だが、
  (1) 事前学習で音声埋め込みの分布を経験済みであること、
  (2) 音声境界トークン `<|audio>` / `<audio|>` とプロンプト構造が手がかりになること、
  から区別できる。Step 1 診断（7 章）で実証済み。
- **PAD トークン**: 設計時は `<unused0>` 系を想定したが、Gemma 4 vocab に存在しないため
  `<mask>` (id=4) を転用した（10.1 参照）。

---

## 9. J-CHAT データ形式調査（2026-06-12 実施）

`transcribed_jchat/podcast_train.json` を再取得し、shar シャード（cuts.000000.jsonl.gz、1000 cuts）を実地検証した。

### 9.1 形式

- JSON はデータ本体ではなく **lhotse shar のマニフェスト**: `{cuts, features, recording}` 各 4226 シャードの URL リスト
  （`https://s3ds.mdx.jp/jchat-transcribed/podcast_train/...`）
- 転写は Whisper ではなく **reazonspeech-nemo-v2** による
- cut = 対話の連続区間（MonoCut、22.05kHz → 16k へ要リサンプル）

### 9.2 cut に含まれる supervision は 2 種類

| 種類 | 内容 | 例 |
|---|---|---|
| 話者セグメント | `speaker`（SPEAKER_00/01...）+ start/duration。**diarization 済み** | `[SPEAKER_01] 9.38-13.08s` |
| テキストトークン | `text` + start/duration。**トークン単位（40〜640ms 粒度）の時刻付き** | `start=0.220 dur=0.400 text='違う'` |

例（2 話者 cut、42.3s）:

```
[SPEAKER_01]  0.00- 1.03s: だいぶ違うんですよ。
[SPEAKER_00]  1.74- 9.38s: なんだ。気になるな。なのでいつかはハイスペックって思ってて。
[SPEAKER_01]  9.38-13.08s: 何だっけエルデンリングがやりたいって言ってましたね昔ね。
[SPEAKER_00] 13.03-23.22s: エルデンリングもやりたかったし。...
```

### 9.3 統計（先頭シャード 1000 cuts）

| 指標 | 値 |
|---|---|
| cut 長 | 平均 50.0s / 中央値 35.2s / 最大 579s |
| 話者数分布 | 2 人: 32%、3 人: 34%、4 人: 23%、5 人以上: 11% |
| 発話オーバーラップ率 | 平均 4.1%、**76% の cut が 5% 未満** |
| 規模 | 1 シャード 13.9h → podcast_train 全体推定 **約 59,000h** |

### 9.4 設計への影響（実装に反映済み）

1. **WhisperX 強制アライメント不要** — トークン単位時刻が既にある（3.2 の要件を満たす）
2. **pyannote diarization 不要** — 話者セグメントが既にある（Step 6 の前提を満たす）
3. **旧パイプラインのスペース区切りテキストの原因が判明** — `phase1_data_pipeline.py` が
   supervision の text を `" ".join()` していた。新パイプラインではスペースなし連結に変更
4. テキストトークンに speaker は付かないため、話者セグメントとの時刻照合で割り当てる
   （セグメント間ギャップに落ちるトークンがあるため、最近傍セグメントへの割り当て等の許容が必要）
5. Phase 2 用には「2 話者 + 低オーバーラップ」の cut を選別すればよい。32% × 76% でも
   全体推定 59,000h の約 24% ≈ 14,000h 相当が候補。必要なのは数百時間なので余裕が大きい

---

## 10. 実装と実行手順（2026-06-12 実装完了）

### 10.1 実装したスクリプト

| スクリプト | 役割 |
|---|---|
| `my_main/scripts/duplex_data_pipeline.py` | J-CHAT shar → 時刻整列データセット（asr / dialog 両モード） |
| `my_main/scripts/duplex_finetune.py` | 時刻整列インターリーブ形式での LoRA 学習（projection 凍結、メトリクス記録付き） |
| `my_main/scripts/plot_metrics.py` | metrics.jsonl → 学習曲線 PNG（学習中でも実行可） |

実装上の確定事項:

- PAD トークン: Gemma 4 vocab に `<unusedN>` が存在しないため **`<mask>` (id=4)** を転用
- 音声境界トークン `<|audio>`=256000 / `<audio|>`=258883 をシーケンスに組み込み
- モード識別: `<|turn>asr\n`（Phase 1'）/ `<|turn>chat\n`（Phase 2）
- スロット実測: PAD 比率 83〜89%（発話密度に依存）。PAD/text 分離 loss を別々に記録
- 学習対象は LoRA のみ（全 decoder 層 q/k/v/o + MLP）。embed_audio は fp32 化のうえ凍結

### 10.2 実行手順

```bash
# 1) データ生成（Phase 1' 用 streaming ASR、約 5000 サンプル）
python3 my_main/scripts/duplex_data_pipeline.py \
    --mode asr --shards 8 --max-samples 5000 \
    --output-dir my_main/duplex_data/asr_train --verify

# 2) 学習（GPU は任意指定）
python3 my_main/scripts/duplex_finetune.py \
    --data   my_main/duplex_data/asr_train \
    --output my_main/duplex_lora \
    --gpu    1 \
    --epochs 3

# 3) 学習曲線（学習中でも実行可。my_main/duplex_lora/curves.png が生成される）
python3 my_main/scripts/plot_metrics.py my_main/duplex_lora/metrics.jsonl

# 4) Phase 2 用データ（並行生成可）
python3 my_main/scripts/duplex_data_pipeline.py \
    --mode dialog --shards 16 --max-samples 2000 \
    --output-dir my_main/duplex_data/dialog_train --verify
```

### 10.3 記録されるメトリクス（output/metrics.jsonl）

| フィールド | 内容 |
|---|---|
| split | train / val |
| loss | PAD 重み付き全体 loss（backward 対象） |
| loss_text / loss_pad | テキストスロット / PAD スロットの CE（分離計測） |
| acc_text / acc_pad | スロット種別ごとの argmax 一致率 |
| gnorm / lr | LoRA 勾配ノルム / 学習率 |
| overflow_tokens | スロット不足で配置できなかったトークン累計 |

判定の目安:
- `loss_text` が下がり `acc_text` が上がる → 転写内容の学習が進行
- `acc_pad` が高止まりしたまま `acc_text` が 0 付近 → 常時 PAD に崩壊（pad_loss_weight を下げる）
- val と train の乖離 → 過学習（データ追加 or epoch 削減）

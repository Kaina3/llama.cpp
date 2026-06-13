# Phase 1' オーバーフィットテスト分析

**日付**: 2026-06-12  
**スクリプト**: `my_main/scripts/duplex_finetune.py`  
**データ**: `my_main/duplex_data/asr_train`

---

## 1. 実施した実験

### 実験 A (pad-loss-weight=0.1, 100 samples, lr=1e-4)

```
acc_t: 0.000 → 0.011 (ep2) → 0.000 (ep5)
loss_text: 14.8 → 9.0 → 7.0
```

### 実験 B (pad-loss-weight=0.05, 500 samples, lr=2e-5)

```
acc_t: 0.000 → 0.017 (step35) → 0.000 (step40以降)
loss_text: 16.2 → 8.6 (ep1末) → 6.7 (ep5末)
loss_pad:  18.8 (初期) → 6.25 (step35) → 1.2 (step40以降)
```

---

## 2. 観察事実（実験 B のログより）

### 事実 1: loss_pad の初期値が 18.8（ランダム基準 12.5 より高い）

ベースモデルは `<mask>`（PAD として転用するトークン）を通常のテキスト生成でほぼ出力しない。
そのため `<mask>` の logit は事前学習で抑制されており、初期 CE loss が 12.5 を上回る。

### 事実 2: step35 → step40 で PAD 学習が急激に完了し、text 学習が崩壊

```
           step35       step40
loss_pad:   6.25   →    2.00   （大幅に改善）
loss_text:  8.21   →    8.30   （悪化）
acc_t:      0.017  →    0.001  （崩壊）
gnorm:      87     →    243    （大きなパラメータ更新）
```

step40 の gnorm=243 は PAD 学習の大きな勾配が走った証拠。
このタイミングで model が「常に PAD を出す」解に一気に収束している。

### 事実 3: 以降は loss_text が緩やかに下がるが acc_t は永続的に 0

loss_text: 8.30 → 7.98 → 7.66 → ... → 6.7  
acc_text: 0.000 のまま

exp(-6.7) ≈ 0.12% → 正解テキストトークンへの確率は 0.12% まで学習できているが、
PAD logit が常に最大なため argmax が 0 になる。

---

## 3. 根本原因: PAD 学習と text 学習の logit 競合

LoRA パラメータは全ポジション（PAD スロットも text スロットも）で共有される。

**PAD ポジションの学習で起きること**:
- `<mask>` の logit を引き上げる → softmax で他のトークンの確率が下がる
- この「下がり」は text ポジションにも波及する
- text ポジションでも `<mask>` logit が高くなり、正解テキストトークンが argmax になれない

**数値で確認（前回の raw logit 診断より）**:
```
t=3: <mask> logit=+28.8, 'この' logit=+24.4  (差 4.4 units)
t=4: <mask> logit=+28.8, 'この' logit=+25.5  (差 3.3 units)
```
PAD が常に 3〜11 unit 高いため、どの温度設定でも argmax は PAD になる。

**損失重み (pad_weight=0.05) を下げても解決しない理由**:  
PAD ポジションは 87% を占める。たとえ 1 サンプル当たりの重みが 0.05 でも、
絶対数が多いため累積勾配でテキスト学習を圧倒する。
また初期の loss_pad が 18.8（異常に高い）なので、PAD 学習の初期勾配は特に強烈。

---

## 4. 解決策: カリキュラム学習

PAD と text を同時に学習しようとするのが問題。段階を分ける。

### Stage A: テキスト専用学習（pad_weight=0.0）

PAD スロットへの損失を完全ゼロにして、テキストスロットだけで学習する。

```bash
python3 my_main/scripts/duplex_finetune.py \
    --data    my_main/duplex_data/asr_train \
    --output  my_main/duplex_lora_stage_a \
    --gpu     1 \
    --epochs  10 \
    --samples 1000 \
    --lr      5e-5 \
    --pad-loss-weight 0.0 \
    --log-steps 10
```

期待する挙動:
- loss_text が 12.5（ランダム）から ep3 で 5 台、ep10 で 3 台以下に収束
- acc_text が ep3 以降で 0.02〜0.1 まで上昇・安定
- loss_pad は計算されるが勾配に寄与しない（表示は目安として監視）

Stage A が成功すれば: モデルが「どの音声フレームでどのテキストを出すか」を
学習できることが確認できる。

### Stage B: PAD 学習の追加（Stage A チェックポイントから再学習）

Stage A のアダプタを base として、PAD 重みを戻す。

```bash
python3 my_main/scripts/duplex_finetune.py \
    --data    my_main/duplex_data/asr_train \
    --output  my_main/duplex_lora_stage_b \
    --resume  my_main/duplex_lora_stage_a \
    --gpu     1 \
    --epochs  3 \
    --lr      1e-5 \
    --pad-loss-weight 0.1
```

Stage B の目的: テキストが出せるようになったモデルに
「テキストを出すべきでないフレームは PAD を出す」を追加で学習させる。
Stage A で確立した text logit の優位を保ちながら PAD logit を調整する。

---

## 5. Stage A が失敗する場合の追加仮説

もし Stage A（pad_weight=0.0）でも loss_text が 6〜7 で詰まる場合:

- **embedding_projection が音声→テキスト対応を学習できていない**: Phase 1B でオーバーフィットした
  アダプタが embedding_projection の事前学習値を壊した可能性（凍結しているが、LoRA の影響で
  downstream タスクへの汎化が低下した可能性）。その場合はベースモデルから LoRA なしで
  テキストスロットの予測能力を診断する必要がある。

- **LoRA rank 16 が不足**: 時刻整列タスクは「どのフレームで何を出すか」という
  位置ごとの細かい制御が必要。rank 32 への引き上げを検討。

---

## 6. まとめ

| 試み | 結果 | 原因 |
|---|---|---|
| pad_weight=1.0 (旧) | acc_text≈0.001 で完全 PAD 崩壊 | PAD が 87% → 常に PAD 出力が最小損失解 |
| pad_weight=0.1 | 同上 | PAD 初期勾配が強すぎて text 学習を圧倒 |
| pad_weight=0.05 | 一時的に acc_t=0.017、その後 0 | step40 の PAD 収束で logit 競合が確定 |
| **pad_weight=0.0** | **未試験** → **Stage A として次に実施** | |

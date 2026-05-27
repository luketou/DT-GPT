# MIMIC DoRA Resume Job 卡住問題分析（2026-05-22）

## 簡短結論

目前看到的是兩個不同現象，不能混在一起判斷：

1. **小資料 smoke test 已證明 `seq_len=2048 + sdpa` 可以正常訓練。**
2. **full job 先前看似卡在訓練開始後，但最新 diagnostic job `37655` 顯示 full dataset 的 DF 轉文字前處理本身非常慢，而且仍在持續前進，不能直接判定為死鎖。**

也就是說，目前最保守的判斷是：

> 目前真正確定的瓶頸是 full dataset 的資料前處理非常慢；先前 full job 在 `Start training` 後長時間沒有 loss，仍需要等 diagnostic job `37655` 完成前處理並進入 training loop 後才能最後判斷是否真的卡在 DeepSpeed/NCCL/optimizer step。

---

## 已測試過的設定與結果

### 1. `seq_len=2048` 不搭配 SDPA 會 OOM

之前的 smoke test 使用 `seq_len=2048`、預設 eager attention 時出現 OOM。

因此 `2048` context 不能直接用預設 attention 跑。

### 2. `seq_len=2048 + sdpa` 的 smoke test 成功

smoke test 使用：

- `DTGPT_SEQ_MAX_LEN=2048`
- `DTGPT_ATTN_IMPLEMENTATION=sdpa`
- 小量訓練資料
- adapter fallback resume from `checkpoint-1395`

結果可以完成少量訓練 step，代表：

- checkpoint adapter 可以被載入；
- `2048 + sdpa` 在小資料下不會立即 OOM；
- 基本 training loop 是可運作的。

### 3. full job `37527` / `37604` 的現象

full job 使用 full dataset 時，曾出現以下現象：

- 資料 map / tokenize 完成；
- 載入 adapter checkpoint；
- 印出 `Start training`；
- 印出 `Training from adapter-initialized checkpoint ... for 2790 optimizer steps`；
- 之後長時間沒有 `loss`、`train_runtime`、`Finished run`。

`37604` 設定包含：

- `DTGPT_SEQ_MAX_LEN=2048`
- `DTGPT_ATTN_IMPLEMENTATION=sdpa`
- `grad_acc=8`
- `logging_steps=1`

因為 `logging_steps=1`，理論上第一個 optimizer/logging step 完成後就應該很快看到 loss。長時間沒有 loss，因此當時判斷它**疑似卡在訓練剛開始的第一個 step 附近**。

可能位置包括：

- DeepSpeed ZeRO 初始化後第一個 forward/backward；
- NCCL collective；
- optimizer state / parameter partition 初始化；
- adapter fallback resume 與 DeepSpeed 分散式訓練交互作用。

但這個判斷目前仍屬於「疑似」，因為最新 diagnostic 顯示 full-data 前處理也可能耗時很久。

---

## 最新 diagnostic job `37655` 的重要發現

為了避免盲目重送 full job，已取消 `37604`，改送 diagnostic job `37655`。

`37655` 的目的：

- 保留 full dataset；
- 仍使用 `seq_len=2048 + sdpa`；
- 使用 `grad_acc=8`、`logging_steps=1`；
- 將目標 step 設成只需要跑很少 optimizer steps，用來確認能不能進 training loop。

`37655` 啟動參數確認如下：

```text
Target max global step: 1397
Sequence max length: 2048
Attention implementation: sdpa
Sweep configs: 16,32,8,3,8e-6
Logging steps: 1
```

截至最新檢查，`37655` 還在跑，且 log 持續更新：

```text
Converting DFs to Strings: 20900 / 22406 elapsed=4092.9s
Converting DFs to Strings: 21000 / 22406 elapsed=4112.4s
Converting DFs to Strings: 21100 / 22406 elapsed=4132.0s
```

這表示：

- 它不是完全卡死；
- 它目前仍在 full dataset 的 DF 轉文字階段；
- full dataset 前處理約每 100 筆要 20 秒左右；
- 這個階段本身會花超過一小時。

因此，若只看「很久沒有 loss」，但 job 還沒真正進入 training loop，就會誤判成訓練卡住。

---

## 目前卡在哪個環節？

### 已確定卡/慢的環節

目前最新 diagnostic 明確顯示，full run 很大的時間花在：

```text
pipeline.DFConversionHelpers - Converting DFs to Strings
```

也就是：

> MIMIC full dataframe 轉成文字 prompt / sequence 的前處理階段。

這不是 GPU training 本身，而是 CPU / pandas / dataframe-to-text conversion 相關流程。

### 仍待確認的環節

先前 `37604` 顯示在 `Start training` 後長時間沒有 loss，所以仍然需要確認：

> full-data 前處理完成後，DeepSpeed 分散式訓練是否能真的跑出第一個 loss。

如果 `37655` 之後進入：

```text
Start training
Training from adapter-initialized checkpoint ...
```

然後 30–60 分鐘內仍然沒有任何 loss，才可以比較確定問題在 training loop，而不是前處理。

---

## 為什麼之前會看起來像卡住？

原因有三個：

1. **full dataset 前處理非常慢**
   - smoke test 只用少量資料，因此很快進 training；
   - full dataset 有 22406 筆 train records，DF 轉文字就可能超過一小時。

2. **log 裡 training / preprocessing 訊息容易混在一起**
   - 有些訊息是 rank 0 / rank 1 重複輸出；
   - NCCL 初始化訊息看起來像卡在 distributed；
   - 但最新 diagnostic 顯示至少有一段時間其實還在資料轉換。

3. **`logging_steps=1` 只能保證進入 training loop 後能快速看到 loss**
   - 如果還沒進 training loop，`logging_steps=1` 不會有任何幫助；
   - 因此要先確認 log 是否已經真的到 `Start training` 後，才能用「沒有 loss」判斷卡住。

---

## 目前最可能的問題排序

### 高可能性：full dataset 前處理過慢

證據：

- `37655` 正在持續輸出 `Converting DFs to Strings`；
- 進度從 15000、16000、17000 一直到 21100 / 22406；
- log timestamp 持續更新。

這代表現在至少有一個主要瓶頸是前處理，不是 GPU 訓練。

### 中可能性：進入 training 後 DeepSpeed / NCCL / adapter fallback 卡住

證據：

- `37604` 先前已經出現 `Start training` 後長時間沒有 loss；
- full job 使用 distributed DeepSpeed；
- checkpoint 不是完整 DeepSpeed checkpoint，而是 adapter checkpoint，因此程式使用 adapter fallback resume；
- 這種組合可能在 ZeRO partition / optimizer init / first backward 時出問題。

但目前需要等 `37655` 進入 training loop 後再確認。

### 低可能性：`seq_len=2048` 本身不行

證據：

- `seq_len=2048 + eager` 會 OOM；
- 但 `seq_len=2048 + sdpa` smoke test 已成功。

所以 `2048` 不是絕對不能用，關鍵是必須搭配 `sdpa`，且 full job 還要另外處理前處理和 DeepSpeed 問題。

---

## 建議接下來怎麼做

### 1. 繼續觀察 `37655`，不要現在取消

因為 `37655` 目前有持續進度，不是死鎖。

建議觀察到它完成：

```text
Converting DFs to Strings: 22406 / 22406
```

並確認後續是否進入：

```text
Start training
```

### 2. 若進入 training 後 30–60 分鐘仍無 loss，再判定 training 卡住

判斷標準：

- 如果看到 `loss`：代表 training loop 可跑，之前主要是前處理慢。
- 如果 `Start training` 後 30–60 分鐘仍無 `loss`：代表卡點很可能在 DeepSpeed/NCCL/adapter fallback 的第一個 training step。

### 3. 後續優化方向

若確認主要瓶頸是前處理，應該優先做：

- cache DF-to-text 的結果；
- 避免每次重送 job 都重新轉 full dataset；
- 將前處理拆成獨立 job；
- 將轉好的 HuggingFace dataset 存成固定路徑並重用。

若確認 training loop 也卡住，則再處理：

- 測 DeepSpeed off / single GPU 小 batch 是否可跑；
- 測 distributed + DeepSpeed 但不用 adapter fallback resume；
- 嘗試保存真正的 DeepSpeed checkpoint，而不是只用 adapter checkpoint resume；
- 檢查 NCCL device mapping warning：

```text
Guessing device ID based on global rank. This can cause a hang if rank to GPU mapping is heterogeneous.
```

---

## 一句話總結

目前最明確的問題不是 `seq_len=2048`，也不是 `grad_acc=8`，而是：

> full MIMIC job 在正式訓練前的 dataframe-to-text 前處理非常慢；先前 full job 也疑似在進入 DeepSpeed training first step 後無 loss，但這需要等 diagnostic job `37655` 完成前處理後才能最終確認。

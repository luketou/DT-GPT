# DT-GPT DoRA 煙霧測試實驗日誌 (Smoke Test Log)

本文件僅保留**實驗目標**、**參數設定**以及**預估目標數值**，供您快速查閱與填寫。

---

## 1. 實驗目標 (Goal)
驗證將微調架構從全參數微調（FFT）變更為 DoRA，並將學習率調大至 `1e-4` 後，梯度流動（grad_norm）與訓練收斂（Loss 下降）是否正常，以確保後續長訓練的穩定性。

---

## 2. 超參數與執行指令設定 (Parameter Settings)

在命令行中執行以下指令進行短期煙霧測試：

```bash
conda activate dtgpt-unsloth

python 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py \
  --learning-rate 1e-4 \
  --train-batch-size 1 \
  --validation-batch-size 10 \
  --train-max-patients 100 \
  --validation-max-patients 50 \
  --num-train-epochs 1 \
  --use-dora
```

* **核心參數設定說明**：
  * `--learning-rate 1e-4`：加大後的學習率，解決更新停滯問題。
  * `--use-dora`：開啟 DoRA 架構。
  * `--train-max-patients 100` / `--validation-max-patients 50`：限制患者數量以加速測試。

---

## 3. 預期目標數值 (Target Values)

您需要觀察訓練日誌中的輸出，對照以下目標數值進行判定：

| 監控指標 | 預期目標數值 | 狀態評估說明 | 實際觀察數值 (填寫) |
| :--- | :--- | :--- | :--- |
| **`grad_norm`** | **`0.5 ~ 3.0`** | **正常更新**。若 $<0.3$ 代表更新緩慢；若 $>10.0$ 則有梯度爆炸風險。 | |
| **`loss`** | **持續下降** | 確保模型正在收斂。若 Loss 曲線平坦或出現 `NaN` 則屬異常。 | |
| **`eval_loss`** | **持續下降** | 確保模型在驗證集上同樣收斂，數值應低於前期訓練。 | |
| **`R²` (Validation)** | **出現上升趨勢** | 數值時序對齊的初步特徵（應開始擺脫負數並往上回升）。 | |

---

## 4. 實際運行記錄與診斷

* **啟動時間**：`[ ]`
* **運行總步數 (Steps)**：`[ ]`
* **診斷結論**：
  * `[ ]` **成功**（指標達標，Loss 下降）$\rightarrow$ *後續行動：移除限制器跑完整數據*。
  * `[ ]` **不穩定**（梯度爆炸或 NaN）$\rightarrow$ *後續行動：調小學習率至 `5e-5`*。
  * `[ ]` **停滯**（梯度極低，Loss 平躺）$\rightarrow$ *後續行動：調大學習率至 `2e-4`*。

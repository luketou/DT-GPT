# 步驟級縮放 MAE (Step-Level Scaled MAE)、患者級 MAE (Patient-Level MAE) 以及 DT-GPT 的計算方式

本說明旨在解釋 **步驟級縮放 MAE (step-level scaled MAE)** 與 **患者級 MAE (patient-level MAE)** 之間的差異，以及它們與 DT-GPT MIMIC 評估工作流的關係。

---

## 1. 什麼是步驟級縮放 MAE (Step-Level Scaled MAE)？

`step-level`（步驟級）意味著我們**單獨評估每個時間步**。

對於某個變數，令：

- $y_i$ = 第 $i$ 步的真實值
- $\hat{y}_i$ = 第 $i$ 步的預測值
- $N$ = 有效的步驟級行數（樣本數）
- $\sigma_y$ = 該變數真實值的標準差

則步驟級 MAE 為：

$$
\text{MAE} = \frac{1}{N}\sum_{i=1}^{N} |y_i - \hat{y}_i|
$$

**縮放後 MAE (scaled MAE)** 為：

$$
\text{scaled MAE} = \frac{\text{MAE}}{\sigma_y}
$$

等價於：

$$
\text{scaled MAE} = \frac{1}{N}\sum_{i=1}^{N}\frac{|y_i - \hat{y}_i|}{\sigma_y}
$$

因此，縮放後 MAE 衡量的是**相對於該變數本身自然波動範圍（自然分佈）的預測誤差**。

---

## 2. 為什麼要對 MAE 進行縮放？

不同的臨床變數具有非常不同的數值範圍（量綱）：

- 呼吸頻率 (Respiratory Rate)：大約在 10–30 之間
- 血氧飽和度 (SpO2)：大約在 90–100 之間
- 鎂離子 (Magnesium)：大約在 1–3 之間

因此，原始的 MAE 值無法直接跨變數進行比較。

透過除以 $\sigma_y$ 進行縮放，可以使誤差無因次化（dimensionless），從而能夠在不同變數之間進行比較。

---

## 3. 患者級 MAE (Patient-Level MAE) vs 步驟級 MAE (Step-Level MAE)

### 患者級平均 (Patient-level averaging)

首先計算每個患者所有時間步的平均值：

$$
\overline{y}_p = \frac{1}{T_p}\sum_{t=1}^{T_p} y_{p,t}
\qquad
\overline{\hat{y}}_p = \frac{1}{T_p}\sum_{t=1}^{T_p} \hat{y}_{p,t}
$$

接著，計算這些患者平均值之間的 MAE。

這就是 **患者級 MAE (patient-level MAE)**。

### 步驟級評估 (Step-level evaluation)

不事先進行平均，而是直接評估每一個時間步：

$$
|y_i - \hat{y}_i|
$$

然後對所有有效的時間步計算平均。

這就是 **步驟級 MAE (step-level MAE)**。

### 主要差異

- **步驟級 MAE** 保留了所有的時間點。
- **患者級 MAE** 將每個患者壓縮為單一的平均值。

論文中主要採用的 ICU `scaled MAE` 是一個**步驟級**的指標。

---

## 4. 這與 DT-GPT 程式庫的關係

在此程式庫中，`MetricManager.mae()` 計算的是：

$$
\text{MAE} = \text{已儲存評估行上的平均絕對誤差 (mean absolute error)}
$$

因此，如果儲存的目標值 (targets) 與預測值 (predictions) 已經是歸一化/縮放後（normalized / scaled）的資料框數值，那麼所報告的 MAE 實際上就是**步驟級縮放 MAE (scaled step-level MAE)**。

相關的實作程式碼位於：

- [pipeline/MetricManager.py](file:///home/r15543056/trajectory_forecast/DT-GPT/pipeline/MetricManager.py)

具體而言：

- `mae()` 使用了 `sklearn.metrics.mean_absolute_error`
- `all_numeric_columns mae overall` 是**所有數值變數的未加權平均值**

---

## 5. 步驟級縮放 MAE 公式總結

對於單一變數：

$$
\text{scaled MAE}_j
=
\frac{1}{N_j}\sum_{i=1}^{N_j}\frac{|y_{i,j} - \hat{y}_{i,j}|}{\sigma_j}
$$

其中：

- $j$ = 變數索引
- $i$ = 步驟索引
- $N_j$ = 該變數有效步驟的行數
- $\sigma_j$ = 經過異常值過濾後，真實值的標準差

---

## 6. DT-GPT MIMIC 評估的重要注意事項

1. 如果您想要得到論文風格的 ICU MAE，**請不要先對患者進行平均**。
2. 應用與繪圖/評估相同的異常值過濾器，例如 `abs(target) > 1000.0`。
3. 報告每個變數的縮放 MAE，以及所有變數的平均值。
4. 如果要與論文進行對比，請確保使用的是相同的歸一化方式以及相同的資料劃分 (split)。

---

## 7. 目前檢查點 (Checkpoint) 的實際範例

使用 checkpoint 5602 的 8 個分片 (shards) 評估 CSV 檔，並過濾 `abs(target) > 1000.0`：

| 變數 | 步驟行數 | 步驟級縮放 MAE |
|---|---:|---:|
| 呼吸頻率 (Respiratory Rate) | 51,169 | 0.6448 |
| 血氧飽和度 (SpO2) | 50,578 | 0.5766 |
| 鎂 (Magnesium) | 3,461 | 0.4578 |
| 變數間的未加權平均值 | — | 0.5598 |

此數值是用於與論文中的 ICU 縮放 MAE 進行對比的，而非患者級繪圖 MAE。

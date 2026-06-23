# DT-GPT 論文重現：評分標準與圖表說明

## 目的

這份筆記用來說明如果要重現 DT-GPT 論文結果，應該如何計算評分標準，以及應該準備哪些圖表來解釋資料刪除、過濾、評估與結果對齊。

重點是：論文主表的 MIMIC-IV ICU 結果不是患者級平均，而是患者-時間步-變數層級的標準化誤差，先跨患者與時間步平均，再跨變數做未加權平均。

## 一、論文結果重現：評分標準與計算方式

### 先對齊資料與預測單位

要重現論文主結果，先確認評估資料表的最小單位是「患者-時間步-變數」，而不是患者平均值。對 MIMIC-IV ICU 來說，輸入是 ICU 前 24 小時，輸出是未來 24 小時；三個目標變數是血氧飽和度、呼吸頻率和鎂。

論文第 2.6 節描述的主指標流程是：

1. 先標準化。
2. 計算每個樣本與時間步的預測值和真實值成對誤差。
3. 跨患者與時間點平均。
4. 最後跨變數平均。

因此，重現時要保存以下欄位：

| 欄位 | 用途 |
|---|---|
| `patient_id` | 確保資料切分是患者層級，避免同一患者跨 train/test 洩漏 |
| `time_step` | 保留逐小時或逐週評估單位 |
| `variable` | 用於逐變數計算 MAE 與 Spearman Corr |
| `y_true` | 真實目標值 |
| `y_pred` | 模型最終預測值；DT-GPT 論文是 30 次生成後逐時間點平均 |
| `valid_mask` | 排除缺失、無效輸出、解析失敗或被異常值規則排除的格子 |

### 標準化 MAE / sMAE

論文表中的 MAE 是標準化後的平均絕對誤差，可稱為 standardized MAE 或 scaled MAE。若 `y_true` 和 `y_pred` 已經在同一個標準化尺度上，單一變數 $j$ 的計算方式是：

$$
\mathrm{sMAE}_j =
\frac{1}{N_j}
\sum_{(p,t)\in \Omega_j}
\left| z_{p,t,j} - \hat{z}_{p,t,j} \right|
$$

其中：

- $p$ 是患者。
- $t$ 是預測時間步。
- $j$ 是變數。
- $\Omega_j$ 是變數 $j$ 的有效患者-時間步集合。
- $N_j = |\Omega_j|$。
- $z_{p,t,j}$ 和 $\hat{z}_{p,t,j}$ 分別是真實值與預測值在標準化尺度上的數值。

若資料仍在原始臨床單位，則要先用同一個 scaler 標準化，或等價地除以該變數的標準差：

$$
\mathrm{sMAE}_j =
\frac{1}{N_j}
\sum_{(p,t)\in \Omega_j}
\frac{\left| y_{p,t,j} - \hat{y}_{p,t,j} \right|}{\sigma_j}
$$

這裡最容易出錯的是 $\sigma_j$ 的來源。論文正文沒有完全展開 scaler 是用 train set、過濾後 train set，還是 pipeline 產物；但若要避免資料洩漏，實作上應使用訓練集擬合 scaler，再套到 validation/test。若目標是對齊論文表格，必須使用與原 pipeline 一致的 scaler 和異常值處理。

### 變數平均：論文主表應用 macro-average

論文第 2.6 節說「先跨患者與時間點平均，再跨變數平均」。因此主表平均值應該是：

$$
\mathrm{sMAE}_{\mathrm{avg}} =
\frac{1}{D}\sum_{j=1}^{D}\mathrm{sMAE}_j
$$

MIMIC-IV ICU 的 DT-GPT 附錄表 A8.2 可直接驗算：

$$
\frac{0.505 + 0.636 + 0.635}{3} = 0.592
$$

正文四捨五入後報告為約 $0.59 \pm 0.03$。因此重現論文時，應先輸出每個變數的 sMAE，再做未加權平均。不要把所有變數的所有有效格子直接攤平成一個大向量後平均；那是 micro-average，會讓觀測較多的變數權重更大。

| 平均方式 | 公式 | 是否對齊論文主表 |
|---|---|---|
| 每變數先平均，再未加權平均 | $\frac{1}{D}\sum_j \mathrm{sMAE}_j$ | 是 |
| 所有有效格子展平後平均 | $\frac{1}{\sum_j N_j}\sum_j\sum_i |e_{i,j}|$ | 不一定 |
| 每患者先平均，再跨患者平均 | $\frac{1}{P}\sum_p \mathrm{MAE}_p$ | 否 |

### Spearman Corr

附錄表 A8.1 和 A8.2 的 `Corr.` 指的是 Spearman correlation，越高越好。對單一變數 $j$，先收集所有有效患者-時間步格子的真實值與預測值：

$$
\{(z_{i,j}, \hat{z}_{i,j})\}_{i=1}^{N_j}
$$

再計算 Spearman 相關：

$$
\rho_j =
\mathrm{corr}_{\mathrm{Pearson}}
\left(
\mathrm{rank}(z_{\cdot,j}),
\mathrm{rank}(\hat{z}_{\cdot,j})
\right)
$$

Spearman Corr 衡量的是預測值和真實值的單調關係，不要求數值尺度完全線性一致。重現表格時應逐變數計算，再回報每個變數的 Corr；若要做平均，也應明確標註是未加權平均。

### Figure 3c 的相關矩陣 $R^2$

論文中的 $R^2=0.98$ 和 $R^2=0.99$ 不是逐樣本迴歸分數，而是「變數間相關結構」的對齊程度。計算方式應理解為：

1. 對真實測試集計算目標變數兩兩相關，得到 ground-truth correlation matrix。
2. 對模型預測值計算同一組變數的兩兩相關，得到 prediction correlation matrix。
3. 取矩陣上三角的變數對，例如 $(j,k)$，形成兩個向量：

$$
\mathbf{c}^{\mathrm{true}} =
\left[
\rho^{\mathrm{true}}_{1,2},
\rho^{\mathrm{true}}_{1,3},
\ldots
\right]
$$

$$
\mathbf{c}^{\mathrm{pred}} =
\left[
\rho^{\mathrm{pred}}_{1,2},
\rho^{\mathrm{pred}}_{1,3},
\ldots
\right]
$$

4. 對這兩個向量做線性對齊程度評估：

$$
R^2_{\mathrm{corr}} =
1 -
\frac{
\sum_m
\left(c^{\mathrm{true}}_m - c^{\mathrm{pred}}_m\right)^2
}{
\sum_m
\left(c^{\mathrm{true}}_m - \bar{c}^{\mathrm{true}}\right)^2
}
$$

等價實作也可能是對 Figure 3c 散點做線性回歸後取 $R^2$。重點是：這個 $R^2$ 衡量的是「預測保留變數間相關結構的程度」，不是單一變數預測的 $R^2$，也不是 patient-level 的 $R^2$。

MIMIC-IV 只有三個目標變數，因此兩兩相關只有三個點：血氧飽和度-呼吸頻率、血氧飽和度-鎂、呼吸頻率-鎂。這會讓 ICU 的相關矩陣 $R^2$ 很容易受單一變數對影響；解讀時要比 NSCLC 更保守。

### 標準誤

論文主結果寫成 $0.59 \pm 0.03$ 這種形式，並說 error bars 是模型在所有變數上聚合後的 standard error。可重現做法是先得到每個變數的指標值，再計算變數層級標準誤：

$$
\mathrm{SE} =
\frac{
\mathrm{std}\left(s_1, s_2, \ldots, s_D\right)
}{
\sqrt{D}
}
$$

其中 $s_j$ 是變數 $j$ 的 sMAE 或 Corr。若改成 patient-level bootstrap 或 time-step bootstrap，數字會不同，不能直接對齊論文 error bar。

### 建議的重現流程

1. 固定 patient-level test split，確認沒有患者洩漏。
2. 套用與論文一致的異常值處理。論文寫的是三標準差過濾，加第二輪 clipping；本地 `abs(target)>1000` 只能視為額外防護，不能當作論文原方法。
3. 用訓練集或原 pipeline scaler 對每個變數做標準化。
4. DT-GPT 對每個樣本生成 30 條軌跡，對同一患者、時間步、變數取平均作為最終預測。
5. 對每個變數計算 sMAE 和 Spearman Corr。
6. 對變數做未加權平均，得到主表 `Avg. MAE` 和 `Avg. Corr.`。
7. 對真實值和預測值分別計算變數間相關矩陣，再算 Figure 3c 的 correlation-structure $R^2$。
8. 回報每個變數有效格子數 $N_j$，因為缺失或過濾不同會直接改變 sMAE。

### Python 參考實作

```python
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

TARGETS = ["magnesium", "respiratory_rate", "spo2"]

def per_variable_metrics(df, targets=TARGETS):
    rows = []
    for var in targets:
        sub = df[(df["variable"] == var) & (df["valid_mask"])].copy()
        y = sub["y_true_z"].to_numpy()
        yhat = sub["y_pred_z"].to_numpy()

        smae = np.mean(np.abs(y - yhat))
        corr = spearmanr(y, yhat, nan_policy="omit").statistic

        rows.append({
            "variable": var,
            "n": len(sub),
            "smae": smae,
            "spearman_corr": corr,
        })

    out = pd.DataFrame(rows)
    summary = {
        "avg_smae_macro": out["smae"].mean(),
        "avg_corr_macro": out["spearman_corr"].mean(),
        "smae_se_by_variable": out["smae"].std(ddof=1) / np.sqrt(len(out)),
        "corr_se_by_variable": out["spearman_corr"].std(ddof=1) / np.sqrt(len(out)),
    }
    return out, summary

def correlation_structure_r2(df, targets=TARGETS):
    true_wide = (
        df[df["valid_mask"]]
        .pivot_table(index=["patient_id", "time_step"], columns="variable", values="y_true_z")
    )
    pred_wide = (
        df[df["valid_mask"]]
        .pivot_table(index=["patient_id", "time_step"], columns="variable", values="y_pred_z")
    )

    true_wide = true_wide[targets]
    pred_wide = pred_wide[targets]

    true_corr = true_wide.corr(method="spearman")
    pred_corr = pred_wide.corr(method="spearman")

    pairs_true = []
    pairs_pred = []
    for i, a in enumerate(targets):
        for b in targets[i + 1:]:
            if pd.notna(true_corr.loc[a, b]) and pd.notna(pred_corr.loc[a, b]):
                pairs_true.append(true_corr.loc[a, b])
                pairs_pred.append(pred_corr.loc[a, b])

    return r2_score(pairs_true, pairs_pred), true_corr, pred_corr
```

### 對齊論文時的檢查清單

| 檢查項 | 必須確認 |
|---|---|
| split | 是否為 patient-level 80/10/10，且 test patients 與 train patients 不重疊 |
| scaler | 標準化統計量是否來自與論文相同的 pipeline |
| outlier | 是否使用三標準差過濾和 clipping，而不只是 `abs(target)>1000` |
| aggregation | 是否 30 samples mean，而不是取最佳樣本或單次生成 |
| MAE average | 是否每變數先算 sMAE，再跨變數未加權平均 |
| Corr | 是否逐變數計算 Spearman correlation |
| $R^2$ | 是否計算變數間相關矩陣的對齊，而不是單變數 prediction $R^2$ |
| valid rows | 是否輸出每變數有效患者-時間步數，方便檢查缺失與過濾差異 |

## 二、建議圖表：用來輔助說明重現結果

### 1. Data Filtering Flowchart

用來說明資料從 raw predictions 到 final evaluation rows 的過濾流程。

建議流程：

```text
Raw prediction rows
→ parse success only
→ valid target / prediction pairs
→ remove missing targets
→ remove invalid generated values
→ apply outlier filter
→ keep patient-time-variable rows
→ compute metrics
```

這張圖最適合解釋 `abs(target)>1000`、三標準差過濾、clipping、缺失值刪除到底發生在哪一步。

### 2. Row Count Funnel Table

用表格呈現每一步刪掉多少資料。

| Step | Rows kept | Rows removed | Removed % | Reason |
|---|---:|---:|---:|---|
| Raw rows | 120,000 | - | - | all predictions |
| Parse valid | 118,500 | 1,500 | 1.25% | invalid model output |
| Non-missing target | 105,000 | 13,500 | 11.4% | missing target |
| Outlier filtered | 104,200 | 800 | 0.76% | target outlier |
| Final eval rows | 104,200 | - | - | used for metrics |

這張表能說明最後 evaluation set 是怎麼來的，也能讓讀者判斷刪除資料是否過多。

### 3. Per-Variable Valid Count Bar Plot

顯示每個變數最後有效時間步數。

| Variable | Valid rows |
|---|---:|
| Respiratory Rate | 51,169 |
| SpO2 | 50,578 |
| Magnesium | 3,461 |

這張圖很重要，因為 Magnesium 的有效 row 可能遠少於其他變數。它能解釋為什麼 macro-average 和 micro-average 會不同。

### 4. Per-Variable sMAE Bar Plot

呈現每個變數的 standardized MAE。

| Variable | sMAE |
|---|---:|
| Magnesium | 0.4578 |
| Respiratory Rate | 0.6448 |
| SpO2 | 0.5766 |
| Macro Avg | 0.5598 |

建議在圖中加一條 horizontal line 表示 macro-average。這張圖是重現結果的核心摘要。

### 5. Paper vs Reproduction Comparison Table

直接對照論文 Table A8.2 和你的重現結果。

| Variable | Paper DT-GPT | Reproduction | Difference |
|---|---:|---:|---:|
| Magnesium | 0.505 | 0.4578 | -0.0472 |
| Respiratory Rate | 0.636 | 0.6448 | +0.0088 |
| O2 Saturation | 0.635 | 0.5766 | -0.0584 |
| Macro Avg | 0.592 | 0.5598 | -0.0322 |

這張表最適合回答「是否重現到論文」。若數字不同，應在旁邊註明可能原因：split、scaler、outlier filter、有效 row 數、checkpoint 或 aggregation 不同。

### 6. Prediction vs Ground Truth Scatter / Hexbin

每個變數各一張，x 軸是真實值，y 軸是預測值。

用途：

- 看模型是否系統性高估或低估。
- 看 outlier filter 是否合理。
- 看標準化後資料是否集中在合理範圍。

如果資料很多，用 hexbin 比 scatter 更清楚。

### 7. Error Distribution Histogram / Violin Plot

每個變數畫 $\left|y_{\mathrm{true},z} - y_{\mathrm{pred},z}\right|$ 的分布。

用途：

- 看 MAE 是由多數樣本穩定貢獻，還是被少數大錯誤拉高。
- 對應論文 Fig. 3h 的 error distribution 討論。
- 檢查刪除 outlier 前後，誤差分布是否合理。

### 8. Error by Forecast Horizon Line Plot

x 軸是 forecast time step，例如 ICU 未來第 1 到 24 小時；y 軸是 sMAE。

| Hour | Respiratory Rate sMAE | SpO2 sMAE | Magnesium sMAE |
|---:|---:|---:|---:|
| 1 | ... | ... | ... |
| 2 | ... | ... | ... |
| ... | ... | ... | ... |
| 24 | ... | ... | ... |

這張圖對 DT-GPT 很有用，因為論文 Fig. 3d/e 也強調不同 forecast time point 的 MAE。

### 9. Correlation Matrix Heatmap

分別畫：

- Ground truth correlation matrix
- Prediction correlation matrix
- Difference matrix

對 MIMIC 只有三個變數，所以 heatmap 很小，但仍然能說明 Figure 3c 的 $R^2$ 是怎麼來的。

### 10. Correlation Structure Scatter

x 軸是真實資料的變數對相關係數，y 軸是預測資料的變數對相關係數。

```text
x = corr_true(variable_i, variable_j)
y = corr_pred(variable_i, variable_j)
```

再標註 correlation-structure $R^2$。這就是論文 Fig. 3c 的核心圖。

## 三、最小重現報告圖表組合

如果只做一份簡潔但完整的重現報告，建議至少包含：

1. Data filtering flowchart
2. Row count funnel table
3. Per-variable valid count bar plot
4. Paper vs reproduction comparison table
5. Per-variable sMAE bar plot
6. Error by forecast horizon line plot
7. Correlation matrix heatmap 或 correlation structure scatter

這 7 個圖表足夠回答：

- 資料怎麼刪？
- 每一步剩多少？
- 每個變數有效資料量是否均衡？
- 每個變數的誤差是多少？
- 和論文差多少？
- 誤差是否隨預測時間變長而變差？
- 變數間相關結構有沒有被重現？

## 四、建議輸出檔案

重現實驗後，建議至少輸出以下檔案：

| 檔案 | 內容 |
|---|---|
| `eval_rows_filtered.csv` | 最終進入 metric 的患者-時間步-變數列 |
| `filtering_funnel.csv` | 每一步刪除與保留 row 數 |
| `per_variable_metrics.csv` | 每個變數的有效 row、sMAE、Spearman Corr |
| `summary_metrics.json` | macro-average sMAE、macro-average Corr、SE、correlation-structure R2 |
| `paper_vs_reproduction.csv` | 論文數值與重現數值的差異 |
| `figures/` | 所有重現圖表 |

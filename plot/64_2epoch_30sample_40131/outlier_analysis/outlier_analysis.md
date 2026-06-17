# Outlier 刪除方法圖表總覽

本檔案整理 job `40131` 的 outlier deletion analysis。這裡最重要的是把三種情況分開：

- **保留下來的有效資料**：有 target，也有 prediction，最後有拿來算 metric。
- **真的被 outlier rule 刪掉的資料**：原本是有效資料，但因為超出規則門檻被移除。
- **missing / invalid 資料**：一開始就沒有完整 target 或 prediction，所以本來就不能拿來算該變數的 metric。

最重要結論：

> Magnesium 看起來少了九萬多筆，不是因為 IQR3 刪掉九萬多筆，而是因為 Magnesium 本來就有大量 missing / invalid row。  
> `target_iqr3` 對 Magnesium 真正刪掉的有效資料只有 **39 筆**。

---

## 圖 1：整體資料去向

![Deletion accounting](outlier_removal_accounting_40131_30sample.png)

這張圖把每個變數、每種刪除方法下的資料分成三類：

- 藍色：保留下來的有效資料
- 橘色：真的被 outlier rule 刪掉的有效資料
- 灰色：missing / invalid，本來就不能拿來算 metric

怎麼看：

- Respiratory Rate 和 SpO2 大部分資料都是藍色，代表有效資料很多。
- Magnesium 大部分是灰色，代表大多數 row 沒有 Magnesium 的有效 target/prediction。
- Magnesium 的橘色很小，代表 outlier rule 並沒有大量刪掉 Magnesium 的有效資料。

白話解釋：

> 這張圖是在說：「資料不見了」不一定等於「被 outlier rule 刪掉」。  
> Magnesium 的主要問題是 missing 多，不是 IQR3 刪太多。

---

## 圖 2：只看真正被 outlier rule 刪掉的資料

![Actual removed counts](outlier_removed_counts_only_40131_30sample.png)

這張圖把 missing / invalid row 排除，只看每種方法真正刪掉多少有效資料。

| 方法 | Respiratory Rate | SpO2 | Magnesium |
|---|---:|---:|---:|
| `target_abs_le_1000` | 1 | 1 | 0 |
| `target_iqr3` | 114 | 292 | 39 |
| `target_and_pred_iqr3` | 651 | 404 | 146 |
| `target_p1_99` | 1861 | 857 | 123 |
| `target_and_pred_p1_99` | 3531 | 2486 | 213 |

怎麼看：

- `target_abs_le_1000` 很保守，只刪掉明顯不可能的極端 target。
- `target_iqr3` 會刪多一點，但仍然不是大量刪除。
- `target_and_pred_iqr3` 會再多刪一些，因為它連 prediction 太極端也刪。
- `target_p1_99` 和 `target_and_pred_p1_99` 最嚴格，刪最多。

白話解釋：

> 這張圖才是真正回答「哪個方法刪掉多少資料」。  
> 如果要做正式主結果，`target_abs_le_1000` 比較保守；IQR3 和 percentile trimming 比較適合當 sensitivity analysis。

---

## 圖 3：不同 rule 的 cutoff 切在哪裡

![Target thresholds](outlier_target_distribution_thresholds_40131_30sample.png)

這張圖顯示每個變數的 target 分布，以及不同 target-based rule 的刪除界線。

圖中的直線代表：

- `IQR3`：用 Q1/Q3 和 3 倍 IQR 算出的統計門檻。
- `p0.1-99.9`：只保留第 0.1 到第 99.9 百分位之間的資料。
- `p1-99`：只保留第 1 到第 99 百分位之間的資料，更嚴格。

怎麼看：

- 線越靠近中間，代表刪得越嚴格。
- 線越靠近兩端，代表刪得越保守。
- 對 ICU 資料來說，極端值不一定是錯，可能是真實重症病人，所以不能只靠統計門檻決定正式刪除。

白話解釋：

> 這張圖是在看每個刪除方法的「刀」切在哪裡。  
> p1-99 切得比較靠近中間，所以刪比較多；IQR3 通常比較保守，但仍可能刪掉臨床上真實的極端病人。

---

## 總結

可以這樣解釋：

> 我們把資料分成三種：保留、真的被 outlier rule 刪掉、以及原本就 missing/invalid。  
> 圖表顯示，Magnesium 少很多資料主要是因為 missing，不是因為 IQR3 刪掉九萬多筆。  
> 如果只看真正被 outlier rule 刪掉的有效資料，`target_iqr3` 對 Magnesium 只刪 39 筆。  
> 因此，IQR3 可以當作敏感度分析，但正式主結果最好使用更保守、可被臨床合理性支持的 outlier rule，例如只刪明顯不可能的 target values。

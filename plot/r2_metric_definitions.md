# R² 指標定義：當前繪圖 vs DT-GPT 論文

本說明記錄了目前在本地患者平均繪圖中使用的 $R^2$ 與 DT-GPT 論文中報告的用於變數間相關性保持的 $R^2$ 之間的差異。

---

## 1. 當前程式碼 / 繪圖的 $R^2$

目前繪圖中的 $R^2$ 是**患者級的真實值與預測值之間的 $R^2$** (patient-level true-vs-predicted $R^2$)。

它回答了以下問題：

> 模型能否預測出每個患者單一變數的平均生理基線？

### 步驟 A：計算每個患者的平均值

對於患者 $p$，計算其所有有效時間步的平均值：

$$
\overline{y}_p = \frac{1}{T_p}\sum_{t=1}^{T_p} y_{p,t}
$$

$$
\overline{\hat{y}}_p = \frac{1}{T_p}\sum_{t=1}^{T_p} \hat{y}_{p,t}
$$

其中：

- $T_p$ = 患者 $p$ 的有效時間步數
- $y_{p,t}$ = 患者 $p$ 在時間步 $t$ 的真實值
- $\hat{y}_{p,t}$ = 患者 $p$ 在時間步 $t$ 的預測值

### 步驟 B：計算跨患者的 $R^2$

$$
R^2_{patient}
=
1 -
\frac{
\sum_p (\overline{y}_p - \overline{\hat{y}}_p)^2
}{
\sum_p (\overline{y}_p - \overline{y})^2
}
$$

其中：

$$
\overline{y} = \frac{1}{N}\sum_p \overline{y}_p
$$

這與 `sklearn.metrics.r2_score` 計算的 $R^2$ 相同，但它應用於患者平均後的真實值與預測值。

### 若進行 Z 分數標準化 (z-score standardized)

若對患者平均值進行標準化：

$$
mt\_z_p = \frac{\overline{y}_p - \mu_{\overline{y}}}{\sigma_{\overline{y}}}
$$

$$
mp\_z_p = \frac{\overline{\hat{y}}_p - \mu_{\overline{y}}}{\sigma_{\overline{y}}}
$$

則：

$$
R^2_{patient}
=
1 -
\frac{
\sum_p (mt\_z_p - mp\_z_p)^2
}{
\sum_p (mt\_z_p - \overline{mt\_z})^2
}
$$

此 $R^2$ 是**針對每個變數分別計算**的。

---

## 2. DT-GPT 論文的 $R^2$

DT-GPT 論文中報告的用於 MIMIC/ICU 變數間關係的 $R^2$ 是**相關性保持 $R^2$** (correlation preservation $R^2$)。

它回答了以下問題：

> 模型是否保持了臨床變數之間的相關性結構？

這與單一變數的真實值 vs 預測值 $R^2$ **並不相同**。

### 步驟 A：計算真實變數間的相關性

針對 MIMIC 的目標變數：

- 呼吸頻率 (Respiratory Rate)
- 血氧飽和度 (SpO2)
- 鎂 (Magnesium)

計算真實數據中的兩兩相關係數：

$$
C^{true}_{RR,SpO2} = corr(y_{RR}, y_{SpO2})
$$

$$
C^{true}_{RR,Mg} = corr(y_{RR}, y_{Mg})
$$

$$
C^{true}_{SpO2,Mg} = corr(y_{SpO2}, y_{Mg})
$$

將它們收集到一個向量中：

$$
c^{true}
=
[
C^{true}_{RR,SpO2},
C^{true}_{RR,Mg},
C^{true}_{SpO2,Mg}
]
$$

### 步驟 B：計算預測變數間的相關性

在預測的數據中計算相同的兩兩相關係數：

$$
C^{pred}_{RR,SpO2} = corr(\hat{y}_{RR}, \hat{y}_{SpO2})
$$

$$
C^{pred}_{RR,Mg} = corr(\hat{y}_{RR}, \hat{y}_{Mg})
$$

$$
C^{pred}_{SpO2,Mg} = corr(\hat{y}_{SpO2}, \hat{y}_{Mg})
$$

將它們收集到一個向量中：

$$
c^{pred}
=
[
C^{pred}_{RR,SpO2},
C^{pred}_{RR,Mg},
C^{pred}_{SpO2,Mg}
]
$$

### 步驟 C：計算相關性向量之間的 $R^2$

$$
R^2_{corr}
=
1 -
\frac{
\sum_k (c^{true}_k - c^{pred}_k)^2
}{
\sum_k (c^{true}_k - \overline{c^{true}})^2
}
$$

其中：

$$
\overline{c^{true}} = \frac{1}{K}\sum_k c^{true}_k
$$

此 $R^2$ 是針對相關係數計算的，而不是針對單個預測值計算的。

---

## 3. 主要差異

| 指標 | 當前繪圖 $R^2$ | DT-GPT 論文 $R^2$ |
|---|---|---|
| 名稱 | 患者級預測 $R^2$ | 變數間相關性保持 $R^2$ |
| 輸入 | 患者平均後的真實值與預測值 | 真實與預測的兩兩相關係數 |
| 主要問題 | 模型是否預測出了每個患者的平均值？ | 模型是否保持了變數之間的關係？ |
| 是否針對每個變數單獨計算？ | 是 | 否，它將多個變數結合在一起進行計算 |
| 輸入範例 | $\overline{y}_p, \overline{\hat{y}}_p$ | $c^{true}, c^{pred}$ |

簡短總結：

```text
當前繪圖 R² = 患者平均後，y_true vs y_pred 的預測 R²。
論文 R² = 真實相關性矩陣 vs 預測相關性矩陣的 R²。
```

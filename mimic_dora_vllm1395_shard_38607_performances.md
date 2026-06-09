# Performance Summary for mimic_dora_vllm1395_shard_38607
This document summarizes the **Resulting performances** across all 8 shards (0 to 7) extracted from the logs and the raw JSON metadata files.
## Overall Numeric Columns Performance Comparison
| Shard | R² | MAE | RMSE | Spearman Corr | NRMSE | Dir Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Shard 0 | 0.263141 | 0.562863 | 0.885479 | 0.560032 | 0.857982 | 0.418011 |
| Shard 1 | 0.462784 | 0.556927 | 0.776988 | 0.575000 | 0.713753 | 0.394509 |
| Shard 2 | 0.196303 | 2.264834 | 137.217290 | 0.535748 | 0.892173 | 0.404378 |
| Shard 3 | 0.377521 | 0.554687 | 0.768199 | 0.589898 | 0.784678 | 0.401716 |
| Shard 4 | 0.329518 | 0.551735 | 0.748339 | 0.552956 | 0.817780 | 0.399670 |
| Shard 5 | 0.326543 | 0.579557 | 0.841199 | 0.568438 | 0.820030 | 0.418046 |
| Shard 6 | 0.208718 | 0.594294 | 1.199576 | 0.570750 | 0.887261 | 0.409572 |
| Shard 7 | 0.328786 | 0.552208 | 0.740251 | 0.532423 | 0.815740 | 0.402207 |

## Per-Variable MAE Comparison (220210, 220277, 220635)
| Shard | 220210 MAE | 220277 MAE | 220635 MAE |
| :--- | :---: | :---: | :---: |
| Shard 0 | 0.653039 | 0.562573 | 0.472978 |
| Shard 1 | 0.669470 | 0.590201 | 0.411111 |
| Shard 2 | 5.749136 | 0.570683 | 0.474682 |
| Shard 3 | 0.649805 | 0.550758 | 0.463496 |
| Shard 4 | 0.631386 | 0.558691 | 0.465127 |
| Shard 5 | 0.654190 | 0.577116 | 0.507366 |
| Shard 6 | 0.621684 | 0.577990 | 0.583210 |
| Shard 7 | 0.632320 | 0.606871 | 0.417432 |

## Raw Log Extractions
Below are the truncated prints directly extracted from each shard's `.out` log file:

<details>
<summary><b>mimic_dora_vllm1395_shard_38607_0.out</b></summary>

```
Resulting performances: 
                                                       Value
TEST shard 000 of 008 all categorical columns f...       NaN
TEST shard 000 of 008 all categorical columns a...       NaN
TEST shard 000 of 008 all numeric columns r2 ov...  0.263141
TEST shard 000 of 008 all numeric columns mae o...  0.562863
TEST shard 000 of 008 all numeric columns rmse ...  0.885479
TEST shard 000 of 008 all numeric columns spear...  0.560032
TEST shard 000 of 008 all numeric columns nrmse...  0.857982
TEST shard 000 of 008 all numeric columns dir a...  0.418011
```

</details>

<details>
<summary><b>mimic_dora_vllm1395_shard_38607_1.out</b></summary>

```
Resulting performances: 
                                                       Value
TEST shard 001 of 008 all categorical columns f...       NaN
TEST shard 001 of 008 all categorical columns a...       NaN
TEST shard 001 of 008 all numeric columns r2 ov...  0.462784
TEST shard 001 of 008 all numeric columns mae o...  0.556927
TEST shard 001 of 008 all numeric columns rmse ...  0.776988
TEST shard 001 of 008 all numeric columns spear...  0.575000
TEST shard 001 of 008 all numeric columns nrmse...  0.713753
TEST shard 001 of 008 all numeric columns dir a...  0.394509
```

</details>

<details>
<summary><b>mimic_dora_vllm1395_shard_38607_2.out</b></summary>

```
Resulting performances: 
                                                         Value
TEST shard 002 of 008 all categorical columns f...         NaN
TEST shard 002 of 008 all categorical columns a...         NaN
TEST shard 002 of 008 all numeric columns r2 ov...    0.196303
TEST shard 002 of 008 all numeric columns mae o...    2.264834
TEST shard 002 of 008 all numeric columns rmse ...  137.217290
TEST shard 002 of 008 all numeric columns spear...    0.535748
TEST shard 002 of 008 all numeric columns nrmse...    0.892173
TEST shard 002 of 008 all numeric columns dir a...    0.404378
```

</details>

<details>
<summary><b>mimic_dora_vllm1395_shard_38607_3.out</b></summary>

```
Resulting performances: 
                                                       Value
TEST shard 003 of 008 all categorical columns f...       NaN
TEST shard 003 of 008 all categorical columns a...       NaN
TEST shard 003 of 008 all numeric columns r2 ov...  0.377521
TEST shard 003 of 008 all numeric columns mae o...  0.554687
TEST shard 003 of 008 all numeric columns rmse ...  0.768199
TEST shard 003 of 008 all numeric columns spear...  0.589898
TEST shard 003 of 008 all numeric columns nrmse...  0.784678
TEST shard 003 of 008 all numeric columns dir a...  0.401716
```

</details>

<details>
<summary><b>mimic_dora_vllm1395_shard_38607_4.out</b></summary>

```
Resulting performances: 
                                                       Value
TEST shard 004 of 008 all categorical columns f...       NaN
TEST shard 004 of 008 all categorical columns a...       NaN
TEST shard 004 of 008 all numeric columns r2 ov...  0.329518
TEST shard 004 of 008 all numeric columns mae o...  0.551735
TEST shard 004 of 008 all numeric columns rmse ...  0.748339
TEST shard 004 of 008 all numeric columns spear...  0.552956
TEST shard 004 of 008 all numeric columns nrmse...  0.817780
TEST shard 004 of 008 all numeric columns dir a...  0.399670
```

</details>

<details>
<summary><b>mimic_dora_vllm1395_shard_38607_5.out</b></summary>

```
Resulting performances: 
                                                       Value
TEST shard 005 of 008 all categorical columns f...       NaN
TEST shard 005 of 008 all categorical columns a...       NaN
TEST shard 005 of 008 all numeric columns r2 ov...  0.326543
TEST shard 005 of 008 all numeric columns mae o...  0.579557
TEST shard 005 of 008 all numeric columns rmse ...  0.841199
TEST shard 005 of 008 all numeric columns spear...  0.568438
TEST shard 005 of 008 all numeric columns nrmse...  0.820030
TEST shard 005 of 008 all numeric columns dir a...  0.418046
```

</details>

<details>
<summary><b>mimic_dora_vllm1395_shard_38607_6.out</b></summary>

```
Resulting performances: 
                                                       Value
TEST shard 006 of 008 all categorical columns f...       NaN
TEST shard 006 of 008 all categorical columns a...       NaN
TEST shard 006 of 008 all numeric columns r2 ov...  0.208718
TEST shard 006 of 008 all numeric columns mae o...  0.594294
TEST shard 006 of 008 all numeric columns rmse ...  1.199576
TEST shard 006 of 008 all numeric columns spear...  0.570750
TEST shard 006 of 008 all numeric columns nrmse...  0.887261
TEST shard 006 of 008 all numeric columns dir a...  0.409572
```

</details>

<details>
<summary><b>mimic_dora_vllm1395_shard_38607_7.out</b></summary>

```
Resulting performances: 
                                                       Value
TEST shard 007 of 008 all categorical columns f...       NaN
TEST shard 007 of 008 all categorical columns a...       NaN
TEST shard 007 of 008 all numeric columns r2 ov...  0.328786
TEST shard 007 of 008 all numeric columns mae o...  0.552208
TEST shard 007 of 008 all numeric columns rmse ...  0.740251
TEST shard 007 of 008 all numeric columns spear...  0.532423
TEST shard 007 of 008 all numeric columns nrmse...  0.815740
TEST shard 007 of 008 all numeric columns dir a...  0.402207
```

</details>


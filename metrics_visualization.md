# DT-GPT Model Evaluation: Shard Metrics Visualization

This artifact presents a side-by-side bar chart comparison of the evaluation metrics across the shards:
1. **R² (Coefficient of Determination)**
2. **Spearman Correlation**

The red dashed line represents the performance of **Checkpoint-700 (Shard 0)** (R² = 0.2452, Spearman = 0.5353) as a baseline comparison.

![DT-GPT Evaluation Metrics: R² and Spearman Correlation](/home/r15543056/.gemini/antigravity-cli/brain/472d6fd6-d98c-4947-aa51-d59de23e30b5/shard_metrics_visualization.png)

## Data Summary Table

| Evaluation Run / Shard | R² Overall | Spearman Correlation | MAE Overall |
| :--- | :---: | :---: | :---: |
| **Checkpoint-700 (Shard 0)** | 0.245194 | 0.535314 | 0.571641 |
| **Checkpoint-1395 (Shard 0)** | 0.263141 | 0.560032 | 0.562863 |
| **Checkpoint-1395 (Shard 1)** | 0.462784 | 0.575000 | 0.556927 |
| **Checkpoint-1395 (Shard 2)** | 0.196303 | 0.535748 | 2.264834 |
| **Checkpoint-1395 (Shard 3)** | 0.377521 | 0.589898 | 0.554687 |
| **Checkpoint-1395 (Shard 4)** | 0.329518 | 0.552956 | 0.551735 |
| **Checkpoint-1395 (Shard 5)** | 0.326543 | 0.568438 | 0.579557 |
| **Checkpoint-1395 (Shard 6)** | 0.208718 | 0.570750 | 0.594294 |
| **Checkpoint-1395 (Shard 7)** | 0.328786 | 0.532423 | 0.552208 |

---
*Generated using matplotlib in `dtgpt` conda environment.*

# True vs. Predicted Evaluation Plot Gallery

This gallery showcases the True vs. Predicted scatter plots and sequence autocorrelation analysis for the clinical variables.

````carousel
![Patient-Averaged True vs Predicted (No Outliers)](/home/r15543056/.gemini/antigravity-cli/brain/472d6fd6-d98c-4947-aa51-d59de23e30b5/patient_avg_true_vs_pred_merged_1epoch.png)
<!-- slide -->
![Patient-Averaged True vs Predicted (With Outliers)](/home/r15543056/.gemini/antigravity-cli/brain/472d6fd6-d98c-4947-aa51-d59de23e30b5/patient_avg_true_vs_pred_merged_with_outlier_1epoch.png)
<!-- slide -->
![Shard 003 Step-Level True vs Predicted Scatter Plot](/home/r15543056/.gemini/antigravity-cli/brain/472d6fd6-d98c-4947-aa51-d59de23e30b5/scatter_comparison.png)
<!-- slide -->
![Patient Sequence Autocorrelation (ACF)](/home/r15543056/.gemini/antigravity-cli/brain/472d6fd6-d98c-4947-aa51-d59de23e30b5/acf_comparison.png)
````

## Plot Types and Explanations

1. **Patient-Averaged True vs. Predicted (No Outliers)**:
   - Evaluates the baseline prediction quality at the patient level by averaging clinical trajectories.
   - Shows strong baseline alignment with $R^2$ values (e.g., $R^2 = 0.5717$ for Respiratory Rate).
2. **Patient-Averaged True vs. Predicted (With Outliers)**:
   - Includes anomaly database recordings (e.g., Patient 13793 in Respiratory Rate) which dominates the plot variance and reduces $R^2$ to $0.0004$.
3. **Shard 003 Step-Level Scatter Plot**:
   - Re-generated step-level predictions of **Magnesium**, **SpO₂**, and **Respiratory Rate** without averaging.
   - Includes reference lines $y = x$.
4. **Patient Sequence Autocorrelation (ACF)**:
   - Displays correlation coefficient across observation step lags (1 to 12) for clinical patient sequences.

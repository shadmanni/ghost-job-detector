# Empirical Results & Dataset Summary Statistics

This report synthesizes dataset metrics, model performance, and validation correlations for direct inclusion in the **IEEE Paper Results & Discussion Section**.

## 1. Dataset & Coverage Overview

| Metric | Empirical Value | Description |
| :--- | :---: | :--- |
| **Total Job Postings ($N$)** | `160` | Scraped job descriptions across corporate career portals and indices |
| **Total Community Reviews ($M$)** | `16` | Reddit / Glassdoor community feedback records |
| **Target Companies Covered** | `5` | Distinct corporate entities evaluated in dataset |

## 2. Ghost Job Score Statistical Distribution

Statistical properties of composite Ghost Job Scores ($0 - 100$ scale) across evaluated postings:

| Statistic | Value |
| :--- | :---: |
| **Mean Score ($\mu$)** | `5.86` |
| **Standard Deviation ($\sigma$)** | `5.62` |
| **Minimum Score** | `1.05` |
| **25th Percentile ($Q_1$)** | `2.96` |
| **Median Score ($Q_2$)** | `3.89` |
| **75th Percentile ($Q_3$)** | `5.79` |
| **Maximum Score** | `24.56` |
| **High-Risk Postings ($\ge 60.0$)** | `0` |

## 3. Classifier Performance Metrics

Performance evaluation of the fine-tuned BERT ghost-signal classifier against hand-verified benchmark annotations:

| Metric | Score |
| :--- | :---: |
| **Precision** | `0.9167` |
| **Recall** | `0.8462` |
| **F1 Score** | `0.8800` |
| **Accuracy** | `0.8846` |

## 4. Empirical Model Validation Correlations

| Validation Signal | Pearson $r$ | $p$-value | Interpretation |
| :--- | :---: | :---: | :--- |
| **Candidate Sentiment vs. Ghost Score** | `0.9942` | `0.0005` | Strong positive correlation validating internal model against community frustration |
| **GA4 Web Traffic vs. Ghost Score** | `-0.7870` | `0.1141` | Inverse engagement trend pilot validation |

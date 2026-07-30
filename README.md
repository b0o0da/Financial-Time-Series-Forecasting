# 📱 Social Media Engagement Analysis

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0.svg)](https://seaborn.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

An end-to-end exploratory data analysis (EDA) project on a social media engagement dataset — covering data cleaning, outlier treatment, visualization, statistical hypothesis testing, encoding, and dimensionality reduction (PCA).

---

## 📚 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Pipeline Steps](#pipeline-steps)
- [Key Analyses](#key-analyses)
- [How to Run](#how-to-run)
- [Notes](#notes)
- [License](#license)

---

## 🔍 Overview

This notebook explores a social media engagement dataset to understand what drives post performance (impressions, likes, comments, shares) across platforms, locations, and content types. It combines classic EDA with statistical testing and prepares the data for downstream modeling via encoding and PCA.

---

## 📊 Dataset

- **Source:** [Social Media Engagement Dataset](https://www.kaggle.com/datasets) (Kaggle)
- **Content:** Posts with metadata (platform, location, language, topic category, sentiment, emotion type, brand/product, mentions, hashtags, text content) and engagement metrics (impressions, likes, comments, shares, toxicity score, engagement rate)

---

## 🛠️ Tech Stack

- **Core Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Missing Data:** `missingno`
- **Preprocessing:** `scikit-learn` (`StandardScaler`, `MinMaxScaler`, `LabelEncoder`, `OrdinalEncoder`, `PCA`)
- **Statistics:** `scipy.stats` (Kruskal-Wallis test)

---

## 🧩 Pipeline Steps

1. **Import libraries & load data** — read the raw CSV, keep a copy for later comparison
2. **Check duplicates & nulls** — inspect missing value percentages per column
3. **Drop & fix columns** — clean the `mentions` column, drop identifier columns (`post_id`, `user_id`, `campaign_name`)
4. **Outlier detection & treatment** — box plots per numeric column, IQR-based capping (Winsorization)
5. **Null value treatment** — visualize missingness with `missingno`, drop remaining nulls
6. **Visualization**
   - Histograms (distribution of numeric features)
   - Scatter plots (features vs. impressions)
   - Line & count plots (toxicity score and post count by location)
   - Regression plots (text length vs. toxicity/likes/comments/shares)
   - Correlation heatmap of engagement metrics
7. **Scaling** — Min-Max scaling of numeric features
8. **Hypothesis testing** — Kruskal-Wallis test on engagement rate across platforms and emotion types
9. **Encoding**
   - One-Hot Encoding (platform, location, language, topic category, sentiment, emotion type, brand, product, mentions, day of week)
   - Ordinal Encoding (`campaign_phase`: Pre-Launch → Launch → Post-Launch)
   - Datetime feature extraction (year, month, day, hour, day of week, is_weekend)
   - Text features (`text_len`, `word_count`)
10. **Dimensionality reduction (PCA)** — standardize numeric features and reduce to 3 principal components

---

## 🔬 Key Analyses

- **Outlier treatment:** IQR method with lower/upper fence capping, applied per numeric column
- **Kruskal-Wallis test:** non-parametric test used to check whether engagement rate distributions differ significantly across `platform` groups and `emotion_type` groups
- **Correlation heatmap:** relationships between impressions, likes, comments, shares, and toxicity score
- **PCA:** reduces the numeric feature space to 3 components for potential downstream modeling or visualization

---

## 🚀 How to Run

1. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy missingno
   ```

2. **Set the dataset path** — update the `pd.read_csv(...)` path to point to your local copy of the dataset

3. **Run the notebook cells in order**:
   Data loading → Cleaning → Outlier treatment → Null handling → Visualization → Scaling → Hypothesis testing → Encoding → PCA

---

## 📝 Notes

- Originally built to run on Kaggle (`/kaggle/input/...` paths) — update paths if running locally.
- Outlier capping and scaling are applied to a working copy (`df`), while a separate `df_copy` is preserved for comparison and used again for the PCA step.
- This is a pure EDA/preprocessing pipeline — no predictive model is trained here.

---

## 📄 License

This project is open-source. Add your preferred license (e.g., MIT) here.

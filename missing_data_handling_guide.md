# Missing Data Handling: Comprehensive Strategy Guide

Handling missing data effectively requires matching the missingness level (percentage missing) with the underlying data mechanism, model type, and domain context.

---

## Decision Flowchart Overview

Before picking a strategy, answer these three questions:
1. **What type of feature is it?** (Numerical vs. Categorical)
2. **What model are you planning to use?** (Tree-based like XGBoost/Random Forest vs. Distance/Linear models like SVM/Logistic Regression/Neural Networks)
3. **Is the missingness informative (MNAR) or random (MCAR/MAR)?**
   * *MCAR (Missing Completely at Random):* Missingness has no relationship with any feature or target.
   * *MAR (Missing at Random):* Missingness relates to other observed features (e.g., older patients skipping online surveys).
   * *MNAR (Missing Not at Random):* The missing value depends on the missing value itself (e.g., high-income individuals refusing to disclose income).

---

## 1. Low Missingness (< 5%)

When missingness is under 5%, the primary goal is efficiency. Complex imputation methods rarely offer performance gains over simple techniques at this scale.

### Strategies

#### A. Listwise Deletion (Row Removal)
* **How:** Remove rows containing missing values (`df.dropna(subset=['column_name'])`).
* **When to choose this over others:**
  * You have a **large dataset** (e.g., > 100,000 rows) where losing 1-3% of rows does not impact statistical power.
  * You suspect data collection corruption or garbage entries in specific rows.
* **When NOT to choose this:**
  * Your dataset is small (e.g., < 1,000 samples).
  * Missingness is not random (MNAR)—dropping rows will introduce bias into your target distribution.

#### B. Simple Mean / Median Imputation (Numerical Features)
* **How:** Fill missing continuous values using mean or median (`df['col'].fillna(df['col'].median())`).
* **When to choose Median over Mean:** Use **Median** if the feature distribution is skewed or contains extreme outliers. Use **Mean** only if the distribution is approximately bell-shaped/normal.
* **When to choose this over Listwise Deletion:**
  * Your dataset is small, and you cannot afford to drop any rows.

#### C. Mode / Frequency Imputation (Categorical Features)
* **How:** Fill missing categories with the most frequent value (`df['col'].fillna(df['col'].mode()[0])`).
* **When to choose this over others:**
  * One single category dominates the distribution (e.g., 90% of samples belong to Category A).

---

## 2. Moderate Missingness (5% to 20%)

At this level, simple mean/median imputation starts to distort feature distributions and understate true variance, while dropping rows leads to noticeable data loss.

### Strategies

#### A. Model-Based Imputation (KNN or MICE / IterativeImputer)
* **How:** Predict missing values using surrounding features via $KNNImputer$ or Multivariate Imputation by Chained Equations ($IterativeImputer$).
* **When to choose this over others:**
  * Strong correlations exist between the column with missing data and other features in your dataset.
  * You are using linear models, SVMs, or Neural Networks that require complete numerical matrices and sensitive scaling.
* **When NOT to choose this:**
  * High-dimensional datasets where KNN or MICE become computationally expensive or cause training bottlenecks.

#### B. Grouped Statistical Imputation
* **How:** Impute missing values based on subgroup statistics (e.g., fill missing blood pressure using the median blood pressure grouped by `age_group` and `sex`).
* **When to choose this over Model-Based (KNN/MICE):**
  * You have clear domain-driven feature groupings (e.g., demographic segments, geographic regions).
  * You need an interpretable, fast, and reproducible imputation pipeline without model overhead.

#### C. Simple Imputation + Binary Missingness Flag (`is_missing`)
* **How:** Create an explicit indicator column (`df['col_is_missing'] = df['col'].isnull().astype(int)`) before performing mean/median imputation.
* **When to choose this over pure imputation:**
  * You suspect the fact that data is missing carries predictive value (MNAR).

---

## 3. High Missingness (20% to 50%)

At 20–50% missingness, substituting missing entries with statistical estimates creates major artificial artifacts. Preserving missingness as a distinct state is usually superior to estimating values.

### Strategies

#### A. Explicit "Missing" Category (Categorical Features)
* **How:** Replace missing string/category values with a dedicated label like `'Missing'` or `'Not_Tested'` (`df['col'].fillna('Missing')`).
* **When to choose this over Imputation:**
  * For categorical variables, this is almost always the preferred choice because it introduces zero assumptions while retaining 100% of the non-null data.

#### B. Dedicated Constant Value + Flag (Numerical Features)
* **How:** Fill missing numbers with an out-of-range sentinel value (e.g., `-1` or `-999`) or pair standard imputation with a mandatory `is_missing` binary flag.
* **When to choose Sentinel Value (-1/-999) over Imputation:**
  * You are using **tree-based algorithms** (XGBoost, LightGBM, Random Forest, CatBoost). Tree algorithms easily isolate sentinel values into separate branches.
* **When to choose Advanced Imputation + Flag over Sentinel Value:**
  * You are training **linear or neural network models**, where extreme sentinel values like `-999` distort weights and gradients.

---

## 4. Very High Missingness (> 50%)

When more than half of a feature's values are absent, the column is dominated by missingness.

### Strategies

#### A. Domain-Specific Flag / Categorical Encoding (`'Not_Tested'`)
* **How:** Convert the column into a categorical indicator or binary feature indicating whether a test/measurement occurred.
* **When to choose this over Column Dropping:**
  * In medical, financial, or behavioral datasets where missingness represents an action or clinical decision (e.g., a specific diagnostic test was skipped because the patient showed no symptoms).

#### B. Column Deletion (Dropping Feature)
* **How:** Remove the column completely (`df.drop(columns=['col_name'])`).
* **When to choose this over retaining the feature:**
  * The feature has low correlation/importance with the target variable.
  * The missingness is pure noise (MCAR due to sensor failure or incomplete record-keeping).
  * Retaining the column adds computational complexity with negligible model benefit.

---

## Strategy Selection Quick Matrix

| Missing % | Feature Type | Pipeline / Model Type | **Recommended Choice** | **Why Pick This Over Alternatives?** |
| :--- | :--- | :--- | :--- | :--- |
| **< 5%** | Any | Large Dataset | **Listwise Deletion** | Simplest path; negligible loss of power. |
| **< 5%** | Skewed Numerical | Small Dataset | **Median Imputation** | Preserves sample size without outlier bias. |
| **5% – 20%** | Numerical | Linear / NN / SVM | **KNN / MICE Imputation** | Preserves feature relationships for sensitive algorithms. |
| **5% – 20%** | Numerical | Domain-grouped data | **Grouped Median** | Faster and more interpretable than KNN/MICE. |
| **20% – 50%** | Categorical | Any Model | **`'Missing'` Category** | Makes zero artificial assumptions. |
| **20% – 50%** | Numerical | Tree-based models | **Sentinel (-1) or XGBoost Native** | Tree models naturally learn splits on missing flags. |
| **> 50%** | Any | Uninformative / Noise | **Drop Column** | Reduces dimensionality and eliminates artificial noise. |
| **> 50%** | Any | Informative / Medical | **`'Not_Tested'` Category** | Captures the diagnostic signal of why data was skipped. |
```

The comprehensive reference guide for handling missing data across all percentages and scenarios is ready in the document editor. It breaks down the selection criteria for each strategy so you know exactly when to choose one method over another based on model type, feature characteristics, and data distribution.
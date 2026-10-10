# Heart Disease Analysis Project

## Power of Data Analysis and Interpretation

This project shows how data analysis is not just about training a model, but about making careful decisions at every step:

- Understanding data quality before modeling
- Handling missing values by percentage and context
- Validating patterns with statistical tests
- Comparing baseline models, then tuning the strongest one
- Interpreting results for practical use

The final outcome is a complete, explainable machine learning pipeline for binary heart disease prediction.

## Project Goal

Predict heart disease presence as a binary target:

- `target = 0`: no heart disease (`num = 0`)
- `target = 1`: heart disease present (`num > 0`)

## Repository Contents

- `heart_disease_dataset.csv`: main dataset used in the notebook
- `heart_disease_dataset_ext.txt`: extended sample view of the dataset
- `missing_data_handling_guide.md`: missing data strategy reference
- `work.ipynb`: full step by step analysis, EDA, modeling, and evaluation
- `README.md`: this study guide

## Dataset Snapshot

- Shape before preprocessing: **920 rows, 17 columns**
- Memory usage: **0.41 MB**
- Problem type: **binary classification**

Main original columns:

- `id`, `age`, `sex`, `dataset`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalch`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, `num`

Target engineering:

- Created `target` from `num`
- Dropped original `num`

## End to End Workflow

## 1) Data Inspection and Quality Checks

Started with:

- Dimension check (`df.shape`)
- Type and non null check (`df.info()`)
- Column wise missing summary

This step established the data structure and guided all later preprocessing choices.

## 2) Missing Data Handling Strategy (Context Driven)

Instead of one generic imputation rule, missing values were handled by range and feature meaning.

### Low missingness (< 5%)

- `chol`:
  - converted to numeric
  - invalid values (`<= 0`) treated as missing
  - capped extreme outliers at `300`
  - median imputation used (robust to outliers)
- `restecg` (~0.217% missing): dropped rows with null values

### Moderate missingness (5 to 20%)

Applied MICE style iterative imputation to:

- `trestbps`, `thalch`, `fbs`, `exang`, `oldpeak`

Post processing:

- rounded `fbs` and `exang` back to binary style numeric values

### High missingness (20 to 50%)

- `slope`: filled missing with explicit category `Missing`

### Very high missingness (> 50%)

- `thal`: filled with `Not Tested`
- `ca`: filled with sentinel `-1`

This is a strong example of interpretation driven preprocessing, not just automatic filling.

## 3) Exploratory Data Analysis and Interpretation

### Univariate analysis

- Visualized numerical distributions with histograms and KDE
- Measured skewness
- Applied log transform to reduce skew in `oldpeak`:
  - clipped lower bound at 0
  - created `oldpeak_log = log1p(oldpeak)`
  - dropped original `oldpeak`

### Categorical balance checks

Reviewed category percentages to detect dominant class issues.

Decision rules used:

- < 85% dominant category: keep
- 85 to 90%: monitor impact
- > 90%: candidate for drop

Result: no categorical column required dropping for dominance.

### Bivariate interpretation (feature vs target)

Used:

- boxplots by class
- overlaid KDE by class

Interpretation highlights:

- strongest visual separation: `thalch`, `oldpeak_log`
- moderate separation: `age`
- weaker separation: `trestbps`, `chol`

### Statistical validation

#### Mann Whitney U test (numerical vs target)

All 5 numerical features were significant (`p < 0.05`):

| Feature     | Group 0 Median | Group 1 Median | U Statistic |  p-value | Significant |
| ----------- | -------------: | -------------: | ----------: | -------: | ----------- |
| age         |          51.00 |          57.00 |     68804.0 | 0.000000 | Yes         |
| thalch      |         150.00 |         130.00 |    151351.0 | 0.000000 | Yes         |
| oldpeak_log |           0.00 |           0.70 |     56543.5 | 0.000000 | Yes         |
| trestbps    |         130.00 |         130.91 |     90375.0 | 0.000521 | Yes         |
| chol        |         239.50 |         239.50 |     91456.0 | 0.001344 | Yes         |

#### Chi square test (categorical vs target)

All 8 categorical features were significant (`p < 0.05`):

| Feature | Chi-square | dof |  p-value | Significant |
| ------- | ---------: | --: | -------: | ----------- |
| sex     |      84.81 |   1 | 0.000000 | Yes         |
| cp      |     268.12 |   3 | 0.000000 | Yes         |
| slope   |     143.49 |   3 | 0.000000 | Yes         |
| exang   |     170.29 |   1 | 0.000000 | Yes         |
| ca      |      87.69 |   4 | 0.000000 | Yes         |
| thal    |     109.09 |   3 | 0.000000 | Yes         |
| fbs     |      10.30 |   1 | 0.001327 | Yes         |
| restecg |      11.71 |   2 | 0.002863 | Yes         |

### Continuous vs continuous checks

- Built Pearson correlation matrix for numerical features
- No problematic high multicollinearity observed

## 4) Preprocessing and Feature Encoding

- Dropped: `id`, `dataset`
- Train test split: stratified 80/20
- Numerical preprocessing: `StandardScaler`
- Categorical preprocessing: `OneHotEncoder(drop='first', handle_unknown='ignore')`
- Combined pipeline with `ColumnTransformer`

Final processed shapes:

- Train: **(734, 23)**
- Test: **(184, 23)**

## 5) Baseline Modeling

Tested three models:

- Logistic Regression
- Random Forest
- XGBoost

### Baseline performance

| Model               | Accuracy | ROC AUC |
| ------------------- | -------: | ------: |
| Logistic Regression |   0.8478 |  0.9248 |
| Random Forest       |   0.8533 |  0.9087 |
| XGBoost             |   0.8424 |  0.8796 |

Interpretation:

- Logistic Regression gave best ROC AUC
- Random Forest gave slightly higher accuracy
- XGBoost underperformed relative to the other two in this run

## 6) Hyperparameter Tuning (Random Forest)

Used `GridSearchCV` with 5 fold cross validation and ROC AUC scoring.

Best parameters:

- `max_depth = 6`
- `min_samples_leaf = 2`
- `min_samples_split = 2`
- `n_estimators = 200`

Tuned performance:

- Accuracy: **0.8641**
- ROC AUC: **0.9193**

Classification report (tuned model):

- Class `0` precision: 0.90, recall: 0.78
- Class `1` precision: 0.84, recall: 0.93
- Macro F1: 0.86

This indicates strong detection of disease positive cases (high recall for class 1).

## Why This Project Is a Good Study Source

This notebook is valuable because it combines:

- Data cleaning with domain logic
- Statistical testing for interpretation confidence
- Visual and quantitative EDA
- Clear preprocessing pipeline
- Baseline benchmarking and tuning
- Practical clinical style reasoning (`Not Tested`, explicit missing categories)

It demonstrates that better interpretation leads to better modeling decisions.

## How to Run

## 1) Install dependencies

```bash
py -3.12 -m pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost
```

## 2) Open and run notebook

- Open `work.ipynb`
- Run cells from top to bottom

## Future Improvements

- Add permutation feature importance and SHAP interpretation
- Add calibration curves and threshold tuning for clinical sensitivity tradeoffs
- Compare tuned Logistic Regression and tuned XGBoost against tuned Random Forest
- Add repeated cross validation for more robust performance estimates
- Package preprocessing and model into a deployable inference pipeline

## Final Takeaway

The key lesson from this project:

**Strong machine learning performance comes from disciplined data analysis and interpretation first, then modeling.**

The preprocessing and interpretation steps here are as important as the final algorithm choice.

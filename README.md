# Gene Expression Classification — Golub Leukemia Dataset

> A reproducible machine-learning analysis of high-dimensional leukemia microarray data.

**Python · Gene Expression · Microarray · PCA · Feature Selection · Logistic Regression · Random Forest · XGBoost · Cross-Validation**

---

## Overview

This project analyzes the landmark **Golub leukemia gene-expression dataset** to test whether microarray expression profiles can distinguish:

- **ALL** — Acute Lymphoblastic Leukemia
- **AML** — Acute Myeloid Leukemia

The dataset contains only **72 patient samples** but **7,129 expression features**, making it a classic high-dimensional, low-sample-size (`p >> n`) machine-learning problem.

The project asks two main questions:

1. Can classical machine-learning models distinguish ALL from AML from gene-expression profiles?
2. Which expression probes contribute most strongly to the classification signal?

---

## Dataset

| Property | Value |
|---|---:|
| Total samples | **72** |
| ALL samples | **47** |
| AML samples | **25** |
| Expression features | **7,129** |
| Features selected within each CV fold | **100** |
| Task | Binary classification: ALL vs AML |

The normalized dataset is downloaded automatically by the analysis script.

Original study:

> Golub TR, Slonim DK, Tamayo P, et al. *Molecular Classification of Cancer: Class Discovery and Class Prediction by Gene Expression Monitoring.* Science. 1999.

---

## Analysis Workflow

```text
Golub gene-expression matrix
            |
            v
     Data / label checks
            |
            v
    Exploratory PCA
            |
            v
 Zero-variance filtering
            |
            v
 Univariate feature selection
   inside each CV training fold
            |
            v
      Model comparison
    /        |         \
Logistic   Random     XGBoost
Regression Forest
    \        |         /
     Stratified 5-fold CV
            |
            v
 Accuracy / Balanced Accuracy
       F1 / ROC-AUC
            |
            v
 Out-of-fold predictions
            |
            v
 Confusion matrices + ROC curves
            |
            v
 Exploratory full-data model fit
            |
            v
 Ranked predictive probes
```

---

## Why Leakage-Safe Feature Selection Matters

This dataset contains **7,129 expression measurements but only 72 patients**.

If feature selection were performed once on the entire dataset before cross-validation, information from validation patients could influence which probes are selected. That would create **data leakage** and could make performance appear better than it really is.

To reduce this risk, feature selection is placed **inside the scikit-learn pipeline**. Each training fold independently chooses its 100 features before being evaluated on the corresponding held-out fold.

---

# Results

## 1. Principal Component Analysis

The first two principal components explained:

| Component | Variance Explained |
|---|---:|
| PC1 | **22.6%** |
| PC2 | **11.7%** |
| PC1 + PC2 | **34.2%** |

![Golub leukemia PCA](results/figures/pca_all_vs_aml.png)

### Interpretation

The first two principal components do **not** produce a clean visual separation between ALL and AML.

This is an important result rather than a failure.

PCA is unsupervised: it finds directions that explain the greatest overall variation in the expression matrix without using the leukemia labels. The strongest sources of total variation therefore do not have to be the same directions that best discriminate ALL from AML.

The later supervised models can still perform strongly because they select expression features specifically useful for classification.

---

## 2. Cross-Validated Model Comparison

Three classifiers were evaluated using the same **stratified 5-fold cross-validation** protocol.

| Model | Accuracy | Balanced Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| **Logistic Regression** | **0.971 ± 0.064** | **0.969 ± 0.070** | **0.960 ± 0.089** | 0.991 ± 0.020 |
| Random Forest | 0.958 ± 0.038 | 0.950 ± 0.050 | 0.937 ± 0.058 | **1.000 ± 0.000** |
| XGBoost | 0.959 ± 0.037 | 0.950 ± 0.050 | 0.937 ± 0.058 | **1.000 ± 0.000** |

### Main result

**Logistic Regression achieved the highest mean classification accuracy and F1 score**, despite being much simpler than the tree-based ensemble models.

This is a useful result in a `p >> n` biomedical setting: a more complex model did not automatically produce better classification performance.

---

## 3. Cross-Validated Confusion Matrices

### Logistic Regression

![Logistic Regression confusion matrix](results/figures/confusion_logistic_regression.png)

Across the pooled out-of-fold predictions:

| True class | Correct | Misclassified |
|---|---:|---:|
| ALL | **46 / 47** | 1 predicted AML |
| AML | **24 / 25** | 1 predicted ALL |

Overall, Logistic Regression correctly classified **70 of 72 patients** in the pooled out-of-fold predictions.

---

### Random Forest

![Random Forest confusion matrix](results/figures/confusion_random_forest.png)

| True class | Correct | Misclassified |
|---|---:|---:|
| ALL | **46 / 47** | 1 predicted AML |
| AML | **23 / 25** | 2 predicted ALL |

Random Forest correctly classified **69 of 72 patients**.

---

### XGBoost

![XGBoost confusion matrix](results/figures/confusion_xgboost.png)

| True class | Correct | Misclassified |
|---|---:|---:|
| ALL | **46 / 47** | 1 predicted AML |
| AML | **23 / 25** | 2 predicted ALL |

XGBoost also correctly classified **69 of 72 patients**.

---

## 4. ROC Analysis

![Cross-validated ROC curves](results/figures/roc_curves.png)

The pooled out-of-fold ROC curves show very strong discrimination:

| Model | Pooled Out-of-Fold ROC-AUC |
|---|---:|
| Logistic Regression | **0.993** |
| Random Forest | **0.998** |
| XGBoost | **0.994** |

The ROC-AUC values in this figure differ slightly from the mean ROC-AUC values in the cross-validation table.

That is expected because the table reports the **mean AUC calculated separately within each fold**, whereas this figure calculates one AUC after pooling all out-of-fold predictions.

---

## 5. Exploratory Predictive Probe Ranking

After cross-validation, a Logistic Regression pipeline was fit to the complete dataset for **exploratory feature interpretation only**.

This final fit is **not an independent validation experiment** and the resulting probe ranking should not be interpreted as a validated biomarker signature.

![Top exploratory predictive probes](results/figures/top_predictive_probes.png)

### Top-ranked probes

| Rank | Probe | Logistic coefficient for AML | Absolute coefficient |
|---:|---|---:|---:|
| 1 | **Y07604_at** | +0.475 | **0.475** |
| 2 | **U82759_at** | +0.307 | **0.307** |
| 3 | **U70063_at** | +0.302 | **0.302** |
| 4 | **J02783_at** | +0.291 | **0.291** |
| 5 | **HG1612-HT1612_at** | -0.268 | **0.268** |
| 6 | **M26708_s_at** | -0.257 | **0.257** |
| 7 | **M31303_rna1_at** | -0.252 | **0.252** |
| 8 | **L07633_at** | -0.249 | **0.249** |
| 9 | **M98399_s_at** | +0.241 | **0.241** |
| 10 | **X85116_rna1_s_at** | +0.240 | **0.240** |

A positive coefficient means higher expression pushes the fitted model toward **AML**, while a negative coefficient pushes it toward **ALL**, after accounting for the other selected features in the model.

---

## Biological Context Check

Several highly ranked probe accessions correspond to biologically recognizable genes:

| Probe / accession | Gene / historical description | Direction in exploratory model |
|---|---|---|
| **Y07604_at** | **NME4 / nm23-H4**, mitochondrial nucleoside-diphosphate kinase | AML |
| **U82759_at** | **HOXA9** | AML |
| **J02783_at** | **P4HB**, protein disulfide-isomerase / prolyl 4-hydroxylase beta subunit | AML |
| **M31303_rna1_at** | **STMN1 / Oncoprotein 18** | ALL in this fitted multivariable model |
| **M92287_at** | **CCND3 / Cyclin D3** | ALL in this fitted multivariable model |

Previous leukemia gene-selection studies have independently reported several of these accession IDs, including **Y07604, M31303, and M92287**, among discriminatory or repeatedly selected leukemia-expression features.

This provides a useful **literature consistency check**, but it does not prove that the current feature ranking is a clinically validated biomarker signature.

---

# Key Findings

1. The first two PCA dimensions explain about **34.2%** of total expression variation but do not cleanly separate ALL from AML.
2. Leakage-safe supervised models nevertheless classify the two leukemia groups with high cross-validated performance.
3. **Logistic Regression** achieved the highest mean accuracy (**97.1%**) and F1 (**0.960**).
4. The pooled out-of-fold Logistic Regression predictions correctly classified **70 of 72 patients**.
5. Random Forest and XGBoost achieved extremely high ranking performance, with fold-wise mean ROC-AUC values of **1.000**.
6. Several highly ranked probes correspond to genes with prior biological or leukemia-related literature support.
7. Because the cohort is small and high-dimensional, the predictive feature ranking should be treated as exploratory rather than as a biomarker discovery claim.

---

# Statistical and Machine-Learning Notes

## Stratified Cross-Validation

Stratified folds preserve approximately the same ALL/AML class proportions in each training and validation split.

This is especially useful for a small dataset where random splits could otherwise produce uneven class distributions.

## Feature Selection

The pipeline uses:

```text
VarianceThreshold
        ↓
SelectKBest(f_classif, k=100)
        ↓
Classifier
```

The feature-selection step is re-fit independently inside every training fold.

## F1 Score

F1 balances precision and recall for the AML class and is useful because the dataset contains more ALL than AML samples.

## ROC-AUC

ROC-AUC evaluates how well a model ranks AML samples above ALL samples across possible classification thresholds.

It is therefore possible for a model to have a near-perfect ROC-AUC while still making several mistakes at the default classification threshold.

---

# Limitations

This analysis has several important limitations:

- The dataset contains only **72 patients**.
- There are **7,129 expression features**, creating a strong risk of overfitting.
- The data come from an older microarray platform rather than modern RNA-seq.
- The cohort is a historical benchmark and is not representative of the diversity or scale expected in modern clinical validation.
- Cross-validation estimates generalization within this dataset but does not replace validation in an independent cohort.
- Hyperparameters were not subjected to a fully nested model-selection procedure.
- Probe rankings from the final full-data fit are exploratory.
- Probe-level coefficients should not be interpreted as causal biological effects.
- Some older Affymetrix probe identifiers require historical annotation resources to map reliably to modern gene symbols.
- High classification performance does not imply clinical readiness.

---

# Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the complete analysis:

```bash
python scripts/run_analysis.py
```

The script will:

- download the normalized Golub dataset,
- validate the expected cohort,
- collapse B-cell and T-cell ALL into the ALL class,
- perform exploratory PCA,
- select expression features inside each cross-validation fold,
- evaluate Logistic Regression, Random Forest, and XGBoost,
- create cross-validated predictions, confusion matrices, and ROC curves,
- fit an exploratory final model,
- and save a ranked list of predictive probes.

---

# Repository Structure

```text
gene-expression-classification/
├── data/
│   ├── raw/
│   │   └── README.md
│   └── processed/
│       └── README.md
├── notebooks/
│   └── gene_expression_classification.ipynb
├── scripts/
│   └── run_analysis.py
├── results/
│   ├── figures/
│   │   ├── pca_all_vs_aml.png
│   │   ├── roc_curves.png
│   │   ├── confusion_logistic_regression.png
│   │   ├── confusion_random_forest.png
│   │   ├── confusion_xgboost.png
│   │   └── top_predictive_probes.png
│   └── tables/
│       ├── dataset_summary.csv
│       ├── model_comparison.csv
│       ├── cross_validated_predictions.csv
│       ├── pca_coordinates.csv
│       └── top_predictive_probes.csv
├── .gitignore
├── requirements.txt
└── README.md
```

---

# Tools and Methods

| Area | Tools / Methods |
|---|---|
| Programming | Python |
| Data manipulation | pandas, NumPy |
| Machine learning | scikit-learn, XGBoost |
| Dimensionality reduction | PCA |
| Feature selection | ANOVA F-statistic (`SelectKBest`) |
| Validation | Stratified 5-fold cross-validation |
| Metrics | Accuracy, balanced accuracy, F1, ROC-AUC |
| Visualization | Matplotlib |
| Data source | Golub leukemia microarray dataset |

---

# References

Golub TR, Slonim DK, Tamayo P, et al. **Molecular Classification of Cancer: Class Discovery and Class Prediction by Gene Expression Monitoring.** *Science*. 1999.

Additional literature used only as a biological consistency check for selected probe accessions:

- Y07604 / nm23-H4 (NME4): mitochondrial nucleoside-diphosphate kinase.
- U82759: HOXA9.
- J02783: P4HB.
- M31303: Oncoprotein 18 / STMN1.
- M92287: CCND3.

---

# Author

**Siham Boumalak**  
M.S. Artificial Intelligence  
Northeastern University

Research interests include **biomedical informatics, computational biology, transcriptomics, machine learning, trustworthy AI, and reproducible computational research**.

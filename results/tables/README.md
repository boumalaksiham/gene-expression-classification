# Analysis tables

Committed tables document a previous run. Run `python scripts/run_analysis.py` from the repository root to regenerate them; existing files are overwritten.

| Table | Contents |
|---|---|
| `dataset_summary.csv` | Cohort counts and expression dimensions |
| `model_comparison.csv` | Five-fold means and sample standard deviations of classification metrics |
| `cross_validated_predictions.csv` | Per-sample out-of-fold predictions and scores |
| `pca_coordinates.csv` | Coordinates from exploratory full-data PCA |
| `top_predictive_probes.csv` | Coefficients from an exploratory final full-data logistic-regression fit |

Metric standard deviations describe variation across folds, not confidence intervals. Save the source commit, package versions, cached input checksum, and all generated artifacts when reporting a rerun.

# Analysis figures

These figures are committed artifacts from a previous analysis run. Running `python scripts/run_analysis.py` from the repository root regenerates and overwrites them.

| Figure | Interpretation |
|---|---|
| `pca_all_vs_aml.png` | Exploratory PCA fitted on the full expression matrix |
| `confusion_*.png` | Confusion matrices from pooled out-of-fold predictions |
| `roc_curves.png` | ROC curves and AUC from pooled out-of-fold scores |
| `top_predictive_probes.png` | Exploratory coefficient ranking from a final full-data fit |

The ROC figure's pooled AUC differs from mean fold AUC in the comparison table. PCA and the final probe ranking are not independent validation. See the root README for methods and limitations.

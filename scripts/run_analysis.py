from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise SystemExit(
        "xgboost is not installed. Run: pip install -r requirements.txt"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"

DATA_URL = "https://raw.githubusercontent.com/pmking123/statistics-data/main/textbook_data/OpenIntro/golub.csv"
DATA_PATH = RAW_DIR / "golub.csv"

RANDOM_STATE = 42
N_SELECTED_FEATURES = 100


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        print(f"Using existing {path.name}")
        return

    print(f"Downloading {path.name}...")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)


def load_data(path: Path):
    df = pd.read_csv(path)

    # Some CSV mirrors include an exported row index.
    index_like = [
        c for c in df.columns
        if str(c).lower().startswith("unnamed:") or str(c) in {"X", "index"}
    ]
    if index_like:
        df = df.drop(columns=index_like)

    if "cancer" not in df.columns:
        raise ValueError(
            "Expected a 'cancer' column in the OpenIntro Golub dataset. "
            f"Columns begin with: {list(df.columns[:10])}"
        )

    cancer_raw = df["cancer"].astype(str).str.strip().str.lower()
    mapping = {"allb": "ALL", "allt": "ALL", "all": "ALL", "aml": "AML"}
    y_text = cancer_raw.map(mapping)

    if y_text.isna().any():
        unexpected = sorted(cancer_raw[y_text.isna()].unique())
        raise ValueError(f"Unexpected cancer labels: {unexpected}")

    # Known sample-level metadata in the OpenIntro representation.
    metadata_columns = {
        "Samples", "BM.PB", "Gender", "Source", "tissue.mf", "cancer"
    }

    expression_columns = [
        c for c in df.columns
        if c not in metadata_columns
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[expression_columns].copy()

    if X.isna().any().any():
        # Median imputation is only a safeguard; the distributed normalized
        # dataset is expected to be complete.
        X = X.fillna(X.median(numeric_only=True))

    y = (y_text == "AML").astype(int).to_numpy()
    labels = y_text.to_numpy()

    return df, X, y, labels


def make_models(k_features: int):
    common_prefix = [
        ("variance_filter", VarianceThreshold(threshold=0.0)),
        ("feature_selection", SelectKBest(score_func=f_classif, k=k_features)),
    ]

    models = {
        "Logistic Regression": Pipeline(
            common_prefix
            + [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            common_prefix
            + [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "XGBoost": Pipeline(
            common_prefix
            + [
                (
                    "model",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                )
            ]
        ),
    }
    return models


def plot_pca(X: pd.DataFrame, labels: np.ndarray) -> tuple[float, float]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    for group in ["ALL", "AML"]:
        mask = labels == group
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            label=group,
            alpha=0.8,
            s=45,
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.set_title("Golub leukemia gene-expression PCA")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_all_vs_aml.png", dpi=180)
    plt.close(fig)

    coords_df = pd.DataFrame(
        {"PC1": coords[:, 0], "PC2": coords[:, 1], "Class": labels}
    )
    coords_df.to_csv(TABLE_DIR / "pca_coordinates.csv", index=False)

    return tuple(pca.explained_variance_ratio_[:2])


def evaluate_models(X, y, models):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    rows = []
    prediction_rows = []
    roc_payload = {}

    for model_name, pipeline in models.items():
        print(f"Evaluating {model_name}...")

        cv_result = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )

        row = {"model": model_name}
        for metric in scoring:
            values = cv_result[f"test_{metric}"]
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        rows.append(row)

        # One out-of-fold prediction per sample using the same 5-fold scheme.
        pred = cross_val_predict(
            clone(pipeline), X, y, cv=cv, method="predict", n_jobs=1
        )
        proba = cross_val_predict(
            clone(pipeline), X, y, cv=cv, method="predict_proba", n_jobs=1
        )[:, 1]

        for i, (truth, guess, score) in enumerate(zip(y, pred, proba)):
            prediction_rows.append(
                {
                    "model": model_name,
                    "sample_index": i,
                    "true_class": "AML" if truth == 1 else "ALL",
                    "predicted_class": "AML" if guess == 1 else "ALL",
                    "aml_probability": float(score),
                }
            )

        fpr, tpr, _ = roc_curve(y, proba)
        auc = roc_auc_score(y, proba)
        roc_payload[model_name] = (fpr, tpr, auc)

        fig, ax = plt.subplots(figsize=(5.5, 5))
        ConfusionMatrixDisplay.from_predictions(
            y,
            pred,
            display_labels=["ALL", "AML"],
            ax=ax,
        )
        ax.set_title(f"{model_name}: cross-validated confusion matrix")
        fig.tight_layout()
        safe_name = model_name.lower().replace(" ", "_")
        fig.savefig(FIG_DIR / f"confusion_{safe_name}.png", dpi=180)
        plt.close(fig)

    comparison = pd.DataFrame(rows).sort_values(
        ["roc_auc_mean", "f1_mean"], ascending=False
    )
    comparison.to_csv(TABLE_DIR / "model_comparison.csv", index=False)

    pd.DataFrame(prediction_rows).to_csv(
        TABLE_DIR / "cross_validated_predictions.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, (fpr, tpr, auc) in roc_payload.items():
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Cross-validated ROC curves: AML vs ALL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curves.png", dpi=180)
    plt.close(fig)

    return comparison


def rank_predictive_probes(X: pd.DataFrame, y: np.ndarray, k_features: int):
    """
    Fit the logistic-regression pipeline on the complete dataset only for
    exploratory feature ranking. This is NOT an independent validation step.
    """
    pipeline = Pipeline(
        [
            ("variance_filter", VarianceThreshold(threshold=0.0)),
            ("feature_selection", SelectKBest(score_func=f_classif, k=k_features)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipeline.fit(X, y)

    variance_mask = pipeline.named_steps["variance_filter"].get_support()
    after_variance = np.asarray(X.columns)[variance_mask]

    select_mask = pipeline.named_steps["feature_selection"].get_support()
    selected_names = after_variance[select_mask]

    coefficients = pipeline.named_steps["model"].coef_.ravel()

    ranking = pd.DataFrame(
        {
            "probe": selected_names,
            "logistic_coefficient_for_AML": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    ).sort_values("absolute_coefficient", ascending=False)

    ranking.to_csv(TABLE_DIR / "top_predictive_probes.csv", index=False)

    top = ranking.head(20).sort_values("absolute_coefficient", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top["probe"].astype(str), top["absolute_coefficient"])
    ax.set_xlabel("Absolute logistic-regression coefficient")
    ax.set_ylabel("Expression probe")
    ax.set_title("Top exploratory predictive probes")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "top_predictive_probes.png", dpi=180)
    plt.close(fig)

    return ranking


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    download(DATA_URL, DATA_PATH)
    df, X, y, labels = load_data(DATA_PATH)

    counts = pd.Series(labels).value_counts()
    print(f"Raw shape: {df.shape}")
    print(f"Expression matrix: {X.shape}")
    print("Class counts:")
    print(counts)

    if len(df) != 72:
        print(
            f"WARNING: expected 72 samples from the normalized Golub dataset, "
            f"but found {len(df)}."
        )

    if X.shape[1] != 7129:
        print(
            f"WARNING: expected 7,129 expression features, "
            f"but found {X.shape[1]}."
        )

    if set(counts.index) != {"ALL", "AML"}:
        raise ValueError(f"Unexpected class labels after mapping: {counts.to_dict()}")

    k_features = min(N_SELECTED_FEATURES, X.shape[1])

    pc1, pc2 = plot_pca(X, labels)

    models = make_models(k_features)
    comparison = evaluate_models(X, y, models)
    ranking = rank_predictive_probes(X, y, k_features)

    summary = pd.DataFrame(
        {
            "value": {
                "samples": len(df),
                "all_samples": int((labels == "ALL").sum()),
                "aml_samples": int((labels == "AML").sum()),
                "expression_features": X.shape[1],
                "selected_features_per_cv_fold": k_features,
                "pca_variance_pc1": pc1,
                "pca_variance_pc2": pc2,
            }
        }
    )
    summary.to_csv(TABLE_DIR / "dataset_summary.csv")

    print("\nModel comparison:")
    print(
        comparison[
            [
                "model",
                "accuracy_mean",
                "balanced_accuracy_mean",
                "f1_mean",
                "roc_auc_mean",
            ]
        ].to_string(index=False)
    )

    print("\nTop exploratory probes:")
    print(ranking.head(10).to_string(index=False))

    print("\nAnalysis complete.")
    print(f"Figures: {FIG_DIR}")
    print(f"Tables:  {TABLE_DIR}")


if __name__ == "__main__":
    main()

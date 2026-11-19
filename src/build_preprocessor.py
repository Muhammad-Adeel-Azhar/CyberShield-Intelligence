"""Day 2, Part C (step 2): time-based split, preprocessing pipeline fitted on training data only, saved."""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "model_dataset.parquet"
MODELS = ROOT / "models"
META_DIR = ROOT / "data" / "metadata"

# Time-based split. 2025 and 2026 are held out: their KEV labels are still maturing.
TRAIN_FIRST, TRAIN_LAST = 2016, 2022
VAL_YEAR, TEST_YEAR = 2023, 2024

USE_PUBLISHED_YEAR = True   # the plan includes it; set False to run the ablation later
MIN_FREQUENCY = 50          # CWE values with fewer training rows are grouped together

NUMERIC = ["cvss_base_score", "desc_words", "desc_chars"] + (["published_year"] if USE_PUBLISHED_YEAR else [])
CATEGORICAL = ["attack_vector", "attack_complexity", "privileges_required", "user_interaction", "scope",
               "confidentiality_impact", "integrity_impact", "availability_impact", "cwe_category"]


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=MIN_FREQUENCY)),
    ])
    return ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])


def main() -> None:
    df = pd.read_parquet(DATA)
    for col in CATEGORICAL:  # make missing values plain NaN so the imputer always sees them
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)

    y = df["is_kev"]
    year = df["published_year"]
    masks = {
        f"train ({TRAIN_FIRST}-{TRAIN_LAST})": year.between(TRAIN_FIRST, TRAIN_LAST),
        f"validation ({VAL_YEAR})": year == VAL_YEAR,
        f"test ({TEST_YEAR})": year == TEST_YEAR,
        f"held out ({TEST_YEAR + 1}+)": year > TEST_YEAR,
    }
    names = list(masks)
    pd.set_option("display.width", 200, "display.max_rows", None)

    print("1. Time-based split")
    rows = []
    for name, m in masks.items():
        n, k = int(m.sum()), int(y[m].sum())
        rows.append({"set": name, "rows": n, "kev": k, "kev_pct": round(k / n * 100, 3)})
    print(pd.DataFrame(rows).to_string(index=False))

    train, val, test = (df[masks[n]] for n in names[:3])
    X_train, X_val, X_test = (d[NUMERIC + CATEGORICAL] for d in (train, val, test))

    # Fit on the training rows only, then apply to the others
    pre = build_preprocessor()
    Xt_train = pre.fit_transform(X_train)
    Xt_val, Xt_test = pre.transform(X_val), pre.transform(X_test)

    print("\n2. Pipeline fitted on the training set only")
    print(f"Numeric features    : {NUMERIC}")
    print(f"Categorical features: {len(CATEGORICAL)}")
    print(f"Output columns      : {Xt_train.shape[1]}  (train {Xt_train.shape}, val {Xt_val.shape}, test {Xt_test.shape})")
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_rows = []
    for col, cats, infreq in zip(CATEGORICAL, ohe.categories_, ohe.infrequent_categories_):
        n_infreq = 0 if infreq is None else len(infreq)
        cat_rows.append({"feature": col, "categories_seen": len(cats), "grouped_as_infrequent": n_infreq,
                         "output_columns": len(cats) - n_infreq + (1 if n_infreq else 0)})
    print(pd.DataFrame(cat_rows).to_string(index=False))

    print("\n3. Distribution checks")
    num_check = pd.DataFrame({
        "train_mean": X_train[NUMERIC].mean(), "val_mean": X_val[NUMERIC].mean(), "test_mean": X_test[NUMERIC].mean(),
        "train_std": X_train[NUMERIC].std(), "test_std": X_test[NUMERIC].std(),
    }).round(2)
    print("Numeric features in original units (a big train-vs-test gap means distribution shift):")
    print(num_check.to_string())
    n_num = len(NUMERIC)
    dense_train = Xt_train[:, :n_num].toarray() if hasattr(Xt_train, "toarray") else Xt_train[:, :n_num]
    print("\nScaled numeric columns on the training set (mean should be ~0 and std ~1):")
    print(pd.DataFrame({"mean": dense_train.mean(axis=0), "std": dense_train.std(axis=0)}, index=NUMERIC).round(3).to_string())

    freq_cwe = set(ohe.categories_[-1]) - set(ohe.infrequent_categories_[-1] if ohe.infrequent_categories_[-1] is not None else [])
    for label, d in (("validation", val), ("test", test)):
        unseen = (~d["cwe_category"].fillna("missing").isin(freq_cwe)).mean() * 100
        print(f"{label}: {unseen:.1f}% of rows have a CWE that is infrequent or unseen in the training data")
    print(f"Missing CWE: train {X_train['cwe_category'].isna().mean() * 100:.2f}% | test {X_test['cwe_category'].isna().mean() * 100:.2f}%")

    # Sanity: no NaN left after preprocessing
    for label, M in (("train", Xt_train), ("val", Xt_val), ("test", Xt_test)):
        data = M.data if hasattr(M, "data") else M
        assert not np.isnan(data).any(), f"NaN left in {label}"

    # 4. Save
    MODELS.mkdir(exist_ok=True)
    pipe_path = MODELS / "preprocessor.joblib"
    joblib.dump(pre, pipe_path)
    feature_names = pre.get_feature_names_out().tolist()
    (MODELS / "feature_names.json").write_text(json.dumps(feature_names, indent=1), encoding="utf-8")
    info = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "split": {"train_years": [TRAIN_FIRST, TRAIN_LAST], "validation_year": VAL_YEAR, "test_year": TEST_YEAR,
                  "held_out": f"{TEST_YEAR + 1} and later (labels still maturing)"},
        "set_sizes": rows,
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "use_published_year": USE_PUBLISHED_YEAR,
        "onehot_min_frequency": MIN_FREQUENCY,
        "output_columns": int(Xt_train.shape[1]),
        "categorical_summary": cat_rows,
        "fitted_on": "training years only",
        "pipeline_file": pipe_path.relative_to(ROOT).as_posix(),
        "pipeline_size_kb": round(pipe_path.stat().st_size / 1024, 1),
    }
    (META_DIR / "preprocessing_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nSaved {pipe_path} ({info['pipeline_size_kb']} KB), models/feature_names.json and data/metadata/preprocessing_summary.json")

    # Reload check
    again = joblib.load(pipe_path)
    assert (again.transform(X_test) != Xt_test).nnz == 0, "reloaded pipeline gives different output"
    print("Reload check passed: the saved pipeline reproduces the test-set output.")


if __name__ == "__main__":
    main()
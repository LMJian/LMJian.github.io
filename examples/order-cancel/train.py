"""训练集内搜索参数，验证集选阈值，最后只评估一次独立测试集。"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predict import CATEGORICAL, FEATURES, MODEL_PATH, NUMERIC, Order, to_frame

ROOT = Path(__file__).resolve().parent


def evaluate(y, probability, threshold):
    predicted = probability >= threshold
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def main():
    data_path = ROOT / "orders.csv"
    df = pd.read_csv(data_path)
    expected = set(FEATURES + ["order_id", "cancelled"])
    if set(df.columns) != expected or df["order_id"].isna().any() or not df["order_id"].is_unique:
        raise ValueError("CSV 列名不符，或订单编号为空/重复")
    if not df["cancelled"].isin([0, 1]).all() or df["cancelled"].nunique() != 2:
        raise ValueError("标签必须完整，且同时包含 0 和 1")
    records = df[FEATURES].astype(object).where(df[FEATURES].notna(), None).to_dict("records")
    # CSV 中整数列有空值时会读成浮点；允许 30.0 转回整数，但拒绝 30.5。
    X = to_frame([Order.model_validate(row, strict=False) for row in records])
    y = df["cancelled"].astype(int)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=42
    )

    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ])
    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=1000, solver="lbfgs")),
    ])
    search = GridSearchCV(
        pipeline,
        {"model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]},
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=1, error_score="raise",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        search.fit(X_train, y_train)
    model = search.best_estimator_
    positive_index = list(model.classes_).index(1)
    valid_probability = model.predict_proba(X_valid)[:, positive_index]
    threshold = float(max(
        np.linspace(0.1, 0.9, 17),
        key=lambda t: f1_score(y_valid, valid_probability >= t, zero_division=0),
    ))

    test_probability = model.predict_proba(X_test)[:, positive_index]
    report = {
        "rows": {"train": len(y_train), "valid": len(y_valid), "test": len(y_test)},
        "best_params": search.best_params_,
        "cv_roc_auc": float(search.best_score_),
        "threshold": threshold,
        "test_positive_rate": float(y_test.mean()),
        "baseline": evaluate(y_test, np.zeros(len(y_test)), 0.5),
        "model": evaluate(y_test, test_probability, threshold),
    }
    bundle = {
        "pipeline": model, "threshold": threshold, "features": FEATURES,
        "model_version": "order-cancel-demo-v1", "evaluation": report,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("numpy", "pandas", "scikit-learn", "joblib")},
    }
    joblib.dump(bundle, MODEL_PATH)
    (ROOT / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("已保存 model.joblib 和 evaluation.json")


if __name__ == "__main__":
    main()

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

import mlflow
import mlflow.sklearn
import dagshub


DATA_PATH = "../telco_preprocessed/telco_preprocessed.csv"  # sesuaikan kalau beda
EXPERIMENT_NAME = "Telco-Churn-Tuning-DagsHub"


def save_confusion_matrix(cm, out_path):
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_roc_curve(y_true, y_proba, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    dagshub.init(repo_owner="Watterprof", repo_name="MSML_MAHDI-SHIDQI_3", mlflow=True)

    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base = RandomForestClassifier(random_state=42, n_jobs=-1)

    param_dist = {
        "n_estimators": [200, 400, 600],
        "max_depth": [None, 8, 12, 16],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=12,
        scoring="roc_auc",
        cv=3,
        random_state=42,
        n_jobs=-1
    )

    # ===== 5) Manual logging (NO autolog) =====
    with mlflow.start_run():
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        # log params terbaik
        mlflow.log_params(search.best_params_)

        # prediksi
        y_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        # metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)

        # log model
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        # ===== 6) Artefak tambahan (minimal 2) =====
        os.makedirs("extra_artifacts", exist_ok=True)

        # (A) confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_path = "extra_artifacts/confusion_matrix.png"
        save_confusion_matrix(cm, cm_path)
        mlflow.log_artifact(cm_path)

        # (B) sample predictions
        sample_path = "extra_artifacts/pred_samples.csv"
        pd.DataFrame({
            "y_true": y_test.values[:300],
            "proba": y_proba[:300],
            "y_pred": y_pred[:300]
        }).to_csv(sample_path, index=False)
        mlflow.log_artifact(sample_path)

        # (C) ROC curve (bonus artefak)
        roc_path = "extra_artifacts/roc_curve.png"
        save_roc_curve(y_test, y_proba, roc_path)
        mlflow.log_artifact(roc_path)

        # (D) summary json (bonus)
        summary_path = "extra_artifacts/summary.json"
        with open(summary_path, "w") as f:
            json.dump({"best_params": search.best_params_, "metrics": metrics}, f, indent=2)
        mlflow.log_artifact(summary_path)

        print("✅ Done. Metrics:", metrics)


if __name__ == "__main__":
    main()

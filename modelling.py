from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "preprocessing" / "telco_preprocessed.csv"

MLRUNS_DIR = BASE_DIR / "mlruns"
mlflow.set_tracking_uri(f"file:{MLRUNS_DIR.as_posix()}")
mlflow.set_experiment("Eksperimen_SML_K2")

df = pd.read_csv(DATA_PATH)

TARGET_COL = "Churn" if "Churn" in df.columns else "target"
y = df[TARGET_COL]

y = pd.to_numeric(y, errors="coerce")

y = (y >= 0.5).astype(int)

X = df.drop(columns=[TARGET_COL])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.sklearn.autolog(log_models=True)

with mlflow.start_run():
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    mlflow.log_metric("accuracy_manual", accuracy_score(y_test, pred))
    mlflow.log_metric("f1_manual", f1_score(y_test, pred))
    mlflow.log_metric("roc_auc_manual", roc_auc_score(y_test, proba))

print("Done. MLruns:", MLRUNS_DIR)

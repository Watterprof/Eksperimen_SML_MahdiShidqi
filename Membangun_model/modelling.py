import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow
import mlflow.sklearn

# 1. Load data
DATA_PATH = "../telco_preprocessed/telco_preprocessed.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["target"])
y = df["target"]

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. MLflow experiment
mlflow.set_experiment("Telco-Churn-Baseline")

# 4. Autolog
mlflow.sklearn.autolog()

with mlflow.start_run():
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("Accuracy:", acc)
    print("ROC-AUC:", auc)

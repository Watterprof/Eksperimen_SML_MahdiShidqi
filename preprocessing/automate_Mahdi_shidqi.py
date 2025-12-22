import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

RAW_PATH = "telco_raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT_PATH = "telco_preprocessed/telco_preprocessed.csv"

def run():
    df = pd.read_csv(RAW_PATH)

    df = df.drop_duplicates().copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "Churn" not in df.columns:
        raise ValueError("Kolom 'Churn' tidak ditemukan. Pastikan dataset telco churn yang benar.")
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])

    num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object","bool"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    X_prep = preprocessor.fit_transform(X)

    X_dense = X_prep.toarray() if hasattr(X_prep, "toarray") else X_prep
    df_out = pd.DataFrame(X_dense)
    df_out["target"] = y.values

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df_out.to_csv(OUT_PATH, index=False)

    print(f"✅ Saved preprocessed dataset to: {OUT_PATH}")
    print("Shape:", df_out.shape)

if __name__ == "__main__":
    run()

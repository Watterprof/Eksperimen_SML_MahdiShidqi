from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = BASE_DIR / "telco_raw.csv"   
OUTPUT_PATH = BASE_DIR / "preprocessing" / "telco_preprocessed.csv"

def run_preprocessing():
    print("Load dataset from:", RAW_PATH)
    df = pd.read_csv(RAW_PATH)

    print("Initial shape:", df.shape)

    df = df.drop_duplicates()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    if "Churn" not in df.columns:
        raise ValueError("Kolom target 'Churn' tidak ditemukan")

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    cat_cols = df.select_dtypes(include=["object"]).columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    scaler = StandardScaler()
    df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

    df_encoded.to_csv(OUTPUT_PATH, index=False)

    print("Preprocessing selesai.")
    print("Final shape:", df_encoded.shape)
    print("Saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    run_preprocessing()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression

def read_receipts_csv(path: str) -> pd.DataFrame:
    """
    Citește robust un CSV care poate fi delimitat cu , sau ;.
    Dacă detectează că a intrat totul într-o singură coloană, reîncearcă cu alt separator.
    """
    df = pd.read_csv(path)

    if df.shape[1] == 1 and isinstance(df.columns[0], str) and "," in df.columns[0]:
        df = pd.read_csv(path, sep=",")

    if df.shape[1] == 1 and isinstance(df.columns[0], str) and ";" in df.columns[0]:
        df = pd.read_csv(path, sep=";")

    return df

def build_dataset_for_crazy_sauce_given_schnitzel(
    df: pd.DataFrame,
    schnitzel_name: str = "Crazy Schnitzel",
    sauce_name: str = "Crazy Sauce",
    product_col: str = "retail_product_name",
    receipt_col: str = "id_bon",
    date_col: str = "data_bon",
    price_col: str = "SalePriceWithVAT",
    product_id_col: str = "retail_product_id",
    use_binary_products: bool = False,
):
    # --- 0) Validări minime ---
    required = [product_col, receipt_col, date_col, price_col, product_id_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Lipsesc coloane: {missing}. Coloane disponibile: {list(df.columns)}")

    df = df.copy()

    # curățare nume produs (important!)
    df[product_col] = df[product_col].astype(str).str.strip()

    # tipuri
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce").fillna(0.0)

    # receipt id consistent
    # (dacă nu merge int, îl lăsăm string)
    try:
        df[receipt_col] = df[receipt_col].astype("int64")
    except Exception:
        df[receipt_col] = df[receipt_col].astype(str)

    # --- 1) Bonuri care conțin schnitzel ---
    schnitzel_bons = (
        df.loc[df[product_col].eq(schnitzel_name), receipt_col]
          .dropna()
          .drop_duplicates()
          .tolist()
    )
    if len(schnitzel_bons) == 0:
        return pd.DataFrame(), pd.Series(dtype=int, name="y")

    df_cs = df[df[receipt_col].isin(schnitzel_bons)].copy()
    master_index = pd.Index(schnitzel_bons, name=receipt_col)

    # --- 2) y ---
    y = (
        df_cs.groupby(receipt_col)[product_col]
             .apply(lambda s: int((s == sauce_name).any()))
             .reindex(master_index, fill_value=0)
    )
    y.name = "y"

    # --- 3) Vector produse (fără schnitzel și sauce) ---
    excluded = {schnitzel_name, sauce_name}
    df_products = df_cs[~df_cs[product_col].isin(excluded)].copy()

    # AICI e partea robustă: crosstab
    product_counts = pd.crosstab(
        index=df_products[receipt_col],
        columns=df_products[product_col]
    ).reindex(master_index, fill_value=0)

    if use_binary_products:
        product_counts = (product_counts > 0).astype(int)

    # --- 4) Agregări pe bon ---
    agg_features = (
        df_cs.groupby(receipt_col).agg(
            cart_size=(product_col, "size"),
            distinct_products=(product_id_col, "nunique"),
            total_value=(price_col, "sum")
        )
        .reindex(master_index)
        .fillna(0)
    )

    # --- 5) Timp ---
    receipt_time = (
        df_cs.groupby(receipt_col)[date_col].min()
             .reindex(master_index)
    )
    day_of_week = (receipt_time.dt.weekday + 1).fillna(0).astype(int)
    is_weekend = day_of_week.isin([6, 7]).astype(int)

    time_features = pd.DataFrame(
        {"day_of_week": day_of_week, "is_weekend": is_weekend},
        index=master_index
    )

    # --- 6) X final ---
    X = product_counts.join(agg_features).join(time_features)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # sanity
    assert X.index.equals(y.index), "Index mismatch între X și y!"

    return X, y

def split_and_standardize(X, y, test_size=0.2, random_state=43):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test, y_train, y_test

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegressionGD:
    def __init__(self, lr=0.05, n_iter=2000):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        self.w = np.zeros(X.shape[1])

        for _ in range(self.n_iter):
            z = X @ self.w
            y_hat = sigmoid(z)
            grad = X.T @ (y_hat - y) / len(y)
            self.w -= self.lr * grad

    def predict_proba(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return sigmoid(X @ self.w)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

def evaluate_model(y_true, y_pred, y_proba, label="Model"):
    print(f"\n=== {label} ===")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1       :", f1_score(y_true, y_pred))
    print("ROC-AUC  :", roc_auc_score(y_true, y_proba))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

def train_sklearn_logreg(X_train, y_train, X_test):
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def main():
    df = read_receipts_csv("ap_dataset.csv")

    X, y = build_dataset_for_crazy_sauce_given_schnitzel(
        df,
        use_binary_products=False
    )

    X_np = X.values
    y_np = y.values

    X_train, X_test, y_train, y_test = split_and_standardize(X_np, y_np)

    model = LogisticRegressionGD(lr=0.05, n_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    evaluate_model(y_test, y_pred, y_proba, label="LogReg (GD, implementare proprie)")

    baseline_pred = np.zeros_like(y_test)
    baseline_proba = baseline_pred.astype(float)
    evaluate_model(y_test, baseline_pred, baseline_proba, label="Baseline (majority class)")

    sk_pred, sk_proba = train_sklearn_logreg(X_train, y_train, X_test)
    evaluate_model(y_test, sk_pred, sk_proba, label="Scikit-learn LogisticRegression")

    coef = pd.Series(model.w[1:], index=X.columns).sort_values(ascending=False)

    print("\nTop features pozitive:")
    print(coef.head(10))

    print("\nTop features negative:")
    print(coef.tail(10))



if __name__ == "__main__":
    main()
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score



def read_receipts_csv(path: str) -> pd.DataFrame:
    """
    Reads a CSV that may be delimited by ',' or ';'. If pandas reads it as a single column,
    re-try with the other delimiter.
    """
    df = pd.read_csv(path)

    if df.shape[1] == 1 and isinstance(df.columns[0], str):
        header = df.columns[0]
        if "," in header:
            df = pd.read_csv(path, sep=",")
        elif ";" in header:
            df = pd.read_csv(path, sep=";")

    return df



def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegressionGD:
    def __init__(self, lr: float = 0.05, n_iter: int = 2000):
        self.lr = lr
        self.n_iter = n_iter
        self.w: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xb = np.c_[np.ones(X.shape[0]), X]
        self.w = np.zeros(Xb.shape[1], dtype=float)

        for _ in range(self.n_iter):
            z = Xb @ self.w
            y_hat = sigmoid(z)
            grad = (Xb.T @ (y_hat - y)) / len(y)
            self.w -= self.lr * grad

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model not fitted.")
        Xb = np.c_[np.ones(X.shape[0]), X]
        return sigmoid(Xb @ self.w)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)



def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


def build_features_for_sauce(
    df: pd.DataFrame,
    sauce_name: str,
    sauces_list: list[str],
    *,
    product_col: str = "retail_product_name",
    receipt_col: str = "id_bon",
    date_col: str = "data_bon",
    price_col: str = "SalePriceWithVAT",
    product_id_col: str = "retail_product_id",
    use_binary_products: bool = False,
    remove_all_sauces_from_features: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns:
      X_s: receipt-level features
      y_s: receipt-level labels for sauce_name
    """

    df = df.copy()

    df[product_col] = df[product_col].astype(str).str.strip()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce").fillna(0.0)

    try:
        df[receipt_col] = df[receipt_col].astype("int64")
    except Exception:
        df[receipt_col] = df[receipt_col].astype(str)

    receipt_ids = df[receipt_col].dropna().drop_duplicates().tolist()
    master_index = pd.Index(receipt_ids, name=receipt_col)

    y_s = (
        df.groupby(receipt_col)[product_col]
          .apply(lambda s: int((s == sauce_name).any()))
          .reindex(master_index, fill_value=0)
    )
    y_s.name = f"y__{sauce_name}"

    if remove_all_sauces_from_features:
        excluded = set(sauces_list)
    else:
        excluded = {sauce_name}

    df_feat = df[~df[product_col].isin(excluded)].copy()

    product_counts = pd.crosstab(
        index=df_feat[receipt_col],
        columns=df_feat[product_col]
    ).reindex(master_index, fill_value=0)

    if use_binary_products:
        product_counts = (product_counts > 0).astype(int)

    agg = (
        df_feat.groupby(receipt_col).agg(
            cart_size=(product_col, "size"),
            distinct_products=(product_id_col, "nunique"),
            total_value=(price_col, "sum"),
        )
        .reindex(master_index)
        .fillna(0)
    )

    receipt_time = (
        df.groupby(receipt_col)[date_col].min()
          .reindex(master_index)
    )
    day_of_week = (receipt_time.dt.weekday + 1).fillna(0).astype(int)
    is_weekend = day_of_week.isin([6, 7]).astype(int)

    time_feat = pd.DataFrame(
        {"day_of_week": day_of_week, "is_weekend": is_weekend},
        index=master_index
    )

    X_s = product_counts.join(agg).join(time_feat)
    X_s = X_s.apply(pd.to_numeric, errors="coerce").fillna(0)

    assert X_s.index.equals(y_s.index)
    return X_s, y_s



def split_receipts(receipt_ids: list, test_size=0.2, random_state=42) -> tuple[list, list]:
    tr, te = train_test_split(
        receipt_ids,
        test_size=test_size,
        random_state=random_state
    )
    return tr, te


def train_models_for_all_sauces(
    df: pd.DataFrame,
    sauces_list: list[str],
    *,
    use_binary_products: bool = False,
    remove_all_sauces_from_features: bool = True,
    lr: float = 0.05,
    n_iter: int = 2000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, dict]:
    """
    Trains one LR model per sauce.

    Returns:
      models[sauce] = {
         'model': LogisticRegressionGD,
         'feature_cols': Index,
         'mean': np.ndarray, 'std': np.ndarray,
         'train_ids': list, 'test_ids': list,
      }
    """
    df_tmp = df.copy()
    try:
        df_tmp["id_bon"] = df_tmp["id_bon"].astype("int64")
    except Exception:
        df_tmp["id_bon"] = df_tmp["id_bon"].astype(str)
    all_receipts = df_tmp["id_bon"].dropna().drop_duplicates().tolist()
    train_ids, test_ids = split_receipts(all_receipts, test_size=test_size, random_state=random_state)

    models: dict[str, dict] = {}

    for s in sauces_list:
        X_s, y_s = build_features_for_sauce(
            df,
            sauce_name=s,
            sauces_list=sauces_list,
            use_binary_products=use_binary_products,
            remove_all_sauces_from_features=remove_all_sauces_from_features,
        )

        X_train_df = X_s.loc[train_ids]
        y_train = y_s.loc[train_ids].values.astype(int)

        X_test_df = X_s.loc[test_ids]
        y_test = y_s.loc[test_ids].values.astype(int)

        X_train = X_train_df.values
        X_test = X_test_df.values
        X_train_std, X_test_std, mean, std = standardize_train_test(X_train, X_test)

        model = LogisticRegressionGD(lr=lr, n_iter=n_iter)
        model.fit(X_train_std, y_train)

        try:
            proba = model.predict_proba(X_test_std)
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, proba)
            else:
                auc = None
        except Exception:
            auc = None

        models[s] = {
            "model": model,
            "feature_cols": X_s.columns,
            "mean": mean,
            "std": std,
            "train_ids": train_ids,
            "test_ids": test_ids,
            "auc": auc,
        }

    return models



def get_receipt_item_sets(
    df: pd.DataFrame,
    *,
    product_col="retail_product_name",
    receipt_col="id_bon"
) -> pd.Series:
    """Returns Series: receipt_id -> set(products)"""
    tmp = df.copy()
    tmp[product_col] = tmp[product_col].astype(str).str.strip()
    try:
        tmp[receipt_col] = tmp[receipt_col].astype("int64")
    except Exception:
        tmp[receipt_col] = tmp[receipt_col].astype(str)

    return tmp.groupby(receipt_col)[product_col].apply(lambda s: set(s.tolist()))


def recommend_top_k_for_receipt(
    models: dict[str, dict],
    X_row_base: pd.DataFrame,
    sauces_in_cart: set[str],
    k: int = 3
) -> list[tuple[str, float]]:
    """
    X_row_base: DataFrame with one row (receipt), columns like some base feature set.
                We'll reindex to each sauce model's columns.
    """
    scores: list[tuple[str, float]] = []

    for sauce, pack in models.items():
        if sauce in sauces_in_cart:
            continue

        cols = pack["feature_cols"]
        x = X_row_base.reindex(columns=cols, fill_value=0).values

        mean = pack["mean"]
        std = pack["std"]
        x_std = (x - mean) / std

        p = float(pack["model"].predict_proba(x_std)[0])
        scores.append((sauce, p))

    scores.sort(key=lambda t: t[1], reverse=True)
    return scores[:k]


def build_features_from_cart(
    cart_items: list[str],
    X_template: pd.DataFrame
) -> pd.DataFrame:
    """
    Construiește un DataFrame cu UN rând, compatibil cu X-ul modelului,
    pornind de la o listă de produse din coș.
    """
    row = pd.DataFrame(0, index=[0], columns=X_template.columns)

    for item in cart_items:
        if item in row.columns:
            row.at[0, item] += 1

    row.at[0, "cart_size"] = len(cart_items)
    row.at[0, "distinct_products"] = len(set(cart_items))
    row.at[0, "total_value"] = 0.0
    row.at[0, "day_of_week"] = 0 
    row.at[0, "is_weekend"] = 0

    return row


def recommend_for_cart(models, cart_items, k=3):
    scores = []

    any_pack = next(iter(models.values()))
    X_template = pd.DataFrame(columns=any_pack["feature_cols"])

    X_cart = build_features_from_cart(cart_items, X_template)

    for sauce, pack in models.items():
        x = X_cart.reindex(columns=pack["feature_cols"], fill_value=0).values
        x_std = (x - pack["mean"]) / pack["std"]

        p = float(pack["model"].predict_proba(x_std)[0])
        scores.append((sauce, p))

    scores.sort(key=lambda t: t[1], reverse=True)
    return scores[:k]


def evaluate_hit_precision_at_k(
    df: pd.DataFrame,
    models: dict[str, dict],
    sauces_list: list[str],
    *,
    k: int = 3,
    product_col="retail_product_name",
    receipt_col="id_bon",
    simulate_cart_without_sauces: bool = True,
    use_binary_products: bool = False,
    remove_all_sauces_from_features: bool = True,
):
    """
    Multi-label evaluation:
      - true_sauces = sauces present in receipt (can be multiple)
      - recommended = top-k sauces not already in cart
      Hit@K = 1 if any true sauce appears in top-k
      Precision@K = |topK ∩ true_sauces| / K
    """

    any_pack = next(iter(models.values()))
    test_ids = any_pack["test_ids"]


    item_sets = get_receipt_item_sets(df, product_col=product_col, receipt_col=receipt_col)


    base_sauce = sauces_list[0]
    X_base, _ = build_features_for_sauce(
        df,
        sauce_name=base_sauce,
        sauces_list=sauces_list,
        use_binary_products=use_binary_products,
        remove_all_sauces_from_features=remove_all_sauces_from_features,
    )

    hits, precs = [], []

    for rid in test_ids:
        if rid not in item_sets.index:
            continue

        items = item_sets.loc[rid]
        true_sauces = set(items).intersection(set(sauces_list))

        if len(true_sauces) == 0:
            continue

        if simulate_cart_without_sauces:
            sauces_in_cart = set()
        else:
            sauces_in_cart = true_sauces.copy()

        X_row = X_base.loc[[rid]]
        topk = recommend_top_k_for_receipt(models, X_row, sauces_in_cart, k=k)
        rec_sauces = {s for s, _ in topk}

        hit = 1 if len(rec_sauces.intersection(true_sauces)) > 0 else 0
        prec = len(rec_sauces.intersection(true_sauces)) / k

        hits.append(hit)
        precs.append(prec)

    hit_at_k = float(np.mean(hits)) if hits else 0.0
    precision_at_k = float(np.mean(precs)) if precs else 0.0
    return hit_at_k, precision_at_k


def popularity_topk_from_train(
    df: pd.DataFrame,
    sauces_list: list[str],
    train_ids: list,
    *,
    product_col="retail_product_name",
    receipt_col="id_bon",
    k: int = 3
) -> list[str]:
    """
    Popularity baseline: Top-K sauces by number of receipts (in TRAIN) containing that sauce.
    """
    tmp = df.copy()
    tmp[product_col] = tmp[product_col].astype(str).str.strip()
    try:
        tmp[receipt_col] = tmp[receipt_col].astype("int64")
    except Exception:
        tmp[receipt_col] = tmp[receipt_col].astype(str)

    tmp = tmp[tmp[receipt_col].isin(train_ids)]
    sauce_counts = (
        tmp[tmp[product_col].isin(sauces_list)]
        .groupby(product_col)[receipt_col]
        .nunique()
        .sort_values(ascending=False)
    )
    return sauce_counts.index.tolist()[:k]


def evaluate_popularity_baseline(
    df: pd.DataFrame,
    sauces_list: list[str],
    test_ids: list,
    topk_sauces: list[str],
    *,
    product_col="retail_product_name",
    receipt_col="id_bon",
    k: int = 3
):
    item_sets = get_receipt_item_sets(df, product_col=product_col, receipt_col=receipt_col)

    rec_sauces = set(topk_sauces)
    hits, precs = [], []

    for rid in test_ids:
        if rid not in item_sets.index:
            continue
        items = item_sets.loc[rid]
        true_sauces = set(items).intersection(set(sauces_list))
        if len(true_sauces) == 0:
            continue

        hit = 1 if len(rec_sauces.intersection(true_sauces)) > 0 else 0
        prec = len(rec_sauces.intersection(true_sauces)) / k

        hits.append(hit)
        precs.append(prec)

    return (float(np.mean(hits)) if hits else 0.0,
            float(np.mean(precs)) if precs else 0.0)




def main():
    df = read_receipts_csv("ap_dataset.csv")

    SAUCES = [
        "Crazy Sauce",
        "Blueberry Sauce",
        "Tomato Sauce",
        "Garlic Sauce",
        "Cheddar Sauce",
        "Pink Sauce",
        "Spicy Sauce",
        "Extra Cheddar Sauce",
    ]

    models = train_models_for_all_sauces(
        df,
        SAUCES,
        use_binary_products=False,
        remove_all_sauces_from_features=True,
        lr=0.05,
        n_iter=2000,
        test_size=0.2,
        random_state=42,
    )

    hit3, prec3 = evaluate_hit_precision_at_k(
        df,
        models,
        SAUCES,
        k=3,
        simulate_cart_without_sauces=True,
        use_binary_products=False,
        remove_all_sauces_from_features=True,
    )
    print(f"\n[Model-based recommender] Hit@3={hit3:.4f}  Precision@3={prec3:.4f}")

    any_pack = next(iter(models.values()))
    train_ids = any_pack["train_ids"]
    test_ids = any_pack["test_ids"]

    top3_pop = popularity_topk_from_train(df, SAUCES, train_ids, k=8)
    hit3_b, prec3_b = evaluate_popularity_baseline(df, SAUCES, test_ids, top3_pop, k=8)
    print(f"[Popularity baseline] Top3={top3_pop}  Hit@3={hit3_b:.4f}  Precision@3={prec3_b:.4f}")

    my_cart = [
        "Baked potatoes",
        "Pepsi Cola 0.25L Doze",
    ]

    topk = recommend_for_cart(models, my_cart, k=8)

    print("\n=== Manual cart recommendation ===")
    print("Cart:", my_cart)
    print("Top-k sauces:")
    for s, p in topk:
        print(f"  {s}: P={p:.3f}")



if __name__ == "__main__":
    main()

"""
Tourism Experience Analytics
src/models.py — Regression, Classification & Recommendation model training.
"""

import warnings
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from sklearn.model_selection     import train_test_split
from sklearn.preprocessing       import StandardScaler, LabelEncoder
from sklearn.linear_model        import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree                import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble            import (RandomForestRegressor, GradientBoostingRegressor,
                                         ExtraTreesRegressor, RandomForestClassifier,
                                         GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.neighbors           import KNeighborsClassifier
from sklearn.metrics             import (mean_squared_error, mean_absolute_error, r2_score,
                                         accuracy_score, f1_score, precision_score, recall_score,
                                         classification_report, confusion_matrix)
from sklearn.metrics.pairwise    import cosine_similarity
from scipy.sparse                import csr_matrix
from scipy.sparse.linalg         import svds

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not installed — skipping.")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed — skipping.")

from src.data_pipeline import CAT_COLS


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def _safe_encode(df_in: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df_in.copy()
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        if col != target_col:
            df[col] = le.fit_transform(df[col].astype(str))
    for col in ["AttractionTypeId", "ContinentId", "RegionId", "CountryId"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()


# ─────────────────────────────────────────────────────────────────
# 1.  REGRESSION — Predict Rating
# ─────────────────────────────────────────────────────────────────

SCALED_REGS = {"Linear Regression", "Ridge Regression", "Lasso Regression"}
NUMPY_REGS  = {"XGBoost", "LightGBM"}


def build_reg_features(df: pd.DataFrame) -> list[str]:
    base = (
        [c + "_enc" for c in CAT_COLS if c + "_enc" in df.columns] +
        ["VisitYear", "VisitMonth", "UserAvgRating", "AttrAvgRating",
         "AttrVisitCount", "AttractionTypeId", "ContinentId", "RegionId", "CountryId"]
    )
    return [f for f in base if f in df.columns]


def train_regression(df: pd.DataFrame):
    """Train all regression models. Returns (best_model, scaler, results_df, features)."""
    REG_FEATURES = build_reg_features(df)
    TARGET = "Rating"

    reg_df = _safe_encode(df[REG_FEATURES + [TARGET]].dropna().copy(), TARGET)
    X = reg_df[REG_FEATURES]
    y = reg_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_r    = StandardScaler()
    X_train_s   = scaler_r.fit_transform(X_train)
    X_test_s    = scaler_r.transform(X_test)

    reg_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=1.0),
        "Lasso Regression":  Lasso(alpha=0.01),
        "Decision Tree":     DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Extra Trees":       ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    }
    if HAS_XGB:
        reg_models["XGBoost"] = xgb.XGBRegressor(n_estimators=100, random_state=42,
                                                   verbosity=0, enable_categorical=False)
    if HAS_LGBM:
        reg_models["LightGBM"] = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)

    results = []
    print(f"\n{'Model':<25}  {'RMSE':>7}  {'MAE':>7}  {'R²':>7}")
    print("─" * 55)
    for name, model in reg_models.items():
        Xtr = X_train_s if name in SCALED_REGS else X_train
        Xte = X_test_s  if name in SCALED_REGS else X_test
        if name in NUMPY_REGS:
            Xtr = np.array(Xtr, dtype=np.float32)
            Xte = np.array(Xte, dtype=np.float32)
        try:
            model.fit(Xtr, y_train)
            preds = model.predict(Xte)
            rmse  = np.sqrt(mean_squared_error(y_test, preds))
            mae   = mean_absolute_error(y_test, preds)
            r2    = r2_score(y_test, preds)
            results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R²": r2})
            print(f"  {name:<25}  {rmse:>7.4f}  {mae:>7.4f}  {r2:>7.4f}")
        except Exception as e:
            print(f"  {name:<25}  ✗ {e}")

    results_df    = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    best_name     = results_df.iloc[0]["Model"]
    best_model    = reg_models[best_name]
    print(f"\n✅  Best Regression Model: {best_name}")
    return best_model, scaler_r, results_df, REG_FEATURES, reg_models


# ─────────────────────────────────────────────────────────────────
# 2.  CLASSIFICATION — Predict Visit Mode
# ─────────────────────────────────────────────────────────────────

SCALED_CLFS = {"Logistic Regression", "K-Nearest Neighbors"}
NUMPY_CLFS  = {"XGBoost", "LightGBM"}


def build_clf_features(df: pd.DataFrame) -> list[str]:
    base = (
        [c + "_enc" for c in CAT_COLS if c + "_enc" in df.columns] +
        ["VisitYear", "VisitMonth", "UserAvgRating", "AttrAvgRating",
         "AttrVisitCount", "AttractionTypeId", "ContinentId", "RegionId", "CountryId"]
    )
    return [f for f in base if f in df.columns]


def train_classification(df: pd.DataFrame, le_mode: LabelEncoder):
    """Train all classifiers. Returns (best_model, scaler, results_df, features, clf_models)."""
    CLF_FEATURES = build_clf_features(df)
    TARGET = "VisitModeLabel"

    clf_df = _safe_encode(df[CLF_FEATURES + [TARGET]].dropna().copy(), TARGET)
    X = clf_df[CLF_FEATURES]
    y = clf_df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler_c  = StandardScaler()
    X_train_s = scaler_c.fit_transform(X_train)
    X_test_s  = scaler_c.transform(X_test)

    n_classes = len(np.unique(y))
    clf_models = {
        "Logistic Regression":  LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree":        DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Extra Trees":          ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "K-Nearest Neighbors":  KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }
    if HAS_XGB:
        clf_models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=100, random_state=42, verbosity=0,
            eval_metric="mlogloss", enable_categorical=False, num_class=n_classes)
    if HAS_LGBM:
        clf_models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=100, random_state=42, verbose=-1)

    results = []
    print(f"\n{'Model':<25}  {'Acc':>7}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}")
    print("─" * 65)
    for name, model in clf_models.items():
        Xtr = X_train_s if name in SCALED_CLFS else X_train
        Xte = X_test_s  if name in SCALED_CLFS else X_test
        if name in NUMPY_CLFS:
            Xtr = np.array(Xtr, dtype=np.float32)
            Xte = np.array(Xte, dtype=np.float32)
        try:
            model.fit(Xtr, y_train)
            preds = model.predict(Xte)
            acc  = accuracy_score(y_test, preds)
            f1   = f1_score(y_test, preds, average="weighted", zero_division=0)
            prec = precision_score(y_test, preds, average="weighted", zero_division=0)
            rec  = recall_score(y_test, preds, average="weighted", zero_division=0)
            results.append({"Model": name, "Accuracy": acc, "F1": f1, "Precision": prec, "Recall": rec})
            print(f"  {name:<25}  {acc:>7.4f}  {f1:>7.4f}  {prec:>7.4f}  {rec:>7.4f}")
        except Exception as e:
            print(f"  {name:<25}  ✗ {e}")

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
    best_name  = results_df.iloc[0]["Model"]
    best_model = clf_models[best_name]
    print(f"\n✅  Best Classifier: {best_name}")

    # Print classification report
    use_scaled = best_name in SCALED_CLFS
    Xte = X_test_s if use_scaled else X_test
    if best_name in NUMPY_CLFS:
        Xte = np.array(Xte, dtype=np.float32)
    preds_best = best_model.predict(Xte)
    print("\nClassification Report:")
    print(classification_report(y_test, preds_best, target_names=le_mode.classes_, zero_division=0))

    return best_model, scaler_c, results_df, CLF_FEATURES, clf_models


# ─────────────────────────────────────────────────────────────────
# 3.  RECOMMENDATION — Collaborative + Content-Based Filtering
# ─────────────────────────────────────────────────────────────────

def train_recommendation(df: pd.DataFrame, item: pd.DataFrame, attr_type: pd.DataFrame):
    """
    Build SVD collaborative filtering and cosine-similarity content-based models.
    Returns (predicted_ratings_df, user_item_matrix, cos_sim_df, rmse_cf, mae_cf).
    """
    TOP_USERS = 5000
    TOP_ITEMS = 2000

    top_users = df["UserId"].value_counts().head(TOP_USERS).index
    top_items = df["AttractionId"].value_counts().head(TOP_ITEMS).index
    cf_df = df[df["UserId"].isin(top_users) & df["AttractionId"].isin(top_items)]

    user_item_matrix = cf_df.pivot_table(
        index="UserId", columns="AttractionId", values="Rating", aggfunc="mean"
    ).fillna(0)
    print(f"User-Item matrix: {user_item_matrix.shape}")

    # SVD
    sparse_matrix = csr_matrix(user_item_matrix.values)
    k = min(50, min(sparse_matrix.shape) - 1)
    U, sigma, Vt = svds(sparse_matrix, k=k)
    sigma_diag = np.diag(sigma)
    all_predicted = np.dot(np.dot(U, sigma_diag), Vt)
    predicted_ratings_df = pd.DataFrame(
        all_predicted, index=user_item_matrix.index, columns=user_item_matrix.columns)

    # Evaluate CF with known ratings (sample)
    known = cf_df[["UserId", "AttractionId", "Rating"]].copy()
    known = known[known["UserId"].isin(predicted_ratings_df.index) &
                  known["AttractionId"].isin(predicted_ratings_df.columns)].copy()
    known["predicted"] = known.apply(
        lambda r: predicted_ratings_df.loc[r["UserId"], r["AttractionId"]], axis=1)
    rmse_cf = np.sqrt(mean_squared_error(known["Rating"], known["predicted"]))
    mae_cf  = mean_absolute_error(known["Rating"], known["predicted"])
    print(f"CF  RMSE: {rmse_cf:.4f}  |  MAE: {mae_cf:.4f}")

    # Content-based — cosine similarity on attraction features
    item_features = item.copy()
    item_features = item_features.merge(attr_type, on="AttractionTypeId", how="left")

    # Only keep numeric ID columns — never include text columns like Attraction name
    candidate_cols = ["AttractionCityId", "AttractionTypeId"]
    feat_cols = [c for c in candidate_cols if c in item_features.columns]

    if feat_cols:
        feat_matrix = item_features[feat_cols].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0)
        sim_matrix  = cosine_similarity(feat_matrix)
        cos_sim_df  = pd.DataFrame(
            sim_matrix, index=item_features["AttractionId"],
            columns=item_features["AttractionId"])
    else:
        cos_sim_df = pd.DataFrame()

    print("✅  Recommendation models built.")
    return predicted_ratings_df, user_item_matrix, cos_sim_df, rmse_cf, mae_cf


# ─────────────────────────────────────────────────────────────────
# 4.  SAVE / LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────────

def save_artifacts(path: str, **kwargs):
    """Pickle all keyword arguments to a single file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(kwargs, f)
    print(f"✅  Artifacts saved to {path}")


def load_artifacts(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)

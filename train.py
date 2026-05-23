"""
Tourism Experience Analytics
train.py — End-to-end training script. Run once before launching the Streamlit app.

Usage:
    python train.py --data ./data
"""

import argparse
import sys
import os

# allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.data_pipeline import run_pipeline
from src.models import (
    train_regression, train_classification, train_recommendation,
    save_artifacts, SCALED_REGS, SCALED_CLFS
)

ARTIFACTS_PATH = "models/tea_artifacts.pkl"


def main(data_path: str):
    print("=" * 60)
    print("  Tourism Experience Analytics — Training Pipeline")
    print("=" * 60)

    # ── 1. Load & preprocess ──────────────────────────────────────
    df, item, attr_type, le_mode, le_dict = run_pipeline(data_path)

    # ── 2. Regression ─────────────────────────────────────────────
    print("\n[REGRESSION]")
    best_reg_model, scaler_r, reg_results_df, REG_FEATURES, reg_models = train_regression(df)

    # ── 3. Classification ─────────────────────────────────────────
    print("\n[CLASSIFICATION]")
    best_clf_model, scaler_c, clf_results_df, CLF_FEATURES, clf_models = train_classification(df, le_mode)

    # ── 4. Recommendation ─────────────────────────────────────────
    print("\n[RECOMMENDATION]")
    predicted_ratings_df, user_item_matrix, cos_sim_df, rmse_cf, mae_cf = \
        train_recommendation(df, item, attr_type)

    # ── 5. Save ───────────────────────────────────────────────────
    print("\n[SAVING ARTIFACTS]")
    save_artifacts(
        ARTIFACTS_PATH,
        df=df,
        item=item,
        attr_type=attr_type,
        le_mode=le_mode,
        le_dict=le_dict,
        REG_FEATURES=REG_FEATURES,
        CLF_FEATURES=CLF_FEATURES,
        reg_results_df=reg_results_df,
        clf_results_df=clf_results_df,
        best_reg_model=best_reg_model,
        best_clf_model=best_clf_model,
        scaler_r=scaler_r,
        scaler_c=scaler_c,
        predicted_ratings_df=predicted_ratings_df,
        user_item_matrix=user_item_matrix,
        cos_sim_df=cos_sim_df,
        rmse_cf=rmse_cf,
        mae_cf=mae_cf,
        SCALED_REGS=SCALED_REGS,
        SCALED_CLFS=SCALED_CLFS,
    )

    print("\n" + "=" * 60)
    print("  ✅  Training complete!")
    print(f"  ✅  Artifacts saved → {ARTIFACTS_PATH}")
    print("  ➡   Now run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tourism Analytics models.")
    parser.add_argument("--data", default="./data",
                        help="Path to folder containing the Excel dataset files.")
    args = parser.parse_args()
    main(args.data)

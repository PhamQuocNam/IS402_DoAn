"""
train.py — Azure ML Training Script
Fozzy Group Retail Sales Forecasting (LightGBM)

Usage (local):
    python train.py --data_dir ./data --model_dir ./outputs

Usage (Azure ML):
    Được gọi tự động bởi Azure ML Job, tham số truyền qua args.
"""

import os
import gc
import json
import argparse
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import mlflow
import mlflow.lightgbm
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Logger ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────
TARGET      = "sales"
FEAT_COLS   = [
    "geoCluster", "SKU", "productCategoryId", "lagerUnitTypeId",
    "day_sin", "day_cos", "month_sin", "month_cos",
    "year", "commodity_group", "trademark",
]
CAT_COLS    = ["commodity_group", "lagerUnitTypeId"]
SKU_COLS    = ["SKU", "productCategoryId", "lagerUnitTypeId", "trademark", "commodity_group"]
TRAIN_COLS  = ["geoCluster", "SKU", "date", "sales"]


# ── Helpers ──────────────────────────────────────────────────────────────────
def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Ép kiểu integer/float xuống loại nhỏ nhất để tiết kiệm RAM."""
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def load_and_preprocess(data_dir: str):
    """
    Đọc raw CSV, tiền xử lý, feature engineering, merge, split.
    Trả về (X_train, y_train, X_valid, y_valid, X_test, y_test).
    """
    logger.info("── Step 1: Load raw data ──")
    sku_df = pd.read_csv(os.path.join(data_dir, "sku_final.csv"), usecols=SKU_COLS)
    sku_df["SKU"]               = sku_df["SKU"].astype("int32")
    sku_df["productCategoryId"] = sku_df["productCategoryId"].astype("Int16")
    sku_df["trademark"]         = sku_df["trademark"].astype("Int16")
    sku_df["commodity_group"]   = sku_df["commodity_group"].astype("category")
    sku_df["lagerUnitTypeId"]   = sku_df["lagerUnitTypeId"].astype("category")
    sku_df = downcast(sku_df)

    train_df = pd.read_csv(
        os.path.join(data_dir, "train_final.csv"),
        usecols=TRAIN_COLS,
        parse_dates=["date"],
    )
    train_df["geoCluster"] = train_df["geoCluster"].astype("int16")
    train_df["SKU"]        = train_df["SKU"].astype("int32")
    train_df["sales"]      = train_df["sales"].astype("float32")
    train_df["date"]       = train_df["date"].dt.floor("D")
    train_df = downcast(train_df)
    logger.info(f"  train_df shape: {train_df.shape}")

    logger.info("── Step 2: Null handling ──")
    # trademark: điền theo mode (productCategoryId × commodity_group)
    mode_by_pair = (
        sku_df.groupby(["productCategoryId", "commodity_group"], observed=True)["trademark"]
        .agg(lambda x: x.mode().iloc[0] if x.notna().any() else pd.NA)
    )
    pair_fill = sku_df.set_index(["productCategoryId", "commodity_group"]).index.map(mode_by_pair)
    pair_fill = pd.Series(pair_fill, index=sku_df.index)
    sku_df["trademark"] = sku_df["trademark"].fillna(pair_fill).fillna(0).astype("Int16")

    # sales: fill 0 (không bán → 0)
    train_df["sales"] = train_df["sales"].fillna(0)

    logger.info("── Step 3: Feature engineering ──")
    train_df["day"]       = train_df["date"].dt.day
    train_df["month"]     = train_df["date"].dt.month
    train_df["year"]      = train_df["date"].dt.year
    train_df["day_sin"]   = np.sin(2 * np.pi * train_df["day"]   / 31).astype("float32")
    train_df["day_cos"]   = np.cos(2 * np.pi * train_df["day"]   / 31).astype("float32")
    train_df["month_sin"] = np.sin(2 * np.pi * train_df["month"] / 12).astype("float32")
    train_df["month_cos"] = np.cos(2 * np.pi * train_df["month"] / 12).astype("float32")
    train_df = train_df.drop(columns=["day", "month"])
    train_df = downcast(train_df)

    logger.info("── Step 4: Merge ──")
    df = train_df.merge(sku_df, on="SKU", how="left")
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    del train_df, sku_df
    gc.collect()

    logger.info("── Step 5: Chronological split (90 / 5 / 5) ──")
    df = df.sort_values("date").reset_index(drop=True)
    n          = len(df)
    train_end  = int(n * 0.90)
    valid_end  = int(n * 0.95)

    train_df = df.iloc[:train_end].copy()
    valid_df = df.iloc[train_end:valid_end].copy()
    test_df  = df.iloc[valid_end:].copy()

    logger.info(
        f"  Train: {len(train_df):,} | Valid: {len(valid_df):,} | Test: {len(test_df):,}"
    )
    del df
    gc.collect()

    # Drop date (không dùng trực tiếp làm feature)
    for d in [train_df, valid_df, test_df]:
        d.drop(columns=["date"], inplace=True, errors="ignore")

    X_train, y_train = train_df[FEAT_COLS], train_df[TARGET]
    X_valid, y_valid = valid_df[FEAT_COLS], valid_df[TARGET]
    X_test,  y_test  = test_df[FEAT_COLS],  test_df[TARGET]

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def train(X_train, y_train, X_valid, y_valid, params: dict):
    """Huấn luyện LightGBM, trả về model đã fit."""
    logger.info("── Step 6: Train LightGBM ──")
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(100),
        ],
    )
    return model


def evaluate(model, X_test, y_test) -> dict:
    """Tính MAE và RMSE trên tập test, trả về dict metrics."""
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    logger.info(f"  Test MAE : {mae:.4f}")
    logger.info(f"  Test RMSE: {rmse:.4f}")
    return {"mae": mae, "rmse": rmse}


# ── Main ─────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGBM sales forecast model")
    parser.add_argument("--data_dir",     type=str, default="./raw-data",    help="Thư mục chứa raw CSV")
    parser.add_argument("--model_dir",    type=str, default="./outputs", help="Nơi lưu model")
    parser.add_argument("--n_estimators", type=int, default=1500)
    parser.add_argument("--learning_rate",type=float, default=0.05)
    parser.add_argument("--max_depth",    type=int, default=8)
    parser.add_argument("--num_leaves",   type=int, default=64)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    lgbm_params = dict(
        n_estimators  = args.n_estimators,
        learning_rate = args.learning_rate,
        max_depth     = args.max_depth,
        num_leaves    = args.num_leaves,
        random_state  = args.random_state,
        n_jobs        = -1,
    )

    # Azure ML tự khởi động MLflow tracking; không cần set URI thủ công.
    mlflow.lightgbm.autolog(log_models=False)   # autolog params + metrics

    with mlflow.start_run():
        # Log hyper-params thủ công để đảm bảo hiển thị trên Azure ML UI
        mlflow.log_params(lgbm_params)

        # ── Preprocessing ──────────────────────────────────────────────────
        X_train, y_train, X_valid, y_valid, X_test, y_test = \
            load_and_preprocess(args.data_dir)

        # ── Training ───────────────────────────────────────────────────────
        model = train(X_train, y_train, X_valid, y_valid, lgbm_params)
        del X_train, y_train, X_valid, y_valid
        gc.collect()

        # ── Evaluation ─────────────────────────────────────────────────────
        metrics = evaluate(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        # ── Save model ─────────────────────────────────────────────────────
        model_path = os.path.join(args.model_dir, "lgb_sales_model.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Model saved → {model_path}")

        # ── Save metrics.json (cho CI/CD pipeline đọc) ─────────────────────
        metrics_path = os.path.join(args.model_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved → {metrics_path}")

        # Log artifact để Azure ML có thể tải về / register
        # mlflow.log_artifact(model_path, artifact_path="model")
        # mlflow.log_artifact(metrics_path, artifact_path="model")

    logger.info("Training completed.")


if __name__ == "__main__":
    main()

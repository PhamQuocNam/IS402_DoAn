"""
score.py — Azure ML Inference / Scoring Script
Fozzy Group Retail Sales Forecasting (LightGBM)

Azure ML gọi hai hàm theo thứ tự:
  1. init()      — chạy một lần khi container khởi động.
  2. run(data)   — chạy mỗi request dự đoán.

Input JSON format (single record):
{
    "geoCluster": 1,
    "SKU": 12345,
    "productCategoryId": 10,
    "lagerUnitTypeId": "A",
    "day_sin": 0.203,
    "day_cos": 0.979,
    "month_sin": 0.866,
    "month_cos": 0.5,
    "year": 2024,
    "commodity_group": "food",
    "trademark": 5
}

Hoặc batch (list of records):
[
    { ... },
    { ... }
]
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib

# Azure ML tự inject biến môi trường AZUREML_MODEL_DIR
# trỏ tới thư mục chứa model đã được register.

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# ── Globals — được điền bởi init() ───────────────────────────────────────────
model = None

FEAT_COLS = [
    "geoCluster",
    "SKU",
    "productCategoryId",
    "lagerUnitTypeId",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
    "year",
    "commodity_group",
    "trademark",
]
CAT_COLS = ["commodity_group", "lagerUnitTypeId"]


# ── init ─────────────────────────────────────────────────────────────────────
def init():
    """
    Được Azure ML gọi một lần khi endpoint container khởi động.
    Load model từ AZUREML_MODEL_DIR khi chạy trên Azure,
    hoặc fallback về ./outputs khi chạy local.
    """
    global model

    model_dir = os.getenv("AZUREML_MODEL_DIR", "./outputs")
    logger.info(f"AZUREML_MODEL_DIR / local model_dir: {model_dir}")

    candidate_paths = [
        os.path.join(model_dir, "lgb_sales_model.pkl"),
        os.path.join(model_dir, "model", "lgb_sales_model.pkl"),
        os.path.join(model_dir, "outputs", "lgb_sales_model.pkl"),
    ]

    # Tìm thêm mọi file .pkl bên trong model_dir
    for root, _, files in os.walk(model_dir):
        for filename in files:
            if filename.endswith(".pkl"):
                candidate_paths.append(os.path.join(root, filename))

    logger.info(f"Candidate model paths: {candidate_paths}")

    for path in candidate_paths:
        if os.path.exists(path):
            logger.info(f"Loading model from: {path}")
            model = joblib.load(path)
            logger.info("Model loaded successfully.")
            return

    raise FileNotFoundError(
        f"Could not find lgb_sales_model.pkl under model_dir={model_dir}. "
        f"Checked paths: {candidate_paths}"
    )


# ── run ──────────────────────────────────────────────────────────────────────
def run(raw_data: str) -> str:
    """
    Được Azure ML gọi cho mỗi HTTP request tới endpoint.

    Parameters
    ----------
    raw_data : str
        JSON string — một dict hoặc list of dicts chứa các feature.

    Returns
    -------
    str
        JSON string chứa danh sách dự đoán và (nếu có) input echo.
    """
    try:
        # ── Parse input ──────────────────────────────────────────────────
        data = json.loads(raw_data)

        # Chuẩn hoá: luôn làm việc với list of records
        if isinstance(data, dict):
            records = [data]
        elif isinstance(data, list):
            records = data
        else:
            raise ValueError(f"Unsupported input type: {type(data)}")

        # ── Build DataFrame ───────────────────────────────────────────────
        df = pd.DataFrame(records)

        # Kiểm tra feature đầy đủ
        missing = set(FEAT_COLS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing features in input: {missing}")

        df = df[FEAT_COLS]  # đảm bảo thứ tự đúng

        # ── Type casting (giống lúc training) ────────────────────────────
        df["geoCluster"] = df["geoCluster"].astype("int16")
        df["SKU"] = df["SKU"].astype("int32")
        df["productCategoryId"] = df["productCategoryId"].astype("Int16")
        df["trademark"] = df["trademark"].astype("Int16")
        df["year"] = df["year"].astype("int16")
        for col in ["day_sin", "day_cos", "month_sin", "month_cos"]:
            df[col] = df[col].astype("float32")
        for col in CAT_COLS:
            df[col] = df[col].astype("category")

        # ── Predict ───────────────────────────────────────────────────────
        predictions = model.predict(df).tolist()

        # ── Format output ─────────────────────────────────────────────────
        response = {
            "predictions": predictions,
            "n_records": len(predictions),
        }
        return json.dumps(response)

    except Exception as exc:
        logger.exception("Prediction failed")
        error_response = {"error": str(exc)}
        return json.dumps(error_response)


# ── Local smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Chạy local để test nhanh trước khi deploy lên Azure.
    Yêu cầu: model đã được lưu tại ./outputs/lgb_sales_model.pkl
    """
    init()

    sample = {
        "geoCluster": 1,
        "SKU": 12345,
        "productCategoryId": 10,
        "lagerUnitTypeId": "A",
        "day_sin": round(np.sin(2 * np.pi * 15 / 31), 6),
        "day_cos": round(np.cos(2 * np.pi * 15 / 31), 6),
        "month_sin": round(np.sin(2 * np.pi * 6 / 12), 6),
        "month_cos": round(np.cos(2 * np.pi * 6 / 12), 6),
        "year": 2024,
        "commodity_group": "food",
        "trademark": 5,
    }

    result = run(json.dumps(sample))
    print("Single record →", result)

    batch = [sample, {**sample, "SKU": 99999, "geoCluster": 2}]
    result_batch = run(json.dumps(batch))
    print("Batch (2 records) →", result_batch)

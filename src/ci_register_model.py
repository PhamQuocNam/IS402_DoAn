"""
ci_register_model.py — CI/CD version of model_registry.py
Tự động đăng ký model từ job outputs, không cần input() thủ công.
Tag model với metrics (MAE, MSE, RMSE) và pipeline build number.

Chạy trong Azure DevOps Pipeline:
    python src/ci_register_model.py
"""

import os
import sys
import json

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential


def main():
    # ── Config ───────────────────────────────────────────────────────────────
    subscription_id = os.environ.get(
        "AZURE_SUBSCRIPTION_ID", "35d715f0-0211-4894-9c18-aea6e5787b86"
    )
    resource_group = os.environ.get("AZURE_RESOURCE_GROUP", "is402_doan")
    workspace_name = os.environ.get("AZURE_ML_WORKSPACE", "machinelearningproject")
    model_name = os.environ.get("MODEL_NAME", "fozzy-lightgbm-sales-model")
    build_id = os.environ.get("BUILD_BUILDID", "local")

    # ── Đọc job name ─────────────────────────────────────────────────────────
    job_output_file = os.environ.get("JOB_OUTPUT_FILE", "job_output.txt")
    with open(job_output_file) as f:
        job_name = f.read().strip()
    print(f"Job name: {job_name}")

    # ── Đọc metrics ──────────────────────────────────────────────────────────
    metrics_tags = {}
    compare_result_file = "compare_result.json"
    if os.path.exists(compare_result_file):
        with open(compare_result_file) as f:
            result = json.load(f)
        metrics_tags = {
            "mae": str(round(result.get("new_mae", 0), 6)),
            "rmse": str(round(result.get("new_rmse", 0), 6)),
        }

    # ── Kết nối Azure ML ─────────────────────────────────────────────────────
    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

    # ── Đường dẫn tới model artifact trong job outputs ────────────────────────
    model_path = (
        f"azureml://jobs/{job_name}/outputs/artifacts/paths/outputs/lgb_sales_model.pkl"
    )

    # ── Tags cho model ───────────────────────────────────────────────────────
    tags = {
        "framework": "lightgbm",
        "task": "regression",
        "dataset": "fozzy-retail-sales",
        "ci_build_id": build_id,
        "training_job": job_name,
        **metrics_tags,
    }

    # ── Đăng ký model ────────────────────────────────────────────────────────
    print(f"Đang đăng ký model: {model_name}")
    print(f"   Path: {model_path}")
    print(f"   Tags: {tags}")

    run_model = Model(
        path=model_path,
        name=model_name,
        description=(
            f"LightGBM sales forecast model. "
            f"MAE={metrics_tags.get('mae', 'N/A')}, "
            f"RMSE={metrics_tags.get('rmse', 'N/A')}. "
            f"Trained by CI pipeline build #{build_id}."
        ),
        type=AssetTypes.CUSTOM_MODEL,
        tags=tags,
    )

    registered = ml_client.models.create_or_update(run_model)
    print(f"Đã đăng ký model thành công!")
    print(f"   Name: {registered.name}")
    print(f"   Version: {registered.version}")

    # ── Lưu version cho deploy script ────────────────────────────────────────
    version_file = os.environ.get("MODEL_VERSION_FILE", "model_version.txt")
    with open(version_file, "w") as f:
        f.write(registered.version)
    print(f"Model version đã lưu vào: {version_file}")

    # Output cho Azure DevOps
    print(f"##vso[task.setvariable variable=MODEL_VERSION;isOutput=true]{registered.version}")


if __name__ == "__main__":
    main()

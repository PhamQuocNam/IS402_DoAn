"""
ci_compare_model.py — So sánh metrics model mới vs model hiện tại
Dùng MAE và RMSE để quyết định có nên deploy model mới hay không.

Output:
  - In ##vso[task.setvariable] SHOULD_DEPLOY = true/false cho Azure DevOps
  - In ##vso[task.setvariable] NEW_MODEL_VERSION cho stage deploy
"""

import os
import sys
import json

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


def get_new_metrics(download_path: str) -> dict:
    """Đọc metrics từ file metrics.json đã download từ job outputs."""
    # Azure ML download structure: download_path/named-outputs/default/outputs/metrics.json
    candidates = [
        os.path.join(download_path, "named-outputs", "default", "outputs", "metrics.json"),
        os.path.join(download_path, "outputs", "metrics.json"),
        os.path.join(download_path, "metrics.json"),
    ]

    # Tìm thêm đệ quy
    for root, _, files in os.walk(download_path):
        for fname in files:
            if fname == "metrics.json":
                candidates.append(os.path.join(root, fname))

    for path in candidates:
        if os.path.exists(path):
            print(f"Đọc metrics từ: {path}")
            with open(path) as f:
                return json.load(f)

    raise FileNotFoundError(
        f"Không tìm thấy metrics.json trong {download_path}. "
        f"Đã kiểm tra: {candidates}"
    )


def get_current_model_metrics(ml_client: MLClient, model_name: str) -> dict | None:
    """Lấy metrics từ tags của model version mới nhất đã registered."""
    try:
        # Lấy model version mới nhất
        model_versions = list(ml_client.models.list(name=model_name))
        if not model_versions:
            print("Chưa có model nào được đăng ký. Sẽ deploy model đầu tiên.")
            return None

        # Sort theo version number (cao nhất = mới nhất)
        latest = max(model_versions, key=lambda m: int(m.version))
        print(f"Model hiện tại: {model_name} v{latest.version}")

        tags = latest.tags or {}
        if "mae" in tags and "rmse" in tags:
            metrics = {
                "mae": float(tags["mae"]),
                "rmse": float(tags["rmse"]),
            }
            print(f"   Metrics: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")
            return metrics
        else:
            print("Model hiện tại không có metrics trong tags. Sẽ deploy model mới.")
            return None

    except Exception as e:
        print(f"Không thể lấy model hiện tại: {e}")
        return None


def main():
    # ── Config ───────────────────────────────────────────────────────────────
    subscription_id = os.environ.get(
        "AZURE_SUBSCRIPTION_ID", "35d715f0-0211-4894-9c18-aea6e5787b86"
    )
    resource_group = os.environ.get("AZURE_RESOURCE_GROUP", "is402_doan")
    workspace_name = os.environ.get("AZURE_ML_WORKSPACE", "machinelearningproject")
    model_name = os.environ.get("MODEL_NAME", "fozzy-lightgbm-sales-model")
    download_path = os.environ.get("JOB_DOWNLOAD_PATH", "./job_downloads")

    # ── Kết nối Azure ML ─────────────────────────────────────────────────────
    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

    # ── Lấy metrics model mới ────────────────────────────────────────────────
    print("=" * 60)
    print("SO SÁNH MODEL METRICS")
    print("=" * 60)

    new_metrics = get_new_metrics(download_path)
    new_mae = new_metrics["mae"]
    new_rmse = new_metrics["rmse"]
    print(f"\nModel mới  — MAE: {new_mae:.4f}, RMSE: {new_rmse:.4f}")

    # ── Lấy metrics model hiện tại ───────────────────────────────────────────
    current_metrics = get_current_model_metrics(ml_client, model_name)

    # ── So sánh ──────────────────────────────────────────────────────────────
    if current_metrics is None:
        # Chưa có model → deploy luôn
        should_deploy = True
        print("\nChưa có model nào → Deploy model đầu tiên.")
    else:
        old_mae = current_metrics["mae"]
        old_rmse = current_metrics["rmse"]
        print(f"Model cũ   — MAE: {old_mae:.4f}, RMSE: {old_rmse:.4f}")

        mae_better = new_mae < old_mae
        rmse_better = new_rmse < old_rmse

        print(f"\n   MAE : {'tốt hơn' if mae_better else 'không tốt hơn'} "
              f"({new_mae:.4f} vs {old_mae:.4f})")
        print(f"   RMSE: {'tốt hơn' if rmse_better else 'không tốt hơn'} "
              f"({new_rmse:.4f} vs {old_rmse:.4f})")

        # Deploy nếu ít nhất MỘT metric tốt hơn
        should_deploy = mae_better or rmse_better
        if should_deploy:
            print("\nModel mới TỐT HƠN → Sẽ deploy!")
        else:
            print("\nModel mới KHÔNG tốt hơn → Bỏ qua deploy.")

    # ── Output cho Azure DevOps ──────────────────────────────────────────────
    print(f"\n##vso[task.setvariable variable=SHOULD_DEPLOY;isOutput=true]{str(should_deploy).lower()}")

    # Lưu new metrics để register script đọc
    with open("compare_result.json", "w") as f:
        json.dump({
            "should_deploy": should_deploy,
            "new_mae": new_mae,
            "new_rmse": new_rmse,
        }, f, indent=2)

    if not should_deploy:
        print("Bỏ qua các bước deploy tiếp theo.")

    return 0  # Luôn return 0, dùng variable để control flow


if __name__ == "__main__":
    sys.exit(main())

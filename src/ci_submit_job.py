"""
ci_submit_job.py — CI/CD version of submit_job.py
Dùng DefaultAzureCredential (Service Principal) thay vì InteractiveBrowserCredential.
Đọc config từ environment variables, tự động chờ job hoàn thành.

Chạy trong Azure DevOps Pipeline:
    python src/ci_submit_job.py
"""

import os
import sys

from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential


def main():
    # ── Config từ environment variables (hoặc mặc định) ─────────────────────
    subscription_id = os.environ.get(
        "AZURE_SUBSCRIPTION_ID", "35d715f0-0211-4894-9c18-aea6e5787b86"
    )
    resource_group = os.environ.get("AZURE_RESOURCE_GROUP", "is402_doan")
    workspace_name = os.environ.get("AZURE_ML_WORKSPACE", "machinelearningproject")
    compute_name = os.environ.get("COMPUTE_NAME", "cpu-cluster")

    # ── Kết nối Azure ML ─────────────────────────────────────────────────────
    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
    print(f"Đã kết nối workspace: {workspace_name}")

    # ── Tạo environment ──────────────────────────────────────────────────────
    my_env = Environment(
        name="frozzy-lightgbm-env",
        description="LightGBM + MLflow environment for CI/CD",
        conda_file="src/conda_env.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    )

    # ── Tạo job ──────────────────────────────────────────────────────────────
    job = command(
        code="src",
        command=(
            "python train.py "
            "--data_dir ${{inputs.training_data}} "
            "--model_dir ./outputs"
        ),
        inputs={
            "training_data": Input(
                type="uri_folder",
                path="azureml://datastores/workspace_storageaccount/paths/",
                mode="download",
            )
        },
        environment=my_env,
        compute=compute_name,
        display_name="frozzy-lightgbm-training-ci",
        experiment_name="retail-sales-forecast-ci",
    )

    # ── Submit & chờ hoàn thành ──────────────────────────────────────────────
    print("Đang gửi job lên Azure ML...")
    returned_job = ml_client.jobs.create_or_update(job)
    print(f"Job name : {returned_job.name}")
    print(f"Studio URL: {returned_job.studio_url}")

    print("Đang chờ job hoàn thành...")
    ml_client.jobs.stream(returned_job.name)

    # ── Kiểm tra trạng thái ──────────────────────────────────────────────────
    final_job = ml_client.jobs.get(returned_job.name)
    print(f"Job status: {final_job.status}")

    if final_job.status != "Completed":
        print(f"Job thất bại với status: {final_job.status}")
        sys.exit(1)

    # ── Lưu job name cho stage tiếp theo ─────────────────────────────────────
    output_file = os.environ.get("JOB_OUTPUT_FILE", "job_output.txt")
    with open(output_file, "w") as f:
        f.write(returned_job.name)
    print(f"Job name đã lưu vào: {output_file}")

    # ── Download metrics từ job outputs ──────────────────────────────────────
    print("Đang download metrics từ job outputs...")
    ml_client.jobs.download(
        returned_job.name,
        output_name="default",
        download_path="./job_downloads",
    )
    print("Download hoàn tất!")


if __name__ == "__main__":
    main()

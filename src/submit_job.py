from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment
from azure.identity import InteractiveBrowserCredential

credential = InteractiveBrowserCredential()

ml_client = MLClient(
    credential=credential,
    subscription_id="35d715f0-0211-4894-9c18-aea6e5787b86",
    resource_group_name="is402_doan",
    workspace_name="machinelearningproject",
)

compute_name = "cpu-cluster"

my_env = Environment(
    name="frozzy-lightgbm-env",
    description="LightGBM + MLflow environment",
    conda_file="conda_env.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
)

job = command(
    code=".",
    command="python train.py --data_dir ${{inputs.training_data}} --model_dir ./outputs",
    inputs={
        "training_data": Input(
            type="uri_folder",
            path="azureml://datastores/workspace_storageaccount/paths/",
            mode="download"  # download toàn bộ folder về compute trước khi chạy
        )
    },
    environment=my_env,
    compute=compute_name,
    display_name="frozzy-lightgbm-training",
    experiment_name="retail-sales-forecast"
)

print("Đang gửi job lên Azure ML...")
returned_job = ml_client.jobs.create_or_update(job)

print(f"✅ Đã gửi job. Tên job: {returned_job.name}")
print(f"🔗 Link xem tiến độ: {returned_job.studio_url}")

ml_client.jobs.stream(returned_job.name)
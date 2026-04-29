from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

# Xác thực tài khoản Azure
credential = DefaultAzureCredential()

# Kết nối với Workspace
ml_client = MLClient(
    credential=credential,
    subscription_id="35d715f0-0211-4894-9c18-aea6e5787b86",
    resource_group_name="is402_doan",
    workspace_name="machinelearningproject",
)

# Nhập tên job từ người dùng hoặc hardcode
# Cần thay thế 'JOB_NAME' bằng tên job thực tế từ submit_job.py
job_name = input("Nhập tên job muốn đăng ký model: ").strip()

# Trỏ đường dẫn tới file model.pkl trong thư mục outputs của Job vừa chạy
model_path = f"azureml://jobs/{job_name}/outputs/artifacts/paths/outputs/model.pkl"

run_model = Model(
    path=model_path,
    name="my-sklearn-model",
    description="Mô hình dự đoán version đầu tiên",
    type=AssetTypes.CUSTOM_MODEL
)

# Đăng ký vào Model Registry
ml_client.models.create_or_update(run_model)
print("Đã đăng ký mô hình thành công vào Azure ML Workspace!")
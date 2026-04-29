from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Xác thực tài khoản Azure của bạn
credential = DefaultAzureCredential()

# Kết nối với Workspace
ml_client = MLClient(
    credential=credential,
    subscription_id="35d715f0-0211-4894-9c18-aea6e5787b86",
    resource_group_name="is402_doan",
    workspace_name="machinelearningproject",
)
print("Đã kết nối Workspace thành công!")
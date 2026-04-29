from azure.ai.ml import MLClient
# from azure.identity import DefaultAzureCredential
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml.entities import AmlCompute
from azure.core.exceptions import ResourceNotFoundError

# Xác thực tài khoản Azure
credential = InteractiveBrowserCredential()

# Kết nối với Workspace
ml_client = MLClient(
    credential=credential,
    subscription_id="35d715f0-0211-4894-9c18-aea6e5787b86",
    resource_group_name="is402_doan",
    workspace_name="machinelearningproject",
)

compute_name = "cpu-cluster"

try:
    # Kiểm tra xem cụm máy ảo đã tồn tại chưa
    compute_target = ml_client.compute.get(compute_name)
    print(f"Đã tìm thấy compute cluster: {compute_name}")
except ResourceNotFoundError:
    print("Đang tạo compute cluster mới...")
    compute_target = AmlCompute(
        name=compute_name,
        type="amlcompute",
        size="STANDARD_DS3_V2", # Cấu hình máy ảo (RAM, CPU)
        min_instances=0,        # Scale về 0 khi rảnh rỗi
        max_instances=2,        # Scale tối đa 2 node khi cần
        idle_time_before_scale_down=120 # Tắt sau 120 giây không dùng
    )
    ml_client.compute.begin_create_or_update(compute_target).result()
    print("Tạo compute thành công!")
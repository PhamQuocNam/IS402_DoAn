# Hướng dẫn chi tiết thiết lập dự án IS402_DoAn với Azure (Từ A đến Z)

Tài liệu này hướng dẫn chi tiết từng bước để đưa dự án **Fozzy Group Retail Sales Forecasting** từ local lên môi trường đám mây Azure, bao gồm cấu hình Azure Machine Learning (Azure ML) và tự động hóa CI/CD với Azure DevOps Pipelines.

---

## Phần 1: Chuẩn bị Môi trường

### 1.1. Công cụ cần thiết
1. **Tài khoản Azure**: Đã có subscription (Ví dụ: `35d715f0-0211-4894-9c18-aea6e5787b86`).
2. **Tài khoản Azure DevOps**: Để chạy CI/CD Pipelines.
3. **Môi trường Local**:
   - Python 3.10
   - [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)

### 1.2. Cài đặt môi trường Local
Mở Terminal tại thư mục `IS402_DoAn`:

```bash
# Đăng nhập vào Azure
az login

# Đặt subscription mặc định (Thay ID bằng subscription của bạn)
az account set --subscription "35d715f0-0211-4894-9c18-aea6e5787b86"

# Cài đặt môi trường Python với Conda
conda env create -f src/conda_env.yml
conda activate lightgbm-env
```

---

## Phần 2: Thiết lập Azure Machine Learning

### 2.1. Tạo Azure ML Workspace
Bạn có thể tạo Workspace thông qua Azure Portal hoặc Azure CLI:

```bash
# Tạo Resource Group
az group create --name "is402_doan" --location "southeastasia"

# Cài đặt extension cho Azure ML
az extension add -n ml -y

# Tạo Azure ML Workspace
az ml workspace create --name "machinelearningproject" --resource-group "is402_doan"
```

### 2.2. Khởi tạo Compute Cluster (Máy chủ huấn luyện)
Chạy file script có sẵn để tạo Compute Cluster (`cpu-cluster`) trên Azure ML:

```bash
cd src
python init_instance.py
```
*Script này sẽ tự động kiểm tra và tạo một cụm máy ảo (STANDARD_DS3_V2, có thể tự động tắt khi không dùng) để dùng cho việc huấn luyện mô hình.*

### 2.3. Đưa dữ liệu lên Azure Blob Storage
Để huấn luyện trên cloud, Azure ML cần đọc dữ liệu từ Cloud (Datastore).

1. Truy cập [Azure ML Studio](https://ml.azure.com/).
2. Chọn Workspace `machinelearningproject`.
3. Vào tab **Data** -> **Datastores**.
4. Bấm vào `workspaceblobstore` (Datastore mặc định).
5. Bấm **Browse** và upload thư mục chứa 2 file dữ liệu (`train_final.csv` và `sku_final.csv`) lên một thư mục, ví dụ đặt tên là `paths/`.
*(Lưu ý: Trong `ci_submit_job.py` và `submit_job.py`, đường dẫn data đang được cấu hình trỏ tới `azureml://datastores/workspace_storageaccount/paths/`)*

---

## Phần 3: Huấn luyện mô hình (Training)

### 3.1. Chạy thử trên máy cá nhân (Local Test)
Trước khi đưa lên cloud, hãy đảm bảo code chạy tốt trên máy:

```bash
# Đảm bảo dữ liệu nằm trong thư mục ../data
python train.py --data_dir ../data --model_dir ./outputs
```
*Kết quả:* Mô hình sẽ được tạo ra tại `./outputs/lgb_sales_model.pkl` và xuất ra các chỉ số MAE, RMSE.

### 3.2. Huấn luyện trên Azure ML
Gửi Job huấn luyện lên Azure:

```bash
python submit_job.py
```
*Hành động:* Azure ML sẽ cấp phát máy ảo, tải dữ liệu, tiến hành huấn luyện, ghi nhận các tham số/metric (bằng MLflow) và lưu mô hình vào Artifacts của Job đó.

---

## Phần 4: Thiết lập CI/CD với Azure DevOps

Dự án sử dụng Azure Pipelines để tự động test và deploy mỗi khi có thay đổi code mới.

### 4.1. Đưa Code lên Azure Repos hoặc GitHub
1. Vào [Azure DevOps](https://dev.azure.com/), tạo một Project (Ví dụ: `IS402_DoAn`).
2. Push toàn bộ mã nguồn hiện tại của bạn lên kho lưu trữ (Repos) của Project đó.

### 4.2. Tạo Service Connection (Ủy quyền cho Azure DevOps)
Để Azure Pipelines có quyền thao tác trên Azure ML:
1. Trong Azure DevOps, vào **Project Settings** (Góc trái dưới).
2. Chọn **Service connections** -> **New service connection**.
3. Chọn **Azure Resource Manager** -> **Service Principal (automatic)**.
4. Chọn Subscription của bạn và Resource Group là `is402_doan`.
5. Đặt tên Service connection là: `AzureMLServiceConnection`.
6. Tích chọn **Grant access permission to all pipelines** -> Save.

### 4.3. Tạo Azure Kubernetes Service (Tùy chọn - Nếu muốn deploy Production)
Dự án được cấu hình deploy Staging lên ACI và Production lên AKS. Nếu muốn deploy lên AKS, bạn cần gắn AKS vào Azure ML:
1. Vào **Azure ML Studio** -> **Compute** -> **Attached computes** -> New -> Kubernetes.
2. Tạo hoặc liên kết một AKS Cluster có sẵn.
3. Đặt tên compute name là `aks-cluster` (Trùng với biến trong cấu hình CI/CD).

### 4.4. Kích hoạt Pipeline
1. Trong Azure DevOps, vào **Pipelines** -> **New Pipeline**.
2. Chọn nguồn code (Azure Repos Git hoặc GitHub).
3. Chọn **Existing Azure Pipelines YAML file**.
4. Chọn file `azure-pipelines.yml` nằm trong dự án của bạn.
5. Xem lại các biến trong file YAML để đảm bảo chính xác:
   - `subscriptionId`
   - `resourceGroup`
   - `workspaceName`
6. Bấm **Run**.

---

## Phần 5: Quy trình hoạt động của CI/CD (Workflow)

Mỗi khi bạn sửa code và gõ lệnh `git push origin main`:

1. **Stage 1: CI (Continuous Integration)**
   - Hệ thống tạo máy ảo Ubuntu trắng.
   - Cài Python, chạy `flake8` để kiểm tra chuẩn format code (chỉ cảnh báo).
   - Chạy `pytest tests/` để test các hàm logic. (Nếu lỗi, Pipeline sẽ báo đỏ và dừng lại).

2. **Stage 2: Train (Huấn luyện)**
   - Gọi file `src/ci_submit_job.py`.
   - Pipeline ra lệnh cho Azure ML chạy huấn luyện lại mô hình với code mới.
   - Chờ job chạy xong và tải các chỉ số (Metrics: MAE, RMSE) về Pipeline.

3. **Stage 3: CD (Continuous Deployment)**
   - So sánh: Chạy file `src/ci_compare_model.py` để so sánh MAE/RMSE của model vừa huấn luyện với model tốt nhất đang chạy.
   - Đăng ký: Nếu model mới tốt hơn, gọi `src/ci_register_model.py` để đăng ký model mới vào Azure ML Model Registry.
   - Triển khai Test: Gọi `src/ci_deploy_model.py --target aci` để deploy lên Managed Online Endpoint (môi trường staging).
   - Triển khai Thực tế: Gọi `src/ci_deploy_model.py --target aks` để deploy lên Kubernetes (môi trường production).

🎉 **Chúc mừng! Bạn đã sở hữu một hệ thống Machine Learning Operations (MLOps) hoàn chỉnh chuẩn doanh nghiệp!**

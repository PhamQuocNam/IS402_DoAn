# Fozzy Group Retail Sales Forecasting

Dự án dự đoán doanh số bán lẻ sử dụng LightGBM với MLflow tracking và Azure Machine Learning.

## 📋 Mô tả dự án

- **Mục tiêu**: Dự đoán doanh số bán hàng cho Fozzy Group
- **Mô hình**: LightGBM Regressor
- **Features**: Geo cluster, SKU, product category, temporal features (day/month cyclical)
- **Platform**: Có thể chạy local hoặc trên Azure ML

## 🛠️ Cài đặt môi trường

### Sử dụng Conda (Khuyến nghị)

> Ghi chú: môi trường hiện dùng Python 3.10 để tương thích tốt với Azure ML inference HTTP server khi test local.

```bash
# Tạo môi trường conda từ file cấu hình
conda env create -f src/conda_env.yml

# Kích hoạt môi trường
conda activate lightgbm-env
```

### Hoặc cài đặt thủ công

```bash
pip install lightgbm scikit-learn pandas numpy mlflow azureml-mlflow azure-ai-ml azure-identity azureml-inference-server-http joblib
```

## 📁 Cấu trúc dự án

```
IS402_DoAn/
├── data/                   # Thư mục dữ liệu
│   ├── sku_final.csv       # Thông tin SKU
│   └── train_final.csv     # Dữ liệu training
├── src/                    # Source code
│   ├── train.py            # Script training chính
│   ├── score.py            # Script inference/scoring cho Azure ML endpoint
│   ├── connect.py          # Kết nối Azure ML
│   ├── submit_job.py       # Gửi job lên Azure ML
│   ├── init_instance.py    # Khởi tạo compute cluster
│   ├── model_registry.py   # Đăng ký model
│   └── conda_env.yml       # Cấu hình môi trường
├── sample-request.json     # Payload mẫu để test scoring script/endpoint
├── data.zip                # Dữ liệu nén
└── README.md               # File này
```

## 🚀 Cách chạy

### Phương pháp 1: Chạy Local

#### Bước 1: Chuẩn bị dữ liệu

```bash
# Giải nén dữ liệu (nếu chưa giải nén)
unzip data.zip

# Hoặc để trong thư mục data/
```

#### Bước 2: Chạy training

```bash
cd src
python train.py --data_dir ../data --model_dir ./outputs
```

**Tham số tùy chọn:**

- `--data_dir`: Thư mục chứa dữ liệu (mặc định: `./raw-data`)
- `--model_dir`: Thư mục lưu model (mặc định: `./outputs`)
- `--n_estimators`: Số lượng cây (mặc định: 1500)
- `--learning_rate`: Tốc độ học (mặc định: 0.05)
- `--max_depth`: Độ sâu tối đa (mặc định: 8)
- `--num_leaves`: Số lượng lá (mặc định: 64)
- `--random_state`: Seed ngẫu nhiên (mặc định: 42)

**Ví dụ với tham số tùy chỉnh:**

```bash
python train.py --data_dir ../data --model_dir ./outputs --n_estimators 2000 --learning_rate 0.03
```

#### Bước 3: Kết quả

- Model được lưu tại: `src/outputs/lgb_sales_model.pkl`
- Metrics (MAE, RMSE) được hiển thị trong terminal
- MLflow logs được lưu trong thư mục `mlruns/`

### Phương pháp 2: Chạy trên Azure ML

#### Bước 1: Cài đặt Azure CLI

```bash
# Cài đặt Azure CLI (nếu chưa cài)
# Windows: Chạy installer từ https://docs.microsoft.com/cli/azure/install-azure-cli

# Đăng nhập Azure
az login
```

#### Bước 2: Cấu hình Azure ML Workspace

Chỉnh sửa các file trong thư mục `src/` để cập nhật thông tin Azure của bạn:

- `subscription_id`: ID subscription Azure
- `resource_group_name`: Tên resource group
- `workspace_name`: Tên Azure ML Workspace

#### Bước 3: Khởi tạo Compute Cluster (Chỉ cần làm 1 lần)

```bash
cd src
python init_instance.py
```

Script này sẽ:

- Kiểm tra xem compute cluster đã tồn tại chưa
- Tạo mới nếu chưa có với cấu hình: STANDARD_DS3_V2, 0-2 nodes

#### Bước 4: Kết nối và kiểm tra Workspace

```bash
python connect.py
```

#### Bước 5: Gửi job training lên Azure ML

```bash
python submit_job.py
```

Script này sẽ:

- Tạo môi trường training từ `conda_env.yml`
- Gửi job training lên Azure ML compute cluster
- Stream log training về terminal
- Cung cấp link để xem tiến độ trên Azure ML Studio

#### Bước 6: Theo dõi job

Sau khi gửi job, bạn sẽ nhận được:

- Tên job (ví dụ: `frozzy-lightgbm-training_1234567890`)
- Link Azure ML Studio để theo dõi

#### Bước 7: Đăng ký model vào Model Registry

```bash
python model_registry.py
```

Khi được hỏi, nhập tên job từ bước 5.

#### Bước 8: Chuẩn bị inference package

Sau khi model được đăng ký trong Azure ML Model Registry, nhóm chuẩn bị các thành phần phục vụ triển khai inference:

- `src/score.py`: scoring script chứa hai hàm `init()` và `run()`.
- `src/conda_env.yml`: môi trường Python có LightGBM, pandas, numpy, scikit-learn, joblib và `azureml-inference-server-http`.
- Registered model: `fozzy-lightgbm-sales-model:1`.

`init()` load model từ biến môi trường `AZUREML_MODEL_DIR`; `run()` nhận JSON input, chuyển thành DataFrame, gọi `model.predict()` và trả về kết quả dự đoán.

Khi deploy lên Azure ML Online Endpoint, model artifact được mount vào biến môi trường `AZUREML_MODEL_DIR`. Trong lần deploy thử, Azure ML mount model theo cấu trúc:

```text
AZUREML_MODEL_DIR/model/lgb_sales_model.pkl
```

Vì vậy `src/score.py` được cấu hình để tìm model linh hoạt ở các vị trí:

```text
AZUREML_MODEL_DIR/lgb_sales_model.pkl
AZUREML_MODEL_DIR/model/lgb_sales_model.pkl
AZUREML_MODEL_DIR/outputs/lgb_sales_model.pkl
```

Khi test local, nhóm tạo thư mục `azure-test-model/model/` để giả lập đúng cấu trúc mount model trên Azure.

> Mục tiêu của bước này là kiểm tra `score.py` có thể load model, nhận JSON input và trả kết quả dự đoán trước khi deploy lên Azure ML Online Endpoint.

Kích hoạt môi trường:

```bash
conda activate lightgbm-env
```

Nếu môi trường chưa được cập nhật sau khi sửa `conda_env.yml`, chạy:

```bash
conda env update -f src/conda_env.yml --prune
conda activate lightgbm-env
```

Tạo cấu trúc thư mục giả lập giống cách Azure ML mount model khi deploy:

```powershell
New-Item -ItemType Directory -Force .\azure-test-model\model
Copy-Item .\src\outputs\lgb_sales_model.pkl .\azure-test-model\model\lgb_sales_model.pkl -Force
```

Kiểm tra model đã được copy:

```powershell
dir .\azure-test-model\model
```

Kết quả kỳ vọng:

```text
lgb_sales_model.pkl
```

> Lưu ý: `azure-test-model/`, `src/outputs/`, và các file `.pkl` chỉ dùng để test local, không push lên GitHub.

Tạo file `sample-request.json` ở root repo:

```json
{
  "geoCluster": 1,
  "SKU": 12345,
  "productCategoryId": 10,
  "lagerUnitTypeId": 1,
  "day_sin": 0.101168,
  "day_cos": -0.994869,
  "month_sin": 0.0,
  "month_cos": -1.0,
  "year": 2021,
  "commodity_group": 5550259,
  "trademark": 5
}
```

`Cách 1`: Chạy trực tiếp `score.py`

Đứng tại root repo `IS402_DoAn`, chạy:

```powershell
python .\src\score.py
```

Kết quả kỳ vọng:

```json
{"predictions": [...], "n_records": 1}
```

Cách này kiểm tra nhanh các phần sau:

- `score.py` chạy được.
- Hàm `init()` load được model từ thư mục local hoặc từ cấu trúc giả lập Azure `azure-test-model/model/lgb_sales_model.pkl`.
- Hàm `run()` nhận JSON sample và gọi `model.predict()` thành công.

`Cách 2`: Test bằng Azure ML inference HTTP server

Chạy inference server ở terminal thứ nhất, vẫn đứng tại root repo `IS402_DoAn`:

```powershell
azmlinfsrv --entry_script .\src\score.py --model_dir .\azure-test-model

# Nếu server chạy đúng, terminal sẽ hiển thị route:
# Score: POST 127.0.0.1:5001/score
```

Mở terminal thứ hai, gửi request mẫu.

Với PowerShell:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:5001/score" `
  -ContentType "application/json" `
  -Body (Get-Content .\sample-request.json -Raw)
```

Hoặc dùng curl:

```bash
curl -X POST http://127.0.0.1:5001/score \
  -H "Content-Type: application/json" \
  --data-binary @sample-request.json
```

Kết quả test local kỳ vọng sẽ trả về JSON chứa predictions:

```json
{"predictions": [-0.32646408393872034], "n_records": 1}
```

Trong terminal inference server, nếu thấy các dòng sau thì scoring script đã load model thành công:

```text
Invoking user's init function
Users's init has completed successfully
Worker with pid ... ready for serving traffic
POST /score 200
```

Ở bước test local, nhóm chưa gọi API prediction trên Azure. Việc cần kiểm tra trên Azure ML Studio là:

- Model đã được đăng ký thành công trong Model Registry chưa.
- `Azure ML Studio → Models → fozzy-lightgbm-sales-model → Version 1`

Sau khi local test nhận được `predictions`, inference package đã sẵn sàng để deploy lại trên Azure ML Online Endpoint ở Bước 5.

Khi tạo deployment mới trên Azure, cần chọn lại `src/score.py` bản đã sửa ở mục **Code + environment**.

## 📊 Chi tiết Model

### Features sử dụng

- **geoCluster**: Cluster địa lý
- **SKU**: Mã sản phẩm
- **productCategoryId**: ID category sản phẩm
- **lagerUnitTypeId**: ID đơn vị lưu trữ
- **day_sin, day_cos**: Biến cyclic cho ngày trong tháng
- **month_sin, month_cos**: Biến cyclic cho tháng
- **year**: Năm
- **commodity_group**: Nhóm hàng hóa
- **trademark**: Thương hiệu

### Preprocessing

1. **Null handling**:
   - `trademark`: Điền theo mode của (productCategoryId × commodity_group)
   - `sales`: Điền 0 (không bán hàng)

2. **Feature engineering**:
   - Tạo temporal features (day, month, year)
   - Chuyển đổi sang cyclic encoding (sin/cos)
   - Downcast dữ liệu để tiết kiệm RAM

3. **Data split**:
   - Train: 90%
   - Validation: 5%
   - Test: 5%
   - Split theo chronological order

### Hyperparameters mặc định

```python
{
    "n_estimators": 1500,
    "learning_rate": 0.05,
    "max_depth": 8,
    "num_leaves": 64,
    "random_state": 42,
    "n_jobs": -1
}
```

### Metrics

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)

## 🔧 Xử lý sự cố

### Lỗi thiếu dữ liệu

```bash
# Kiểm tra xem dữ liệu đã được giải nén chưa
ls -la data/

# Nếu không có, giải nén data.zip
unzip data.zip -d data/
```

### Lỗi cài đặt package

```bash
# Xóa và tạo lại môi trường conda
conda deactivate
conda env remove -n lightgbm-env
conda env create -f src/conda_env.yml
```

### Lỗi Azure authentication

```bash
# Đăng nhập lại Azure
az login
az account set --subscription <YOUR_SUBSCRIPTION_ID>
```

### Lỗi compute cluster

```bash
# Kiểm tra trạng thái compute cluster
python -c "
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id='YOUR_SUBSCRIPTION_ID',
    resource_group_name='is402_doan',
    workspace_name='machinelearningproject'
)

compute = ml_client.compute.get('cpu-cluster')
print(f'Status: {compute.provisioning_state}')
print(f'Size: {compute.size}')
print(f'Nodes: {compute.scale_settings.min_instances}-{compute.scale_settings.max_instances}')
"
```

## 📝 Ghi chú

- Model được tracking bằng MLflow - tất cả params, metrics, và artifacts đều được log
- Khi chạy trên Azure ML, MLflow tracking được tích hợp sẵn
- Compute cluster tự scale về 0 khi không sử dụng để tiết kiệm chi phí
- Model được lưu dưới dạng `.pkl` có thể load lại bằng `joblib.load()`

## 🔄 Bước 6: CI/CD với Azure Pipelines

### Tổng quan Pipeline

Pipeline tự động chạy khi push code lên nhánh `main`, gồm 3 stages:

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────────┐
│   CI Stage   │────▶│  Train Stage  │────▶│      CD Stage         │
│ Lint + Test  │     │ Azure ML Job │     │ Compare → Register   │
│              │     │              │     │ → Deploy ACI + AKS   │
└─────────────┘     └──────────────┘     └──────────────────────┘
```

- **CI**: Chạy `flake8` lint + `pytest` unit tests
- **Train**: Gửi job huấn luyện LightGBM lên Azure ML compute cluster
- **CD**: So sánh MAE/MSE model mới vs cũ → đăng ký → deploy lên ACI (staging) & AKS (production)

### Cấu trúc file CI/CD

```
IS402_DoAn/
├── azure-pipelines.yml           # Pipeline definition (3 stages)
├── requirements-dev.txt          # pytest, flake8
├── tests/
│   ├── __init__.py
│   └── test_train.py             # Unit tests cho train.py
└── src/
    ├── ci_submit_job.py          # Gửi job (dùng Service Principal)
    ├── ci_compare_model.py       # So sánh MAE/MSE model mới vs cũ
    ├── ci_register_model.py      # Đăng ký model với metrics tags
    └── ci_deploy_model.py        # Deploy lên ACI (staging) + AKS (production)
```

### Thiết lập Azure DevOps

#### Bước 1: Tạo Service Connection

1. Azure DevOps → Project Settings → **Service connections**
2. **New service connection** → **Azure Resource Manager** → **Service Principal (automatic)**
3. Chọn Subscription: `35d715f0-0211-4894-9c18-aea6e5787b86`
4. Chọn Resource Group: `is402_doan`
5. Đặt tên: `AzureMLServiceConnection`
6. Tick ✅ **Grant access permission to all pipelines**

#### Bước 2: Tạo Variable Group (Tuỳ chọn)

Pipelines → Library → **New Variable Group**, thêm các biến:

| Variable | Value |
|----------|-------|
| `AZURE_SUBSCRIPTION_ID` | `35d715f0-0211-4894-9c18-aea6e5787b86` |
| `AZURE_RESOURCE_GROUP` | `is402_doan` |
| `AZURE_ML_WORKSPACE` | `machinelearningproject` |
| `COMPUTE_NAME` | `cpu-cluster` |
| `MODEL_NAME` | `fozzy-lightgbm-sales-model` |

#### Bước 3: Tạo Pipeline

1. Push code (bao gồm `azure-pipelines.yml`) lên Azure DevOps Repos
2. Pipelines → **New Pipeline** → Azure Repos Git
3. Chọn repo → **Existing Azure Pipelines YAML file** → chọn `azure-pipelines.yml`
4. **Run** pipeline

### Chạy Unit Test Local

```bash
# Cài dependencies test
pip install -r requirements-dev.txt

# Chạy tests
pytest tests/ -v
```

### Flow hoạt động chi tiết

1. Developer push code lên `main`
2. **CI Stage**: Pipeline cài Python 3.10, chạy lint + pytest
3. **Train Stage**: Gửi job lên Azure ML cluster, chờ hoàn thành, download metrics
4. **CD Stage**:
   - So sánh MAE & MSE model mới với model đang registered
   - Nếu model mới tốt hơn → đăng ký model mới (kèm metrics tags)
   - Deploy lên **ACI** endpoint (staging) để test
   - Deploy lên **AKS** endpoint (production) để chạy thực tế

## 👥 Tác giả

Dự án được phát triển cho môn học IS402 - Machine Learning

## 📄 License

Dành cho mục đích học thuật

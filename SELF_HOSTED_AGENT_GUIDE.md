# Hướng dẫn Setup Self-hosted Agent trên Docker (VM) cho Azure DevOps

## Tại sao cần Self-hosted Agent?

Khi chạy CI/CD trên Azure DevOps, bạn có thể gặp lỗi:

```
##[error]No hosted parallelism has been purchased or granted.
To request a free parallelism grant, please fill out the following form
https://aka.ms/azpipelines-parallelism-request
```

**Nguyên nhân**: Tài khoản Azure DevOps mới không còn được cấp miễn phí Microsoft-hosted agent. Phải điền form chờ duyệt (2–5 ngày hoặc hơn).

**Giải pháp**: Tự chạy Agent trên Docker container trong VM của bạn — **miễn phí, không giới hạn, setup trong 15–30 phút**.

| Loại Agent | Mô tả | Chi phí |
|------------|-------|---------|
| **Microsoft-hosted** | VM do Microsoft quản lý (`vmImage: 'ubuntu-latest'`) | Cần mua hoặc chờ cấp free grant |
| **Self-hosted** | Agent bạn tự chạy trên VM/Docker | **Miễn phí** |

---

## Kiến trúc

```
Push code → Azure DevOps trigger
                ↓
        Agent Pool (Self-hosted: SelfHostedDocker)
                ↓
        VM Ubuntu (Docker Container chạy Azure Pipelines Agent)
                ↓
        Chạy CI/CD Jobs: Lint, Test, Train, Deploy
                ↓
        Azure ML Workspace
```

---

## Yêu cầu VM tối thiểu

| Thông số | Tối thiểu | Khuyến nghị |
|----------|-----------|-------------|
| **OS** | Ubuntu 20.04+ | Ubuntu 22.04 LTS |
| **CPU** | 2 vCPU | 4 vCPU |
| **RAM** | 4 GB | 8 GB |
| **Disk** | 30 GB | 50 GB SSD |

---

## Bước 1: Tạo Agent Pool trên Azure DevOps

1. Truy cập **Azure DevOps** → **Organization Settings** (góc trái dưới)
2. Vào **Pipelines** → **Agent pools**
3. Bấm **Add pool**
4. Cấu hình:
   - **Pool type**: `Self-hosted`
   - **Name**: `SelfHostedDocker`
   - ✅ Tích **Grant access permission to all pipelines**
5. Bấm **Create**

> 📌 Ghi nhớ tên pool `SelfHostedDocker` — sẽ dùng trong `azure-pipelines.yml`.

---

## Bước 2: Tạo Personal Access Token (PAT)

1. Bấm vào **avatar** (góc phải trên) → **Personal access tokens**
2. Bấm **New Token**
3. Cấu hình:
   - **Name**: `self-hosted-agent`
   - **Expiration**: 90 ngày (hoặc tùy chỉnh)
   - **Scopes**: Chọn **Custom defined**, tích:
     - ✅ **Agent Pools** → `Read & manage`
4. Bấm **Create**

> ⚠️ **Sao chép PAT ngay lập tức** — bạn sẽ không thể xem lại sau khi đóng dialog!

---

## Bước 3: Cài đặt Docker trên VM

SSH vào VM và chạy lần lượt:

```bash
# Cập nhật hệ thống
sudo apt-get update && sudo apt-get upgrade -y

# Cài dependencies
sudo apt-get install -y \
    ca-certificates curl gnupg lsb-release

# Thêm Docker GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Thêm Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Cài Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Cho phép user hiện tại dùng Docker (không cần sudo)
sudo usermod -aG docker $USER
newgrp docker

# Kiểm tra
docker --version
docker run hello-world
```

---

## Bước 4: Tạo Docker Image cho Agent

### 4.1. Tạo thư mục

```bash
mkdir -p ~/azagent && cd ~/azagent
```

### 4.2. Tạo `Dockerfile`

```dockerfile
# ═══════════════════════════════════════════════════════════════
# Dockerfile: Azure DevOps Self-hosted Agent
# Project: IS402_DoAn — Fozzy Group Retail Sales Forecasting
# ═══════════════════════════════════════════════════════════════

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── 1. System dependencies ───────────────────────────────────
RUN apt-get update && apt-get install -y \
    curl wget git jq unzip zip sudo \
    apt-transport-https ca-certificates gnupg lsb-release \
    software-properties-common build-essential libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Python 3.10 ───────────────────────────────────────────
RUN add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y \
        python3.10 python3.10-venv python3.10-dev python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# ── 3. Azure CLI ─────────────────────────────────────────────
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash
RUN az extension add --name ml -y

# ── 4. Python packages (dùng venv tránh conflict system packages) ──
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    flake8 pytest \
    lightgbm scikit-learn pandas numpy \
    mlflow joblib \
    azure-ai-ml azure-identity azureml-mlflow

# ── 5. Tạo user (không chạy root) ────────────────────────────
RUN useradd -m -s /bin/bash agentuser \
    && echo "agentuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER agentuser
WORKDIR /home/agentuser

# ── 6. Tải Azure Pipelines Agent ─────────────────────────────
# Domain: download.agent.dev.azure.com (KHÔNG dùng vstsagentpackage.azureedge.net)
# Xem version mới: https://github.com/microsoft/azure-pipelines-agent/releases
ARG AGENT_VERSION=4.272.0

RUN curl -fSL -o vsts-agent.tar.gz \
    "https://download.agent.dev.azure.com/agent/${AGENT_VERSION}/vsts-agent-linux-x64-${AGENT_VERSION}.tar.gz" \
    && mkdir agent && cd agent \
    && tar xzf ../vsts-agent.tar.gz \
    && rm ../vsts-agent.tar.gz

# ── 7. Script khởi động ──────────────────────────────────────
COPY --chown=agentuser:agentuser start.sh /home/agentuser/start.sh
RUN chmod +x /home/agentuser/start.sh

ENTRYPOINT ["/home/agentuser/start.sh"]
```

### 4.3. Tạo `start.sh`

```bash
#!/bin/bash
set -e

# Kiểm tra biến bắt buộc
if [ -z "$AZP_URL" ]; then
    echo "❌ ERROR: Thiếu AZP_URL (Ví dụ: https://dev.azure.com/YourOrg)"
    exit 1
fi

if [ -z "$AZP_TOKEN" ]; then
    echo "❌ ERROR: Thiếu AZP_TOKEN (Personal Access Token)"
    exit 1
fi

AZP_POOL=${AZP_POOL:-"SelfHostedDocker"}
AZP_AGENT_NAME=${AZP_AGENT_NAME:-"docker-agent-$(hostname)"}
AZP_WORK=${AZP_WORK:-"_work"}

echo "════════════════════════════════════════════════════"
echo "  Azure DevOps Self-hosted Agent"
echo "════════════════════════════════════════════════════"
echo "  URL:   $AZP_URL"
echo "  Pool:  $AZP_POOL"
echo "  Agent: $AZP_AGENT_NAME"
echo "════════════════════════════════════════════════════"

cd /home/agentuser/agent

# Gỡ đăng ký agent khi container dừng
cleanup() {
    echo "🧹 Đang gỡ đăng ký agent..."
    ./config.sh remove --unattended \
        --auth pat --token "$AZP_TOKEN" || true
}

trap 'cleanup; exit 0' EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

# Cấu hình agent
echo "⚙️  Đang cấu hình agent..."
./config.sh --unattended \
    --url "$AZP_URL" \
    --auth pat \
    --token "$AZP_TOKEN" \
    --pool "$AZP_POOL" \
    --agent "$AZP_AGENT_NAME" \
    --work "$AZP_WORK" \
    --replace \
    --acceptTeeEula

# Chạy agent
echo "🚀 Khởi chạy agent..."
exec ./run.sh & wait $!
```

### 4.4. Build Image

```bash
cd ~/azagent
docker build -t azdevops-agent:latest .
```

> Thời gian build: khoảng 5–10 phút.

---

## Bước 5: Chạy Agent Container

```bash
docker run -d \
    --name azagent-01 \
    --restart always \
    -e AZP_URL="https://dev.azure.com/YOUR_ORGANIZATION" \
    -e AZP_TOKEN="YOUR_PAT_TOKEN" \
    -e AZP_POOL="SelfHostedDocker" \
    -e AZP_AGENT_NAME="docker-agent-01" \
    azdevops-agent:latest
```

> ⚠️ Thay `YOUR_ORGANIZATION` và `YOUR_PAT_TOKEN` bằng giá trị thật.

Kiểm tra:

```bash
# Xem container đang chạy
docker ps

# Xem logs (chờ thấy "Listening for Jobs")
docker logs azagent-01 -f
```

---

## Bước 6: Xác nhận Agent Online

1. Vào **Azure DevOps** → **Organization Settings** → **Agent pools**
2. Chọn pool `SelfHostedDocker` → tab **Agents**
3. Xác nhận agent hiển thị **🟢 Online**

---

## Bước 7: Sửa `azure-pipelines.yml`

Thay đoạn `pool` ở đầu file:

```yaml
# ❌ CŨ — Microsoft-hosted agent (bị lỗi parallelism)
pool:
  vmImage: 'ubuntu-latest'

# ✅ MỚI — Self-hosted agent (miễn phí)
pool:
  name: 'SelfHostedDocker'
```

> ⚠️ **LƯU Ý QUAN TRỌNG**: Phải dùng `name:` chứ KHÔNG phải `vmImage:`.
> - `vmImage` = Microsoft-hosted agent → cần parallelism → **BỊ LỖI**
> - `name` = Self-hosted agent pool → miễn phí → **HOẠT ĐỘNG**

Phần còn lại của pipeline **giữ nguyên hoàn toàn**, không cần sửa gì thêm.

---

## Bước 8: Push & Chạy Pipeline

```bash
git add azure-pipelines.yml
git commit -m "chore: switch to self-hosted agent pool"
git push origin main
```

Pipeline sẽ tự trigger và chạy trên self-hosted agent! 🎉

---

## Quản lý & Bảo trì

### Xem logs agent

```bash
docker logs -f azagent-01
```

### Restart agent

```bash
docker restart azagent-01
```

### Cập nhật agent version

```bash
# Dừng & xóa container cũ
docker stop azagent-01 && docker rm azagent-01

# Sửa AGENT_VERSION trong Dockerfile, build lại
docker build -t azdevops-agent:latest .

# Chạy lại container
docker run -d --name azagent-01 --restart always \
    -e AZP_URL="..." -e AZP_TOKEN="..." \
    -e AZP_POOL="SelfHostedDocker" \
    -e AZP_AGENT_NAME="docker-agent-01" \
    azdevops-agent:latest
```

### Gia hạn PAT

PAT có thời hạn! Nhớ gia hạn trước khi hết:
1. Azure DevOps → **Personal access tokens** → Tìm token → **Edit** → **Extend expiration**
2. Restart container với token mới nếu cần tạo lại

---

## Xử lý sự cố

| Lỗi | Nguyên nhân | Cách fix |
|-----|-------------|----------|
| `Unable to find image locally` | Chưa build Docker image | Chạy `docker build -t azdevops-agent:latest .` |
| `Could not resolve host: vstsagentpackage.azureedge.net` | Domain cũ đã ngưng | Dùng domain `download.agent.dev.azure.com` |
| `Cannot uninstall blinker` | Conflict system packages | Dùng `python3.10 -m venv /opt/venv` trong Dockerfile |
| `Pool: Azure Pipelines, Image: SelfHostedDocker` | Dùng `vmImage` thay vì `name` | Sửa thành `pool: name: 'SelfHostedDocker'` |
| Agent không Online | Sai URL/PAT/Pool name | Kiểm tra logs: `docker logs azagent-01` |
| `No agent found` | Agent offline hoặc sai tên pool | Kiểm tra agent Online trên Azure DevOps |
| `UsePythonVersion` thất bại | Task chỉ dùng cho hosted agent | Xóa task hoặc giữ nguyên (warning only) |

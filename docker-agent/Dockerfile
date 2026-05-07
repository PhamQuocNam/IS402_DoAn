# ═══════════════════════════════════════════════════════════════
# Dockerfile: Azure DevOps Self-hosted Agent
# Project: IS402_DoAn — Fozzy Group Retail Sales Forecasting
# ═══════════════════════════════════════════════════════════════

FROM ubuntu:22.04

# Tránh prompt tương tác khi cài đặt packages
ENV DEBIAN_FRONTEND=noninteractive

# ── 1. Cài đặt system dependencies ───────────────────────────
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    jq \
    unzip \
    zip \
    sudo \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    software-properties-common \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Cài đặt Python 3.10 ───────────────────────────────────
RUN add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

# Đặt Python 3.10 làm mặc định
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Cài pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# ── 3. Cài đặt Azure CLI ─────────────────────────────────────
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

# Cài Azure ML extension
RUN az extension add --name ml -y

# ── 4. Cài đặt Python packages cho dự án ─────────────────────
RUN pip install --no-cache-dir --ignore-installed \
    flake8 \
    pytest \
    lightgbm \
    scikit-learn \
    pandas \
    numpy \
    mlflow \
    joblib \
    azure-ai-ml \
    azure-identity \
    azureml-mlflow

# ── 5. Tạo user cho agent (không chạy bằng root) ─────────────
RUN useradd -m -s /bin/bash agentuser \
    && echo "agentuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER agentuser
WORKDIR /home/agentuser

# ── 6. Tải và cài đặt Azure Pipelines Agent ──────────────────
# Kiểm tra version mới nhất tại:
# https://github.com/microsoft/azure-pipelines-agent/releases
ARG AGENT_VERSION=4.272.0
ARG TARGETARCH=x64
RUN curl -fSL -o vsts-agent.tar.gz \
    "https://download.agent.dev.azure.com/agent/${AGENT_VERSION}/vsts-agent-linux-${TARGETARCH}-${AGENT_VERSION}.tar.gz" \
    && mkdir agent && cd agent \
    && tar xzf ../vsts-agent.tar.gz \
    && rm ../vsts-agent.tar.gz

# ── 7. Script khởi động agent ────────────────────────────────
COPY --chown=agentuser:agentuser start.sh /home/agentuser/start.sh
RUN chmod +x /home/agentuser/start.sh

ENTRYPOINT ["/home/agentuser/start.sh"]
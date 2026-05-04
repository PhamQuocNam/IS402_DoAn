"""
ci_deploy_model.py — Deploy model lên ACI (staging) và AKS (production)
Sử dụng Azure ML SDK v2 Managed Online Endpoint (ACI) và Kubernetes Online Endpoint (AKS).

Chạy trong Azure DevOps Pipeline:
    python src/ci_deploy_model.py --target aci      # Deploy lên ACI (staging/test)
    python src/ci_deploy_model.py --target aks      # Deploy lên AKS (production)
    python src/ci_deploy_model.py --target all      # Deploy cả hai
"""

import os
import sys
import argparse
import time

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    KubernetesOnlineEndpoint,
    KubernetesOnlineDeployment,
    Environment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError


def get_config():
    """Đọc config từ environment variables."""
    return {
        "subscription_id": os.environ.get(
            "AZURE_SUBSCRIPTION_ID", "35d715f0-0211-4894-9c18-aea6e5787b86"
        ),
        "resource_group": os.environ.get("AZURE_RESOURCE_GROUP", "is402_doan"),
        "workspace_name": os.environ.get("AZURE_ML_WORKSPACE", "machinelearningproject"),
        "model_name": os.environ.get("MODEL_NAME", "fozzy-lightgbm-sales-model"),
        "aci_endpoint": os.environ.get("ACI_ENDPOINT_NAME", "fozzy-sales-staging"),
        "aks_endpoint": os.environ.get("AKS_ENDPOINT_NAME", "fozzy-sales-production"),
        "aks_compute": os.environ.get("AKS_COMPUTE_NAME", "aks-cluster"),
    }


def get_model_version(ml_client, model_name):
    """Lấy version mới nhất của model đã registered."""
    # Thử đọc từ file trước
    version_file = os.environ.get("MODEL_VERSION_FILE", "model_version.txt")
    if os.path.exists(version_file):
        with open(version_file) as f:
            return f.read().strip()

    # Fallback: lấy version mới nhất từ registry
    versions = list(ml_client.models.list(name=model_name))
    if not versions:
        raise ValueError(f"Không tìm thấy model: {model_name}")
    latest = max(versions, key=lambda m: int(m.version))
    return latest.version


def create_environment():
    """Tạo environment cho deployment."""
    return Environment(
        name="frozzy-lightgbm-inference-env",
        description="LightGBM inference environment",
        conda_file="src/conda_env.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    )


# ═════════════════════════════════════════════════════════════════════════════
# ACI DEPLOYMENT (Staging / Test)
# ═════════════════════════════════════════════════════════════════════════════
def deploy_aci(ml_client, config, model_version):
    """Deploy model lên ACI thông qua Managed Online Endpoint (staging)."""
    endpoint_name = config["aci_endpoint"]
    model_name = config["model_name"]
    deployment_name = "current"

    print("=" * 60)
    print("DEPLOY LÊN ACI (Staging)")
    print("=" * 60)

    # ── Tạo / cập nhật endpoint ──────────────────────────────────────────
    print(f"\nEndpoint: {endpoint_name}")
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Staging endpoint cho test (ACI-equivalent)",
        auth_mode="key",
    )

    try:
        existing = ml_client.online_endpoints.get(endpoint_name)
        print(f"   Endpoint đã tồn tại, sẽ cập nhật deployment.")
    except ResourceNotFoundError:
        print(f"   Tạo endpoint mới...")
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        print(f"   Endpoint đã tạo!")

    # ── Tạo deployment ───────────────────────────────────────────────────
    model = ml_client.models.get(name=model_name, version=model_version)
    env = create_environment()

    print(f"\nModel: {model_name} v{model_version}")
    print(f"Đang tạo deployment: {deployment_name}...")

    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        code_configuration=CodeConfiguration(
            code="src",
            scoring_script="score.py",
        ),
        environment=env,
        instance_type="Standard_DS2_v2",
        instance_count=1,
    )

    ml_client.online_deployments.begin_create_or_update(deployment).result()
    print(f"   Deployment tạo thành công!")

    # ── Route 100% traffic ───────────────────────────────────────────────
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    # ── Lấy scoring URI ──────────────────────────────────────────────────
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    print(f"\nScoring URI: {endpoint.scoring_uri}")
    print(f"ACI (Staging) deployment hoàn tất!")

    return endpoint.scoring_uri


# ═════════════════════════════════════════════════════════════════════════════
# AKS DEPLOYMENT (Production)
# ═════════════════════════════════════════════════════════════════════════════
def deploy_aks(ml_client, config, model_version):
    """Deploy model lên AKS thông qua Kubernetes Online Endpoint (production)."""
    endpoint_name = config["aks_endpoint"]
    model_name = config["model_name"]
    aks_compute = config["aks_compute"]
    deployment_name = "current"

    print("\n" + "=" * 60)
    print("DEPLOY LÊN AKS (Production)")
    print("=" * 60)

    # ── Kiểm tra AKS cluster ────────────────────────────────────────────
    try:
        compute = ml_client.compute.get(aks_compute)
        print(f"\nAKS compute: {aks_compute} (status: {compute.provisioning_state})")
    except ResourceNotFoundError:
        print(f"\nKhông tìm thấy AKS compute: {aks_compute}")
        print("   Bỏ qua AKS deployment. Hãy attach AKS cluster vào workspace trước.")
        return None

    # ── Tạo / cập nhật endpoint ──────────────────────────────────────────
    print(f"Endpoint: {endpoint_name}")
    endpoint = KubernetesOnlineEndpoint(
        name=endpoint_name,
        compute=aks_compute,
        description="Production endpoint trên AKS",
        auth_mode="key",
    )

    try:
        existing = ml_client.online_endpoints.get(endpoint_name)
        print(f"   Endpoint đã tồn tại, sẽ cập nhật deployment.")
    except ResourceNotFoundError:
        print(f"   Tạo endpoint mới...")
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        print(f"   Endpoint đã tạo!")

    # ── Tạo deployment ───────────────────────────────────────────────────
    model = ml_client.models.get(name=model_name, version=model_version)
    env = create_environment()

    print(f"\nModel: {model_name} v{model_version}")
    print(f"Đang tạo deployment: {deployment_name}...")

    deployment = KubernetesOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        code_configuration=CodeConfiguration(
            code="src",
            scoring_script="score.py",
        ),
        environment=env,
        instance_type="defaultInstanceType",
        instance_count=1,
    )

    ml_client.online_deployments.begin_create_or_update(deployment).result()
    print(f"   Deployment tạo thành công!")

    # ── Route 100% traffic ───────────────────────────────────────────────
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    # ── Lấy scoring URI ──────────────────────────────────────────────────
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    print(f"\nScoring URI: {endpoint.scoring_uri}")
    print(f"AKS (Production) deployment hoàn tất!")

    return endpoint.scoring_uri


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description="Deploy model lên ACI/AKS")
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=["aci", "aks", "all"],
        help="Deploy target: aci (staging), aks (production), hoặc all",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config()

    # ── Kết nối Azure ML ─────────────────────────────────────────────────
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential,
        config["subscription_id"],
        config["resource_group"],
        config["workspace_name"],
    )
    print(f"Đã kết nối workspace: {config['workspace_name']}")

    # ── Lấy model version ────────────────────────────────────────────────
    model_version = get_model_version(ml_client, config["model_name"])
    print(f"Sẽ deploy model: {config['model_name']} v{model_version}")

    # ── Deploy ───────────────────────────────────────────────────────────
    if args.target in ("aci", "all"):
        try:
            deploy_aci(ml_client, config, model_version)
        except Exception as e:
            print(f"ACI deployment thất bại: {e}")
            if args.target == "aci":
                sys.exit(1)

    if args.target in ("aks", "all"):
        try:
            deploy_aks(ml_client, config, model_version)
        except Exception as e:
            print(f"AKS deployment thất bại: {e}")
            if args.target == "aks":
                sys.exit(1)

    print("\n" + "=" * 60)
    print("DEPLOY HOÀN TẤT!")
    print("=" * 60)


if __name__ == "__main__":
    main()

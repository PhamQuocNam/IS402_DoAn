#!/bin/bash
set -e

# ═══════════════════════════════════════════════════════════════
# start.sh — Cấu hình & chạy Azure DevOps Agent
# ═══════════════════════════════════════════════════════════════

# Kiểm tra biến môi trường bắt buộc
if [ -z "$AZP_URL" ]; then
    echo "❌ ERROR: Thiếu biến AZP_URL (URL tổ chức Azure DevOps)"
    echo "   Ví dụ: https://dev.azure.com/your-organization"
    exit 1
fi

if [ -z "$AZP_TOKEN" ]; then
    echo "❌ ERROR: Thiếu biến AZP_TOKEN (Personal Access Token)"
    exit 1
fi

# Giá trị mặc định
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

# ── Xóa cấu hình cũ (nếu có) ────────────────────────────────
cleanup() {
    echo ""
    echo "🧹 Đang gỡ đăng ký agent..."
    ./config.sh remove --unattended \
        --auth pat \
        --token "$AZP_TOKEN" || true
}

# Gỡ đăng ký agent khi container bị dừng
trap 'cleanup; exit 0' EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

# ── Cấu hình agent ───────────────────────────────────────────
echo ""
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

# ── Chạy agent ────────────────────────────────────────────────
echo ""
echo "🚀 Khởi chạy agent..."
exec ./run.sh --once & wait $!

# Nếu muốn agent chạy liên tục (không tắt sau mỗi job):
# exec ./run.sh & wait $!
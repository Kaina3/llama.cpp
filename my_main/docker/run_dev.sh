#!/usr/bin/env bash
# run_dev.sh — build (if needed) and launch the general-purpose dev container
#
# Usage:
#   ./my_main/docker/run_dev.sh                     # interactive bash
#   ./my_main/docker/run_dev.sh python3 my_main/scripts/foo.py
#   GPUS='"device=0"' ./my_main/docker/run_dev.sh   # single GPU
#
# Environment variables:
#   GPUS      GPU selector for --gpus  (default: all)
#   IMAGE     Docker image name        (default: gemma4-ua-dev)
#   HF_TOKEN  Hugging Face token       (forwarded to container if set)
#
# Mounts inside the container:
#   /workspace              ← /home/anderson/llama/vishnu/  (rw)
#   /root/.ssh              ← /home/anderson/.ssh            (ro)
#   /root/.cache/huggingface ← ~/.cache/huggingface          (rw)
#   /root/.gitconfig        ← ~/.gitconfig                   (ro, if exists)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

IMAGE="${IMAGE:-kaina_gemma4-ua-dev}"
CONTAINER_NAME="${CONTAINER_NAME:-kaina_gemma4-ua-dev}"
GPUS="${GPUS:-all}"
HF_CACHE="${HOME}/.cache/huggingface"

# Build image if not already present
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    echo "[run_dev] Building Docker image '$IMAGE' (first run only, ~10min)..."
    docker build -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile.dev" "$SCRIPT_DIR"
fi

mkdir -p "$HF_CACHE"

# Mount ~/.gitconfig if it exists on the host
GITCONFIG_MOUNT=()
if [[ -f "${HOME}/.gitconfig" ]]; then
    GITCONFIG_MOUNT=(-v "${HOME}/.gitconfig:/root/.gitconfig:ro")
fi

# Forward HF_TOKEN only if set
HF_TOKEN_ENV=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    HF_TOKEN_ENV=(-e "HF_TOKEN=${HF_TOKEN}")
fi

# コンテナが起動していなければデーモンとして立ち上げる
if ! docker ps --filter "name=^${CONTAINER_NAME}$" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[run_dev] Starting container '$CONTAINER_NAME'..."
    docker run -d \
        --name "$CONTAINER_NAME" \
        --gpus "$GPUS" \
        -v "$REPO_ROOT":/workspace \
        -v "${HOME}/.ssh":/root/.ssh:ro \
        -v "$HF_CACHE":/root/.cache/huggingface \
        "${GITCONFIG_MOUNT[@]}" \
        "${HF_TOKEN_ENV[@]}" \
        -e HF_HOME=/root/.cache/huggingface \
        -e GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new" \
        -w /workspace \
        "$IMAGE" \
        sleep infinity
    echo "[run_dev] Container started."
else
    echo "[run_dev] Container '$CONTAINER_NAME' is already running."
fi

# コンテナにアタッチ
docker exec -it "$CONTAINER_NAME" "${@:-/bin/bash}"

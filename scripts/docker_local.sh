#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-credit-scoring:local}"
PORT="${PORT:-8000}"
docker build -t "$IMAGE" .
mkdir -p models data artifacts/mlflow
# MLflow stores absolute artifact paths; keep the host path inside the container.
mounts=(-v "$PWD/models:/app/models" -v "$PWD/data:/app/data" -v "$PWD/artifacts/mlflow:$PWD/artifacts/mlflow" -e "MLFLOW_DIR=$PWD/artifacts/mlflow")
user_args=(--user "$(id -u):$(id -g)" -e HOME=/tmp -e "USER=${USER:-local}")

if [[ "${1:-api}" == "train" ]]; then
    shift
    docker run --rm "${user_args[@]}" "${mounts[@]}" "$IMAGE" python -m src.data.make_dataset
    docker run --rm "${user_args[@]}" "${mounts[@]}" "$IMAGE" python -m src.models.train "$@"
    exit 0
fi

if [[ ! -f models/best_model.joblib ]]; then
    echo "Model missing. First run: bash scripts/docker_local.sh train --models log_reg --n-jobs 2" >&2
    exit 1
fi
echo "API: http://127.0.0.1:$PORT/docs (stop with Ctrl+C)"
docker run --rm --init "${user_args[@]}" -p "127.0.0.1:$PORT:8000" \
    -v "$PWD/models:/app/models:ro" "$IMAGE"

#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
action=${1:-deploy}
if [[ $# -gt 1 ]]; then echo "Usage: $0 [deploy|start|stop|restart|status|logs|errors|gpu-status|check|verify]" >&2; exit 2; fi
case "$action" in
  -h|--help|help)
    cat <<'HELP'
Usage: ./deploy.sh [deploy|start|stop|restart|status|logs|errors|gpu-status|check|verify]
  deploy   Default: install a private Python/CUDA environment, download pinned
           Ref2VA weights, start the API + eight GPU ranks, wait for real warmup.
  start    Start an already installed deployment and wait for readiness.
  stop     Stop the API and all eight GPU workers.
  restart  Stop and start with the current code/configuration; reuse dependencies.
  status   Return 0 only when the model service is ready.
  logs     Follow supervisor, API and GPU logs.
  errors   Show the original worker exception without loading the model again.
  gpu-status  Capture live GPU owners, parent processes and Docker containers.
  check    Check the installed CUDA/kernel environment and checkpoint layout.
  verify   Run live tuning A/B, duration and resolution checks; keep MP4 results.

Target: Linux with 8 NVIDIA H200 GPUs and driver >=570.26.
Optional configuration: .env (see .env.example); shell environment takes priority.
No Docker or systemd required. Ctrl-C during startup cancels the new service.
HELP
    exit 0 ;;
  deploy|start|stop|restart|status|logs|errors|gpu-status|check|verify) ;;
  *) echo "Unknown action: $action (use --help)" >&2; exit 2 ;;
esac
[[ $(uname -s) == Linux ]] || { echo "Deployment requires a Linux H200 host." >&2; exit 1; }

# Preserve explicit shell overrides when loading optional .env configuration.
keys=(HOST PORT CUDA_VISIBLE_DEVICES HF_TOKEN HF_ENDPOINT DATA_DIR CHECKPOINT_DIR MODEL_DIR ADAPTER_PATH
      READY_TIMEOUT TASK_TIMEOUT MAX_QUEUE MAX_UPLOAD_MB VAE_COMPILE DIT_COMPILE MODEL_LOAD_PARALLELISM
      SOL_ATTN_ENABLED SOL_ATTN_TAU SOL_ATTN_DENSE_STEPS CACHE_DIT_ENABLED CACHE_DIT_WARMUP
      CACHE_DIT_RDT CACHE_DIT_MAX_CONTINUOUS MAX_OUTPUT_PIXELS MAX_PACKED_TOKENS)
saved_keys=() saved_values=()
for key in "${keys[@]}"; do
  if value=$(printenv "$key"); then saved_keys+=("$key"); saved_values+=("$value"); fi
done
if [[ -f .env ]]; then set -a; source .env; set +a; fi
for index in "${!saved_keys[@]}"; do
  printf -v "${saved_keys[$index]}" '%s' "${saved_values[$index]}"
  export "${saved_keys[$index]}"
done

mkdir -p .runtime
export PYTHONPATH="$ROOT:$ROOT/vendor/sol_h3${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HF_HOME="$ROOT/.runtime/huggingface"
export UV_CACHE_DIR="$ROOT/.runtime/uv-cache"
export UV_PYTHON_INSTALL_DIR="$ROOT/.runtime/python"
export TRITON_CACHE_DIR="$ROOT/.runtime/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/.runtime/inductor-cache"
export TORCH_CUDA_ARCH_LIST=9.0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Fixed H200 profile. INT8 QKV requires the upstream SOL/BSA backend.
export H3_ULYSSES_COMM_DTYPE=bf16 H3_ULYSSES_OUTPUT_DTYPE=bf16
export H3_VAE_GLOBAL_BATCH=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
PYTHON="$ROOT/.venv/bin/python"

if [[ $action == gpu-status ]]; then
  if [[ ! -x $PYTHON ]]; then PYTHON=python3; fi
  exec "$PYTHON" -m navid.gpu_diagnostics
fi
if [[ $action == logs ]]; then
  touch .runtime/service.log .runtime/api.log .runtime/worker.log
  exec tail -n 60 -F .runtime/service.log .runtime/api.log .runtime/worker.log
fi
if [[ $action == verify ]]; then
  exec "$PYTHON" scripts/verify_tuning.py
fi
if [[ $action == stop || $action == status || $action == errors ]]; then
  if [[ ! -x $PYTHON ]]; then echo "Not deployed yet."; [[ $action == stop ]]; exit; fi
  exec "$PYTHON" -m navid.service "$action"
fi

trap 'echo "Deployment failed at line $LINENO. Fix the reported error and rerun ./deploy.sh." >&2' ERR
deployment_locked=0
if command -v flock >/dev/null 2>&1; then
  exec 9>.runtime/deploy.lock
  flock -n 9 || { echo "Another deployment command is running." >&2; exit 1; }
  deployment_locked=1
fi

if [[ $action == deploy ]]; then
  if [[ -x $PYTHON ]] && "$PYTHON" -m navid.service is-running; then
    "$PYTHON" -m navid.service start
    exit $?
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA driver / GPU passthrough is missing: nvidia-smi must work on this host." >&2; exit 1
  fi
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
  IFS=. read -r driver_major driver_minor _ <<< "$driver"
  if (( driver_major < 570 || (driver_major == 570 && driver_minor < 26) )); then
    echo "Driver $driver is too old; this CUDA 12.8 profile requires NVIDIA >=570.26." >&2; exit 1
  fi
  missing=()
  for name in curl git gcc g++ make sha256sum tar flock; do
    command -v "$name" >/dev/null 2>&1 || missing+=("$name")
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Installing host build tools: ${missing[*]}"
    if ! command -v apt-get >/dev/null 2>&1; then
      echo "Install curl, git, GCC/G++, make, coreutils, tar and util-linux, then rerun." >&2; exit 1
    fi
    privilege=()
    if [[ $EUID -ne 0 ]]; then
      command -v sudo >/dev/null 2>&1 || { echo "sudo/root is required to install missing build tools." >&2; exit 1; }
      privilege=(sudo)
    fi
    "${privilege[@]}" apt-get update
    "${privilege[@]}" apt-get install -y --no-install-recommends curl git build-essential ca-certificates coreutils tar util-linux
  fi
  if [[ $deployment_locked == 0 ]]; then exec 9>.runtime/deploy.lock; flock -n 9; fi
  uv_bin="$ROOT/.runtime/bin/uv"
  if [[ ! -x $uv_bin ]]; then
    case $(uname -m) in
      x86_64) platform=x86_64-unknown-linux-gnu; checksum=741ff1f5742c5a4a25d2f829e8395355e43f7a5ae2ebc6368e9ae2df0efb69cf ;;
      aarch64) platform=aarch64-unknown-linux-gnu; checksum=726b72a137fda33565143325f7d31c42cd30ff9ccdf067e00d124d37b4081cb2 ;;
      *) echo "Unsupported CPU architecture: $(uname -m)" >&2; exit 1 ;;
    esac
    mkdir -p .runtime/bin .runtime/uv-download
    archive="$ROOT/.runtime/uv-download/uv.tar.gz"
    curl --fail --location --retry 3 --connect-timeout 20 \
      "https://github.com/astral-sh/uv/releases/download/0.8.22/uv-$platform.tar.gz" -o "$archive"
    printf '%s  %s\n' "$checksum" "$archive" | sha256sum --check -
    tar -xzf "$archive" -C .runtime/uv-download
    install -m 755 ".runtime/uv-download/uv-$platform/uv" "$uv_bin"
  fi
  if [[ ! -x $PYTHON ]]; then
    "$uv_bin" python install 3.12
    "$uv_bin" venv --python 3.12 "$ROOT/.venv"
  fi
  fingerprint=$(sha256sum requirements.txt requirements.lock | sha256sum | cut -d ' ' -f 1)
  if [[ ! -f .runtime/requirements.sha256 || $(cat .runtime/requirements.sha256) != "$fingerprint" ]]; then
    "$uv_bin" pip install --python "$PYTHON" --index https://download.pytorch.org/whl/cu128 \
      --index-strategy unsafe-best-match -r requirements.txt -c requirements.lock
    "$uv_bin" pip check --python "$PYTHON"
    printf '%s\n' "$fingerprint" > .runtime/requirements.sha256
  fi
elif [[ ! -x $PYTHON ]]; then
  echo "Run ./deploy.sh first to install the environment." >&2; exit 1
fi

# Triton 3.6 may bundle a newer ptxas. Use CUDA 12.8 ptxas on R570 Hopper hosts.
export TRITON_PTXAS_PATH
TRITON_PTXAS_PATH=$("$PYTHON" -c 'import sysconfig; from pathlib import Path; p=Path(sysconfig.get_paths()["purelib"])/"nvidia/cuda_nvcc/bin/ptxas"; assert p.is_file(), f"Missing {p}"; print(p)')

if [[ $action == restart ]]; then "$PYTHON" -m navid.service stop; fi
if [[ $action == deploy || $action == check ]]; then
  "$PYTHON" -m navid.prepare preflight
fi
if [[ $action == deploy ]]; then
  "$PYTHON" -m navid.prepare download
elif [[ $action == check ]]; then
  "$PYTHON" -c 'from navid.prepare import validate_model, validate_adapter; from navid import config; validate_model(config.MODEL); validate_adapter(config.ADAPTER); print("Checkpoint validation passed")'
  exit 0
fi
"$PYTHON" -m navid.service start

#!/usr/bin/env bash
# Provision a Colab GPU, upload this repo's training files, run train.py,
# and download the checkpoint. Default GPU is T4; override with --gpu or
# COLAB_GPU. Extra arguments are forwarded to train.py.
#
#   ./colab_train.sh --resume --unfreeze fc --data-dir data_real --epochs 8
#   ./colab_train.sh --gpu L4 --keep -- --resume --unfreeze layer4
set -euo pipefail

# Google often grants only profile/email for this OAuth client.
# Without this, oauthlib raises "Scope has changed" and login dies.
export OAUTHLIB_RELAX_TOKEN_SCOPE=1

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

GPU="${COLAB_GPU:-T4}"
SESSION="${COLAB_SESSION:-trainer}"
KEEP=0
TIMEOUT="${COLAB_TIMEOUT:-21600}"
DATA_DIR=""
TRAIN_ARGS=()

usage() {
    cat <<'EOF'
Usage: ./colab_train.sh [helper flags] [train.py flags...]

Helper flags:
  --gpu GPU          Colab accelerator (default: T4, or $COLAB_GPU)
  --session NAME     Session name (default: trainer, or $COLAB_SESSION)
  --keep             Leave the VM running after training
  --timeout SECONDS  colab exec timeout (default: 21600)
  --data-dir DIR     Dataset root to upload (also passed to train.py)
  -h, --help         Show this help

All other flags are forwarded to train.py.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --session)
            SESSION="$2"
            shift 2
            ;;
        --keep)
            KEEP=1
            shift
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            TRAIN_ARGS+=(--data-dir "$2")
            shift 2
            ;;
        --)
            shift
            TRAIN_ARGS+=("$@")
            break
            ;;
        *)
            TRAIN_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$DATA_DIR" && ${#TRAIN_ARGS[@]} -gt 0 ]]; then
    for i in "${!TRAIN_ARGS[@]}"; do
        if [[ "${TRAIN_ARGS[$i]}" == "--data-dir" ]]; then
            DATA_DIR="${TRAIN_ARGS[$((i + 1))]}"
            break
        fi
    done
fi

if [[ -z "$DATA_DIR" ]]; then
    if [[ -d data_real ]]; then
        DATA_DIR="data_real"
        TRAIN_ARGS+=(--data-dir data_real)
    else
        DATA_DIR="data"
    fi
fi

if [[ ! -d "$DATA_DIR" ]]; then
    echo "Missing dataset directory: $DATA_DIR" >&2
    exit 1
fi

RESUME=0
MODEL_PATH="models/resnext50_metal_plastic.pt"

if [[ ${#TRAIN_ARGS[@]} -gt 0 ]]; then
    for i in "${!TRAIN_ARGS[@]}"; do
        case "${TRAIN_ARGS[$i]}" in
            --resume)
                RESUME=1
                ;;
            --model)
                MODEL_PATH="${TRAIN_ARGS[$((i + 1))]}"
                ;;
        esac
    done
fi

ensure_colab_cli() {
    if command -v colab >/dev/null 2>&1; then
        return
    fi

    echo "google-colab-cli is not installed. Installing..."

    if command -v uv >/dev/null 2>&1; then
        uv tool install google-colab-cli
    else
        python3 -m pip install google-colab-cli
    fi
}

ensure_colab_cli

BUNDLE="$ROOT/.colab_bundle.zip"
RUN_FILE="$ROOT/.colab_run.py"
EXTRACT_FILE="$ROOT/.colab_extract.py"
rm -f "$BUNDLE" "$RUN_FILE" "$EXTRACT_FILE"

DATA_SIZE="$(du -sh "$DATA_DIR" | cut -f1)"
echo "Packing $DATA_DIR ($DATA_SIZE) plus train.py into a Colab zip. This can take a minute with no upload yet..."

python3 - "$BUNDLE" "$DATA_DIR" "$RESUME" "$MODEL_PATH" <<'PY'
from pathlib import Path
import sys
import zipfile

bundle, data_dir, resume, model_path = sys.argv[1:5]
data_dir = Path(data_dir)
model_path = Path(model_path)
skip = {"__pycache__", ".DS_Store"}

def log(message):
    print(message, flush=True)

if not data_dir.is_dir():
    raise SystemExit(f"Missing dataset directory: {data_dir}")

files = [
    path
    for path in data_dir.rglob("*")
    if path.is_file()
    and path.name not in skip
    and "__pycache__" not in path.parts
]
log(f"Zipping {len(files)} files from {data_dir}...")

with zipfile.ZipFile(bundle, "w", zipfile.ZIP_STORED) as archive:
    for name in ("train.py", "trashnet.py"):
        archive.write(name)

    for index, path in enumerate(files, start=1):
        archive.write(path, path.as_posix())
        if index == 1 or index == len(files) or index % 50 == 0:
            log(f"  {index}/{len(files)} files")

    if resume == "1":
        source = model_path
        if not source.exists() and model_path == Path(
            "models/resnext50_metal_plastic.pt"
        ):
            legacy = Path("resnext50_metal_plastic.pt")
            if legacy.exists():
                source = legacy

        if not source.exists():
            raise SystemExit(
                f"Cannot --resume: {model_path} was not found."
            )

        archive.write(source, Path(model_path).as_posix())
        log(f"Bundled checkpoint {source} as {model_path}")

log(f"Wrote {bundle}")
PY

if [[ ${#TRAIN_ARGS[@]} -gt 0 ]]; then
    ARGS_JSON="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[2:]))' -- "${TRAIN_ARGS[@]}")"
else
    ARGS_JSON='[]'
fi

python3 - "$RUN_FILE" "$ARGS_JSON" <<'PY'
from pathlib import Path
import json
import sys

run_file = Path(sys.argv[1])
train_args = json.loads(sys.argv[2])
run_file.write_text(
    "import json\n"
    "import runpy\n"
    "import sys\n"
    f"sys.argv = ['train.py'] + {json.dumps(train_args)}\n"
    "runpy.run_path('train.py', run_name='__main__')\n",
    encoding="utf-8",
)
print(f"Wrote {run_file} with {train_args}")
PY

cat > "$EXTRACT_FILE" <<'PY'
import zipfile
from pathlib import Path

bundle = Path("/content/colab_bundle.zip")
if not bundle.exists():
    bundle = Path("colab_bundle.zip")

with zipfile.ZipFile(bundle) as archive:
    names = archive.namelist()
    archive.extractall("/content")

print("Extracted %s files into /content" % len(names))
PY

cleanup() {
    rm -f "$BUNDLE" "$RUN_FILE" "$EXTRACT_FILE"
    if [[ "${KEEP:-0}" -eq 0 ]]; then
        if command -v colab >/dev/null 2>&1 && colab status -s "$SESSION" >/dev/null 2>&1; then
            echo "Stopping session $SESSION"
            colab stop -s "$SESSION" || true
        fi
    else
        echo "Leaving session $SESSION running (--keep)"
    fi
}

download_if_present() {
    local remote="$1"
    local local_path="$2"
    if colab download -s "$SESSION" "$remote" "$local_path"; then
        echo "Downloaded $remote -> $local_path"
    else
        echo "Skip download (missing on VM): $remote"
    fi
}

trap cleanup EXIT

echo "Checking Colab session '$SESSION' on $GPU..."
echo "First run prints a Google sign-in URL. Open it, then paste the authorization code (not the URL)."
if colab status -s "$SESSION"; then
    echo "Reusing existing Colab session $SESSION"
else
    echo "Starting Colab session $SESSION on $GPU"
    colab new -s "$SESSION" --gpu "$GPU"
fi

echo "Uploading training bundle"
colab upload -s "$SESSION" "$BUNDLE" colab_bundle.zip

echo "Extracting training bundle on the VM"
colab exec -s "$SESSION" --timeout 120 --env MPLBACKEND=Agg -f "$EXTRACT_FILE"

echo "Installing Python packages (Colab already has CUDA PyTorch)"
colab install -s "$SESSION" -r requirements-colab.txt

echo "Running train.py on $GPU"
colab exec -s "$SESSION" --timeout "$TIMEOUT" --env MPLBACKEND=Agg -f "$RUN_FILE"

mkdir -p models

download_if_present "$MODEL_PATH" "$MODEL_PATH"
download_if_present confusion_matrix.png confusion_matrix.png
download_if_present misclassified_validation.png misclassified_validation.png
download_if_present misclassified_validation.txt misclassified_validation.txt

echo "Training finished"

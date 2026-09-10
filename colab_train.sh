#!/usr/bin/env bash
# Provision a Colab GPU, upload this repo's training files, run train.py,
# and download the checkpoint. Default GPU is T4; override with --gpu or
# COLAB_GPU. Extra arguments are forwarded to train.py.
#
#   ./colab_train.sh --drive --resume --unfreeze fc --epochs 8
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
USE_DRIVE=0
DRIVE_DIR="${COLAB_DRIVE_DIR:-stage1_binary_v2}"
TRAIN_ARGS=()
PHASE=start
ONE_SHOT_UPLOAD_MAX=$((8 * 1024 * 1024))

usage() {
    cat <<'EOF'
Usage: ./colab_train.sh [helper flags] [train.py flags...]

Helper flags:
  --gpu GPU          Colab accelerator (default: T4, or $COLAB_GPU)
  --session NAME     Session name (default: trainer, or $COLAB_SESSION)
  --keep             Leave the VM running after training
  --timeout SECONDS  colab exec timeout (default: 21600)
  --drive            Mount Google Drive instead of uploading images
  --drive-dir PATH   Folder under MyDrive (default: stage1_binary_v2)
  --data-dir DIR     Local dataset root to upload (ignored with --drive)
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
        --drive)
            USE_DRIVE=1
            shift
            ;;
        --drive-dir)
            DRIVE_DIR="$2"
            USE_DRIVE=1
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

if [[ "$USE_DRIVE" -eq 1 ]]; then
    FILTERED_ARGS=()
    SKIP_VALUE=0
    if [[ ${#TRAIN_ARGS[@]} -gt 0 ]]; then
        for arg in "${TRAIN_ARGS[@]}"; do
            if [[ "$SKIP_VALUE" -eq 1 ]]; then
                SKIP_VALUE=0
                continue
            fi
            if [[ "$arg" == "--data-dir" ]]; then
                SKIP_VALUE=1
                continue
            fi
            FILTERED_ARGS+=("$arg")
        done
    fi
    TRAIN_ARGS=()
    if [[ ${#FILTERED_ARGS[@]} -gt 0 ]]; then
        TRAIN_ARGS=("${FILTERED_ARGS[@]}")
    fi
    TRAIN_ARGS+=(--data-dir /content/data_real)
    DATA_DIR=""
elif [[ -z "$DATA_DIR" ]]; then
    if [[ -d data_real ]]; then
        DATA_DIR="data_real"
        TRAIN_ARGS+=(--data-dir data_real)
    else
        DATA_DIR="data"
    fi
fi

if [[ "$USE_DRIVE" -eq 0 && ! -d "$DATA_DIR" ]]; then
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
    if ! command -v colab >/dev/null 2>&1; then
        echo "google-colab-cli is not installed. Installing..."

        if command -v uv >/dev/null 2>&1; then
            uv tool install google-colab-cli
        else
            python3 -m pip install google-colab-cli
        fi
    fi

    COLAB_BIN="$(command -v colab)"
    COLAB_PY="$(head -1 "$COLAB_BIN" | tr -d '#!')"
    if ! "$COLAB_PY" -c "from jupyter_kernel_client import KernelClient" >/dev/null 2>&1; then
        echo "Pinning jupyter-kernel-client==0.15.0 so colab exec works..."
        if command -v uv >/dev/null 2>&1; then
            uv pip install --python "$COLAB_PY" "jupyter-kernel-client==0.15.0"
        else
            "$COLAB_PY" -m pip install "jupyter-kernel-client==0.15.0"
        fi
    fi
}

ensure_colab_cli

BUNDLE="$ROOT/.colab_bundle.zip"
RUN_FILE="$ROOT/.colab_run.py"
EXTRACT_FILE="$ROOT/.colab_extract.py"
PREPARE_FILE="$ROOT/.colab_prepare.py"
CHECKPOINT_FILE="$ROOT/.colab_checkpoint.py"
LAST_EXEC_LOG="$ROOT/.colab_last_exec.log"
rm -f "$BUNDLE" "$RUN_FILE" "$EXTRACT_FILE" "$PREPARE_FILE" "$CHECKPOINT_FILE" "$LAST_EXEC_LOG"

if [[ "$USE_DRIVE" -eq 1 ]]; then
    echo "Packing training scripts only. Images will come from Google Drive ($DRIVE_DIR)."
    PACK_DATA=""
    ZIP_RESUME=0
else
    DATA_SIZE="$(du -sh "$DATA_DIR" | cut -f1)"
    echo "Packing $DATA_DIR ($DATA_SIZE) plus train.py into a Colab zip. This can take a minute with no upload yet..."
    PACK_DATA="$DATA_DIR"
    ZIP_RESUME="$RESUME"
fi

python3 - "$BUNDLE" "$PACK_DATA" "$ZIP_RESUME" "$MODEL_PATH" <<'PY'
from pathlib import Path
import sys
import zipfile

bundle, data_dir, resume, model_path = sys.argv[1:5]
data_dir = Path(data_dir) if data_dir else None
model_path = Path(model_path)
skip = {"__pycache__", ".DS_Store"}
code_files = (
    "train.py",
    "trashnet.py",
    "split_data.py",
    "prepare_drive_data.py",
)

def log(message):
    print(message, flush=True)

with zipfile.ZipFile(bundle, "w", zipfile.ZIP_STORED) as archive:
    for name in code_files:
        archive.write(name)

    if data_dir is not None:
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
    "runpy.run_path('train.py', run_name='__main__')\n"
    "print('COLAB_STEP_OK', flush=True)\n",
    encoding="utf-8",
)
print(f"Wrote {run_file} with {train_args}")
PY

if [[ "$DRIVE_DIR" == /content/drive/* || "$DRIVE_DIR" == /* ]]; then
    DRIVE_SOURCE="$DRIVE_DIR"
else
    DRIVE_SOURCE="/content/drive/MyDrive/$DRIVE_DIR"
fi

python3 - "$PREPARE_FILE" "$DRIVE_SOURCE" <<'PY'
from pathlib import Path
import json
import sys

prepare_file = Path(sys.argv[1])
source = sys.argv[2]
prepare_file.write_text(
    "import json\n"
    "import runpy\n"
    "import sys\n"
    "sys.argv = ['prepare_drive_data.py', '--source', "
    f"{json.dumps(source)}, '--dest', '/content/data_real']\n"
    "runpy.run_path('prepare_drive_data.py', run_name='__main__')\n"
    "print('COLAB_STEP_OK', flush=True)\n",
    encoding="utf-8",
)
print(f"Wrote {prepare_file} for {source}")
PY

python3 - "$CHECKPOINT_FILE" <<'PY'
from pathlib import Path
import sys

Path(sys.argv[1]).write_text(
    "import shutil\n"
    "from pathlib import Path\n"
    "dest = Path('/content/models/resnext50_metal_plastic.pt')\n"
    "if dest.exists():\n"
    "    print('Using checkpoint already on the VM', dest, flush=True)\n"
    "else:\n"
    "    dest.parent.mkdir(parents=True, exist_ok=True)\n"
    "    candidates = [\n"
    "        Path('/content/drive/MyDrive/metal-plastic-sorting/resnext50_metal_plastic.pt'),\n"
    "        Path('/content/drive/MyDrive/stage1_binary_v2/resnext50_metal_plastic.pt'),\n"
    "    ]\n"
    "    for src in candidates:\n"
    "        if src.is_file():\n"
    "            shutil.copy2(src, dest)\n"
    "            print('Copied checkpoint from', src, flush=True)\n"
    "            break\n"
    "    else:\n"
    "        raise SystemExit(\n"
    "            'Missing checkpoint. Copy models/resnext50_metal_plastic.pt '\n"
    "            'to MyDrive/metal-plastic-sorting/ and re-run.'\n"
    "        )\n"
    "print('COLAB_STEP_OK', flush=True)\n",
    encoding="utf-8",
)
print("Wrote", sys.argv[1])
PY

cat > "$EXTRACT_FILE" <<'PY'
import zipfile
from pathlib import Path

content = Path("/content")
search_roots = [content, Path("/"), Path.cwd()]
seen = set()
parts = []
for root in search_roots:
    if not root.exists():
        continue
    for part in sorted(root.glob("colab_bundle.part.*")):
        key = str(part.resolve()) if part.exists() else str(part)
        if key in seen:
            continue
        seen.add(key)
        parts.append(part)

candidates = [
    content / "colab_bundle.zip",
    Path("colab_bundle.zip"),
    Path("/colab_bundle.zip"),
]
for root in search_roots:
    if not root.exists():
        continue
    for found in root.glob("colab_bundle.zip"):
        candidates.append(found)

bundle = next((path for path in candidates if path.is_file()), content / "colab_bundle.zip")

if bundle.is_file() and parts:
    print("Ignoring leftover upload parts; using", bundle, flush=True)
    for part in parts:
        part.unlink()
    parts = []

if parts:
    print("Joining %s upload parts..." % len(parts), flush=True)
    with bundle.open("wb") as out:
        for index, part in enumerate(parts, start=1):
            out.write(part.read_bytes())
            part.unlink()
            if index == 1 or index == len(parts) or index % 25 == 0:
                print("  joined %s/%s" % (index, len(parts)), flush=True)

if not bundle.is_file():
    listing = sorted(p.name for p in content.iterdir()) if content.is_dir() else []
    print("Zip not found. /content has:", listing, flush=True)
    raise SystemExit("Missing colab_bundle.zip after upload")

print("Extracting %s..." % bundle, flush=True)
with zipfile.ZipFile(bundle) as archive:
    names = archive.namelist()
    archive.extractall("/content")

print("Extracted %s files into /content" % len(names), flush=True)
if not Path("/content/train.py").is_file():
    raise SystemExit("Extract finished but /content/train.py is missing")
print("COLAB_STEP_OK", flush=True)
PY

cleanup() {
    rm -f "$BUNDLE" "$RUN_FILE" "$EXTRACT_FILE" "$PREPARE_FILE" "$CHECKPOINT_FILE" "$LAST_EXEC_LOG"
    if [[ "${KEEP:-0}" -eq 1 ]]; then
        echo "Leaving session $SESSION running (--keep)"
        return
    fi
    if [[ "$PHASE" != "start" && "$PHASE" != "done" ]]; then
        echo "Leaving session $SESSION running so a re-run can skip the upload."
        echo "Re-run the same command. Pass --keep if you want to inspect the VM."
        return
    fi
    if command -v colab >/dev/null 2>&1 && colab_session_is_up; then
        echo "Stopping session $SESSION"
        colab stop -s "$SESSION" || true
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

# colab status prints "not found" but still exits 0.
colab_session_is_up() {
    local output
    output="$(colab status -s "$SESSION" 2>&1)" || true
    if [[ "${1:-}" == "print" ]]; then
        printf '%s\n' "$output"
    fi
    printf '%s\n' "$output" | grep -Eq 'IDLE|BUSY|READY'
}

colab_exec() {
    local timeout="$1"
    local file="$2"
    set +e
    colab exec -s "$SESSION" --timeout "$timeout" -f "$file" 2>&1 | tee "$LAST_EXEC_LOG"
    set -e
    if grep -q 'COLAB_STEP_OK' "$LAST_EXEC_LOG"; then
        return 0
    fi
    echo "Colab step failed. The VM Python never printed COLAB_STEP_OK." >&2
    return 1
}

vm_ls() {
    colab ls -s "$SESSION" 2>/dev/null || true
}

vm_has() {
    vm_ls | grep -q "$1"
}

bundle_bytes() {
    if stat -f%z "$BUNDLE" >/dev/null 2>&1; then
        stat -f%z "$BUNDLE"
    else
        stat -c%s "$BUNDLE"
    fi
}

upload_bundle() {
    local size
    size="$(bundle_bytes)"
    if [[ "$size" -le "$ONE_SHOT_UPLOAD_MAX" ]]; then
        echo "Uploading $(du -h "$BUNDLE" | cut -f1) in one request..."
        if colab upload -s "$SESSION" "$BUNDLE" /content/colab_bundle.zip; then
            return
        fi
        echo "One-shot upload failed; falling back to 4MB chunks."
    else
        echo "Uploading $(du -h "$BUNDLE" | cut -f1) in 4MB chunks..."
    fi
    python3 colab_upload.py "$SESSION" "$BUNDLE" content/colab_bundle.part
}

ensure_scripts_on_vm() {
    if vm_has 'train.py'; then
        echo "Training scripts already on the VM; skipping upload."
        return
    fi
    echo "Uploading training scripts (photos and checkpoint stay on Drive)"
    upload_bundle
    PHASE=uploaded
    echo "Extracting training scripts on the VM"
    colab_exec 120 "$EXTRACT_FILE"
}

mount_google_drive() {
    echo "Mounting Google Drive on the VM..."
    "$COLAB_PY" "$ROOT/colab_drivemount.py" "$SESSION"
}

stage_checkpoint_on_drive() {
    local my_drive dest
    [[ "$RESUME" -eq 1 ]] || return 0
    [[ -f "$MODEL_PATH" ]] || return 0

    for my_drive in "$HOME/Library/CloudStorage"/GoogleDrive-*/"My Drive"; do
        if [[ -d "$my_drive" ]]; then
            dest="$my_drive/metal-plastic-sorting/resnext50_metal_plastic.pt"
            mkdir -p "$(dirname "$dest")"
            if [[ -f "$dest" ]]; then
                echo "Checkpoint already on Google Drive at metal-plastic-sorting/"
                return 0
            fi
            echo "Copying checkpoint to Google Drive (once). Colab will read it from there."
            cp "$MODEL_PATH" "$dest"
            return 0
        fi
    done
}

trap cleanup EXIT

if [[ "$USE_DRIVE" -eq 1 && "$RESUME" -eq 1 ]]; then
    stage_checkpoint_on_drive
fi

echo "Checking Colab session '$SESSION' on $GPU..."
echo "First run prints a Google sign-in URL. Open it, then paste the authorization code (not the URL)."
if colab_session_is_up print; then
    echo "Reusing existing Colab session $SESSION"
else
    echo "Starting Colab session $SESSION on $GPU"
    colab new -s "$SESSION" --gpu "$GPU"
    echo "Waiting a few seconds for the T4 proxy to come up..."
    sleep 5
fi

if [[ "$USE_DRIVE" -eq 1 ]]; then
    ensure_scripts_on_vm
    PHASE=uploaded
    mount_google_drive
    echo "Linking Drive photos into train/validation/test (NonPlastic -> metal, Plastic -> plastic)"
    colab_exec 600 "$PREPARE_FILE"
    if [[ "$RESUME" -eq 1 ]]; then
        echo "Copying checkpoint from Drive onto the VM"
        colab_exec 120 "$CHECKPOINT_FILE"
    fi
elif vm_has 'train.py'; then
    echo "Training scripts already on the VM; skipping upload."
    PHASE=uploaded
elif vm_has 'colab_bundle'; then
    echo "Found a previous bundle on the VM; extracting it instead of uploading again."
    PHASE=uploaded
    if ! colab_exec 120 "$EXTRACT_FILE"; then
        echo "Extract failed; uploading a fresh bundle."
        upload_bundle
        colab_exec 120 "$EXTRACT_FILE"
    fi
else
    echo "Uploading local dataset (this is the slow path; prefer --drive)"
    upload_bundle
    PHASE=uploaded
    echo "Extracting training bundle on the VM"
    colab_exec 120 "$EXTRACT_FILE"
fi

echo "Installing Python packages (Colab already has CUDA PyTorch)"
colab install -s "$SESSION" -r requirements-colab.txt

echo "Running train.py on $GPU"
colab_exec "$TIMEOUT" "$RUN_FILE"

mkdir -p models

download_if_present "$MODEL_PATH" "$MODEL_PATH"
download_if_present confusion_matrix.png confusion_matrix.png
download_if_present misclassified_validation.png misclassified_validation.png
download_if_present misclassified_validation.txt misclassified_validation.txt

PHASE=done
echo "Training finished"

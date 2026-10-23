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
DATA_ZIP="${COLAB_DATA_ZIP:-belt_stage1_224.zip}"
LOCAL_ZIP="${COLAB_LOCAL_ZIP:-$HOME/Downloads/$DATA_ZIP}"
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
  --drive            Upload ~/Downloads/belt_stage1_224.zip from this
                     computer. Drive's copy of that zip is not read.
                     Weights still come from --model on this computer.
  --drive-dir PATH   Folder under MyDrive (default: stage1_binary_v2).
                     Used only if the local zip is not how photos arrive.
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

STAMP="$(date +%Y%m%d-%H%M%S)"
RESULT_NAME="$(basename "${MODEL_PATH%.pt}")_${STAMP}.pt"
RESULT_PATH="models/${RESULT_NAME}"
if [[ "$RESUME" -eq 1 ]]; then
    echo "Starting weights on this computer: $MODEL_PATH"
    echo "Trained weights will be saved as a new file: $RESULT_PATH"
    echo "A Drive checkpoint with the same name is not loaded."
fi
if [[ "$USE_DRIVE" -eq 1 ]]; then
    if [[ ! -f "$LOCAL_ZIP" ]]; then
        echo "Cannot train. This computer has no photo zip at $LOCAL_ZIP" >&2
        echo "Drive's copy is not used. Colab was still serving the 5 Oct zip after this file changed." >&2
        exit 1
    fi
    LOCAL_ZIP_BYTES="$(python3 -c 'import os,sys; print(os.path.getsize(sys.argv[1]))' "$LOCAL_ZIP")"
    echo "Photos from this computer: $LOCAL_ZIP ($LOCAL_ZIP_BYTES bytes)"
    echo "Drive's $DATA_ZIP is not read."
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
ASSEMBLE_FILE="$ROOT/.colab_assemble_checkpoint.py"
LAST_EXEC_LOG="$ROOT/.colab_last_exec.log"
rm -f "$BUNDLE" "$RUN_FILE" "$EXTRACT_FILE" "$PREPARE_FILE" "$ASSEMBLE_FILE" "$LAST_EXEC_LOG"

if [[ "$USE_DRIVE" -eq 1 ]]; then
    echo "Packing training scripts only."
    echo "Photos will be uploaded from $LOCAL_ZIP. Drive's zip is not read."
    echo "The checkpoint named by --model is uploaded from this computer."
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
code_files = [
    "train.py",
    "trashnet.py",
    "split_data.py",
    "prepare_drive_data.py",
    "preprocess.py",
    "colab_copy_checkpoint.py",
    "stage1_locked_test_stems.txt",
]
for extra in ("stage1_locked_val_stems.txt",):
    if Path(extra).is_file():
        code_files.append(extra)

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

python3 - "$RUN_FILE" "$ARGS_JSON" "$RESULT_NAME" <<'PY'
from pathlib import Path
import json
import sys

run_file = Path(sys.argv[1])
train_args = json.loads(sys.argv[2])
result_name = sys.argv[3]
model_rel = "models/resnext50_metal_plastic.pt"
if "--model" in train_args:
    model_rel = train_args[train_args.index("--model") + 1]
run_file.write_text(
    "import json\n"
    "import runpy\n"
    "import shutil\n"
    "import sys\n"
    "from pathlib import Path\n"
    f"sys.argv = ['train.py'] + {json.dumps(train_args)}\n"
    "runpy.run_path('train.py', run_name='__main__')\n"
    f"src = Path('/content') / {json.dumps(model_rel)}\n"
    "if not src.is_file():\n"
    "    raise SystemExit('train.py finished without writing the checkpoint')\n"
    "shutil.copy2(src, Path('/content/best.pt'))\n"
    "drive_root = Path('/content/drive/MyDrive')\n"
    "if drive_root.is_dir():\n"
    "    dest_dir = drive_root / 'metal-plastic-sorting'\n"
    "    dest_dir.mkdir(parents=True, exist_ok=True)\n"
    f"    shutil.copy2(src, dest_dir / {json.dumps(result_name)})\n"
    "    print('Saved a Drive copy of the trained file. The next run does not read it.', flush=True)\n"
    "    for name in (\n"
    "        'confusion_matrix.png',\n"
    "        'misclassified_validation.png',\n"
    "        'misclassified_validation.txt',\n"
    "    ):\n"
    "        art = Path('/content') / name\n"
    "        if art.is_file():\n"
    "            shutil.copy2(art, dest_dir / name)\n"
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

python3 - "$PREPARE_FILE" "$DRIVE_SOURCE" "$DATA_ZIP" <<'PY'
from pathlib import Path
import json
import sys

prepare_file = Path(sys.argv[1])
source = sys.argv[2]
zip_path = "/content/" + sys.argv[3]
prepare_file.write_text(
    "import json\n"
    "import runpy\n"
    "import sys\n"
    "sys.argv = ['prepare_drive_data.py', '--source', "
    f"{json.dumps(source)}, '--zip', {json.dumps(zip_path)}, "
    "'--dest', '/content/data_real']\n"
    "runpy.run_path('prepare_drive_data.py', run_name='__main__')\n"
    "print('COLAB_STEP_OK', flush=True)\n",
    encoding="utf-8",
)
print(f"Wrote {prepare_file} for {source} (zip {zip_path})")
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
    rm -f "$BUNDLE" "$RUN_FILE" "$EXTRACT_FILE" "$PREPARE_FILE" "$ASSEMBLE_FILE" "$LAST_EXEC_LOG"
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
    # Jupyter contents is rooted at /, so VM /content/foo is content/foo.
    # Do not use content/models/: listing that folder hangs on the 88MB .pt.
    if colab download -s "$SESSION" "$remote" "$local_path"; then
        echo "Downloaded $remote -> $local_path"
        return 0
    fi
    echo "Skip download (missing on VM): $remote"
    return 0
}

download_checkpoint() {
    mkdir -p models
    if [[ -e "$RESULT_PATH" ]]; then
        echo "Refusing to overwrite $RESULT_PATH" >&2
        return 1
    fi
    if colab download -s "$SESSION" content/best.pt "$RESULT_PATH"; then
        echo "Trained checkpoint saved to $RESULT_PATH"
        echo "Left the starting file in place: $MODEL_PATH"
        return 0
    fi
    echo "Failed to download the checkpoint. It is on the VM at /content/best.pt." >&2
    echo "The starting file was not changed: $MODEL_PATH" >&2
    return 1
}

# A real terminal can take the sign-in URL. The lab job has no
# controlling terminal, and /dev/tty then fails with "Device not configured".
tty_is_usable() {
    : >/dev/tty 2>/dev/null
}

# colab status prints "not found" but still exits 0.
colab_session_is_up() {
    local output tmp
    tmp="$(mktemp)"
    # Show the sign-in URL as it arrives. Command substitution would
    # hide it until this command exits, which is after the code is pasted.
    set +e
    set +o pipefail
    if tty_is_usable; then
        colab status -s "$SESSION" 2>&1 | tee /dev/tty >"$tmp"
    else
        colab status -s "$SESSION" 2>&1 | tee "$tmp"
    fi
    set -o pipefail
    set -e
    output="$(cat "$tmp")"
    rm -f "$tmp"
    if [[ "${1:-}" == "print" ]] && tty_is_usable; then
        printf '%s\n' "$output"
    fi
    printf '%s\n' "$output" | grep -Eq 'IDLE|BUSY|READY'
}

revive_colab_kernel() {
    echo "Colab kernel looks wedged. Restarting it."
    echo "The VM disk stays, including /content/data_real and the checkpoint."
    colab restart-kernel -s "$SESSION" || true
    echo "Waiting for a fresh kernel..."
    sleep 8
}

kernel_looks_dead() {
    grep -Eq \
        'Timeout waiting for output|Connection was lost|Transport endpoint is not connected|Kernel is dead|Websocket connection' \
        "$LAST_EXEC_LOG"
}

colab_exec() {
    local timeout="$1"
    local file="$2"
    local attempt=1
    while true; do
        set +e
        colab exec -s "$SESSION" --timeout "$timeout" -f "$file" 2>&1 | tee "$LAST_EXEC_LOG"
        set -e
        if grep -q 'COLAB_STEP_OK' "$LAST_EXEC_LOG"; then
            return 0
        fi
        if kernel_looks_dead && [[ "$attempt" -lt 3 ]]; then
            revive_colab_kernel
            attempt=$((attempt + 1))
            continue
        fi
        echo "Colab step failed. The VM Python never printed COLAB_STEP_OK." >&2
        if kernel_looks_dead; then
            echo "Kernel still dead after restart. Stop the session and start a new T4:" >&2
            echo "  colab stop -s $SESSION" >&2
            echo "  ./colab_train.sh --keep --drive --resume --unfreeze fc --epochs 4" >&2
        fi
        return 1
    done
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
    echo "Uploading training scripts"
    upload_bundle
    PHASE=uploaded
    echo "Extracting training scripts on the VM"
    colab_exec 120 "$EXTRACT_FILE"
}

mount_google_drive() {
    echo "Mounting Google Drive on the VM..."
    "$COLAB_PY" "$ROOT/colab_drivemount.py" "$SESSION"
}

upload_exact_file() {
    local src="$1"
    local dest_rel="$2"
    local tag="$3"
    local expected prefix
    expected="$(python3 -c 'import os,sys; print(os.path.getsize(sys.argv[1]))' "$src")"
    echo "Uploading $src"
    echo "  $expected bytes -> /content/$dest_rel"
    prefix="content/${tag}_${STAMP}.part"
    python3 "$ROOT/colab_upload.py" "$SESSION" "$src" "$prefix"
    python3 - "$ASSEMBLE_FILE" "$dest_rel" "$tag" "$STAMP" "$expected" <<'PY'
from pathlib import Path
import sys

script_path, dest_rel, tag, stamp, expected = sys.argv[1:]
expected = int(expected)
text = f"""from pathlib import Path

dest_rel = {dest_rel!r}
tag = {tag!r}
stamp = {stamp!r}
expected = {expected}
content = Path("/content")
parts = sorted(content.glob(f"{{tag}}_{{stamp}}.part.*"))
if not parts:
    raise SystemExit(f"Upload of {{dest_rel}} left no parts on the VM.")
dest = content / dest_rel
dest.parent.mkdir(parents=True, exist_ok=True)
tmp = dest.with_name(dest.name + ".uploading")
total = 0
with tmp.open("wb") as out:
    for part in parts:
        data = part.read_bytes()
        out.write(data)
        total += len(data)
if total != expected:
    tmp.unlink(missing_ok=True)
    raise SystemExit(
        f"Uploaded {{dest_rel}} is {{total}} bytes. This computer's file is {{expected}} bytes."
    )
tmp.replace(dest)
for part in parts:
    part.unlink()
print(f"VM file is this computer's file: {{dest}} ({{total}} bytes)", flush=True)
print("COLAB_STEP_OK", flush=True)
"""
Path(script_path).write_text(text, encoding="utf-8")
PY
    colab_exec 300 "$ASSEMBLE_FILE"
}

upload_local_zip() {
    echo "Photos come from this computer. Drive's zip is not read."
    upload_exact_file "$LOCAL_ZIP" "$DATA_ZIP" "zip"
}

upload_local_checkpoint() {
    local real name
    [[ "$RESUME" -eq 1 ]] || return 0
    if [[ ! -f "$MODEL_PATH" ]]; then
        echo "Cannot resume. This computer has no file at $MODEL_PATH" >&2
        echo "Drive is not searched, and no other checkpoint is substituted." >&2
        exit 1
    fi
    real="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$MODEL_PATH")"
    name="$(basename "$MODEL_PATH")"
    echo "Weights come from this computer. A Drive checkpoint is not substituted."
    echo "  selected: $MODEL_PATH"
    echo "  bytes:    $real"
    upload_exact_file "$real" "models/$name" "ckpt"
}

trap cleanup EXIT

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
    upload_local_zip
    mount_google_drive
    upload_local_checkpoint
    echo "Preparing photos from this computer's zip"
    colab_exec 3600 "$PREPARE_FILE"
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

if [[ "$USE_DRIVE" -eq 0 ]]; then
    upload_local_checkpoint
fi

echo "Installing Python packages (Colab already has CUDA PyTorch)"
colab install -s "$SESSION" -r requirements-colab.txt
echo "Restarting the kernel after pip so train.py is not talking to a dead websocket"
revive_colab_kernel

echo "Running train.py on $GPU"
colab_exec "$TIMEOUT" "$RUN_FILE"

mkdir -p models

download_checkpoint
download_if_present content/confusion_matrix.png confusion_matrix.png
download_if_present content/misclassified_validation.png misclassified_validation.png
download_if_present content/misclassified_validation.txt misclassified_validation.txt

PHASE=done
echo "Training finished"

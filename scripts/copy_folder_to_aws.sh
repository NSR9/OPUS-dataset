#!/usr/bin/env bash
set -euo pipefail

# --- Config ---
REMOTE_HOST=""
REMOTE_BASE="~/Projects"

# --- Usage ---
usage() {
  echo "Usage: $(basename "$0") -r <remote_host> <local_folder>"
  echo ""
  echo "  -r <remote_host>   Host alias from ~/.ssh/config (e.g. myserver)"
  echo "  <local_folder>     Path to the local folder you want to deploy"
  echo ""
  echo "Example: $(basename "$0") -r myserver ./my-app"
  exit 1
}

# --- Parse args ---
while getopts "r:" opt; do
  case $opt in
    r) REMOTE_HOST="$OPTARG" ;;
    *) usage ;;
  esac
done
shift $((OPTIND - 1))

[[ -z "$REMOTE_HOST" ]] && { echo "Error: remote host is required (-r)"; usage; }
[[ $# -lt 1 ]] && { echo "Error: local folder is required"; usage; }

LOCAL_FOLDER="${1%/}"  # strip trailing slash

[[ ! -d "$LOCAL_FOLDER" ]] && { echo "Error: '$LOCAL_FOLDER' is not a directory or doesn't exist"; exit 1; }

# Resolve to an absolute path so "." and relative paths always yield a real name
LOCAL_FOLDER="$(cd "$LOCAL_FOLDER" && pwd)"
PARENT_DIR="$(dirname "$LOCAL_FOLDER")"
FOLDER_NAME="$(basename "$LOCAL_FOLDER")"
ARCHIVE_NAME="${FOLDER_NAME}.zip"
TMP_ARCHIVE="/tmp/${ARCHIVE_NAME}"

# --- Steps ---
echo "==> [1/4] Creating $REMOTE_BASE on $REMOTE_HOST..."
ssh "$REMOTE_HOST" "mkdir -p $REMOTE_BASE"

echo "==> [2/4] Compressing '$LOCAL_FOLDER' -> $TMP_ARCHIVE..."
(cd "$PARENT_DIR" && zip -r "$TMP_ARCHIVE" "$FOLDER_NAME" -x "*.DS_Store" -x "*__MACOSX*" -x "*__pycache__*" -x "*.git*" -x "*.venv*")

echo "==> [3/4] Uploading archive to $REMOTE_HOST:$REMOTE_BASE/..."
scp "$TMP_ARCHIVE" "${REMOTE_HOST}:${REMOTE_BASE}/"

echo "==> [4/4] Extracting archive on remote into ${REMOTE_BASE}/..."
ssh "$REMOTE_HOST" "unzip -o ${REMOTE_BASE}/${ARCHIVE_NAME} -d ${REMOTE_BASE} && rm -rf ${REMOTE_BASE}/__MACOSX && rm ${REMOTE_BASE}/${ARCHIVE_NAME}"

echo "==> Cleaning up local archive..."
rm "$TMP_ARCHIVE"

echo ""
echo "Done! '$FOLDER_NAME' is live at ${REMOTE_HOST}:${REMOTE_BASE}/${FOLDER_NAME}/"

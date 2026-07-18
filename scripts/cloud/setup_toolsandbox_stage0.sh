#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="${TOOLSANDBOX_ENV_DIR:-/data/hxy/venvs/rescuecredit-toolsandbox}"
VENDOR_DIR="${TOOLSANDBOX_VENDOR_DIR:-/data/hxy/vendor/ToolSandbox}"
COMMIT=165848b9a78cead7ca7fe7c89c688b58e6501219
OUT="$REPO_ROOT/outputs/toolsandbox_stage0"

export PROMPT_COMMAND=
source "$(conda info --base)/etc/profile.d/conda.sh"
if [ ! -x "$ENV_DIR/bin/python" ]; then
  conda create -y -p "$ENV_DIR" python=3.9 pip
fi

mkdir -p "$(dirname "$VENDOR_DIR")" "$OUT"
if [ ! -d "$VENDOR_DIR/.git" ]; then
  git clone https://github.com/apple/ToolSandbox.git "$VENDOR_DIR"
fi
git -C "$VENDOR_DIR" fetch origin "$COMMIT"
git -C "$VENDOR_DIR" checkout --detach "$COMMIT"
test "$(git -C "$VENDOR_DIR" rev-parse HEAD)" = "$COMMIT"

"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install -e "$VENDOR_DIR"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$ENV_DIR/bin/python" "$REPO_ROOT/scripts/inspect_toolsandbox_contract.py" \
  --vendor-dir "$VENDOR_DIR" --limit 40 --seed 42 --output-dir "$OUT" \
  2>&1 | tee "$OUT/console.log"

echo TOOLSANDBOX_STAGE0_PASS
echo "ENV_DIR=$ENV_DIR"
echo "VENDOR_DIR=$VENDOR_DIR"

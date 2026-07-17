#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/data/hxy/projects/RescueCredit}
APPWORLD_ROOT=${APPWORLD_ROOT:-$PROJECT}
APPWORLD_VENV=${APPWORLD_VENV:-/data/hxy/venvs/rescuecredit-appworld}

cd "$PROJECT"
source data_disk_env.sh

# AppWorld officially documents Python 3.11. Keep it isolated from the
# existing RescueCredit Python 3.13 environment, while placing the entire
# environment on the data disk.
if [[ ! -x "$APPWORLD_VENV/bin/python" ]]; then
  mkdir -p "$(dirname "$APPWORLD_VENV")"
  if command -v python3.11 >/dev/null 2>&1; then
    python3.11 -m venv "$APPWORLD_VENV"
  elif command -v conda >/dev/null 2>&1; then
    conda create -y -p "$APPWORLD_VENV" python=3.11 pip
  else
    echo "Python 3.11 is required; neither python3.11 nor conda is available." >&2
    exit 2
  fi
fi
# Activate Conda environments through Conda so CONDA_PREFIX/IPython agree.
# A stdlib venv still uses its normal activation script.
if [[ -d "$APPWORLD_VENV/conda-meta" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$APPWORLD_VENV"
else
  source "$APPWORLD_VENV/bin/activate"
fi
hash -r
if [[ "$(python -c 'import sys; print(sys.prefix)')" != "$APPWORLD_VENV" ]]; then
  echo "Failed to select AppWorld Python at $APPWORLD_VENV" >&2
  exit 3
fi

export APPWORLD_ROOT
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$APPWORLD_ROOT" outputs/appworld_contract_probe

python -m pip install --upgrade pip wheel
python -m pip install -e ".[appworld,dev]"
python -m pip install --upgrade --force-reinstall "freezegun==1.5.1"

if command -v appworld >/dev/null 2>&1; then
  appworld install
  appworld download data
  appworld verify tests
  appworld verify tasks
else
  python -m appworld.cli install
  python -m appworld.cli download data
  python -m appworld.cli verify tests
  python -m appworld.cli verify tasks
fi

# Current AppWorld releases may materialize benchmark data relative to the
# working directory. Since this project already lives on /data, prefer it when
# the expected task tree is present there.
if [[ -d "$PROJECT/data/tasks" ]]; then
  APPWORLD_ROOT="$PROJECT"
  export APPWORLD_ROOT
fi

# Some AppWorld releases still materialize downloaded data/tests under the
# legacy $HOME/.appworld root even when APPWORLD_ROOT is exported. Relocate the
# files to the data disk, then retain the legacy path as a symlink.
LEGACY_APPWORLD_ROOT=${LEGACY_APPWORLD_ROOT:-$HOME/.appworld}
if [[ ! -d "$APPWORLD_ROOT/data/tasks" && -d "$LEGACY_APPWORLD_ROOT/data/tasks" ]]; then
  mkdir -p "$APPWORLD_ROOT"
  rsync -aH --remove-source-files "$LEGACY_APPWORLD_ROOT/" "$APPWORLD_ROOT/"
  find "$LEGACY_APPWORLD_ROOT" -depth -type d -empty -delete
  if [[ -d "$LEGACY_APPWORLD_ROOT" ]]; then
    echo "Legacy AppWorld root still contains files; refusing to replace it." >&2
    exit 4
  fi
  ln -s "$APPWORLD_ROOT" "$LEGACY_APPWORLD_ROOT"
fi

test -d "$APPWORLD_ROOT/data/tasks"

python scripts/inspect_appworld_contract.py \
  --appworld-root "$APPWORLD_ROOT" \
  --subset train \
  --limit 3 \
  --output-dir outputs/appworld_contract_probe

echo APPWORLD_STAGE0_FINISHED

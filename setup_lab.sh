#!/usr/bin/env bash
# One-command setup + launch for the BITS Virtual Lab (fresh or re-provisioned VM).
#
#   cd ~/workspace/Tilak/MMIR && bash setup_lab.sh
#
# Creates (or reuses) ~/workspace/mmir-venv that also sees the lab's preinstalled packages,
# installs only the missing ones, registers the Jupyter kernel and starts Jupyter.
# TRAIN_PER_CAT (default 10) = training clips per category for the learned temporal heads;
# 10 keeps a CPU-only run to roughly 45-60 min. Use TRAIN_PER_CAT=25 bash setup_lab.sh for more.
set -e
cd "$(dirname "$0")"
VENV="$HOME/workspace/mmir-venv"
LABPY=/opt/aiml-python/bin/python
PY=$( [ -x "$LABPY" ] && echo "$LABPY" || command -v python3 )

if [ ! -f "$VENV/bin/activate" ]; then
  echo ">> creating venv $VENV"
  "$PY" -m venv "$VENV"
  SP=$("$PY" -c "import site; print(site.getsitepackages()[0])")
  VSP=$("$VENV/bin/python" -c "import site; print(site.getsitepackages()[0])")
  echo "$SP" > "$VSP/aiml.pth"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo ">> lab packages visible in venv:"
python -c "import torch, numpy, cv2; print('   torch', torch.__version__, '| numpy', numpy.__version__, '| cuda', torch.cuda.is_available())"

if ! python -c "import transformers, streamlit, faiss, docx, altair" 2>/dev/null; then
  echo ">> installing missing packages"
  pip freeze | grep -i "^numpy==" > lab-constraints.txt || true
  pip install -q -r requirements-lab.txt -c lab-constraints.txt
fi

python -m ipykernel install --user --name mmir-venv --display-name "Python (mmir-venv)" >/dev/null
echo ">> kernel 'Python (mmir-venv)' registered"

export MVR_TRAIN_VIDEOS_PER_CATEGORY="${TRAIN_PER_CAT:-10}"
echo ">> training clips per category: $MVR_TRAIN_VIDEOS_PER_CATEGORY"
echo ">> starting Jupyter - open MIR_Assignment2.ipynb, kernel 'Python (mmir-venv)', Kernel -> Restart & Run All"
jupyter notebook

#!/usr/bin/env bash
# Devbox setup: a venv that can run everything here locally, plus the connectome data.
#
# The Flyte task image fetches the same three files at BUILD time (config._CONNECTOME_FETCH),
# pinned to the same two commits. This script is only for running outside a pod.
set -euo pipefail
cd "$(dirname "$0")"

BRAIN_COMMIT=91bdd1e7dcf193f3e7ca5a8933497fcef63b7960
ANNOT_COMMIT=8587524c1748ce5ef2080822a2fc890fc03bf597

uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

mkdir -p data
fetch () {  # url, destination
  [ -s "$2" ] || curl -sSLf -o "$2" "$1"
}
fetch "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/$BRAIN_COMMIT/Completeness_783.csv" data/completeness.csv
fetch "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/$BRAIN_COMMIT/Connectivity_783.parquet" data/connectivity.parquet
fetch "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/$ANNOT_COMMIT/supplemental_files/Supplemental_file1_neuron_annotations.tsv" data/annotations.tsv

echo
echo "done. 138,639 neurons in data/."
echo "  source .venv/bin/activate && export CONNECTOME_DIR=\$PWD/data MUJOCO_GL=egl"

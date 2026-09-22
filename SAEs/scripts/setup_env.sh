#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
base_python="${1:-/home/haoqian/Data/miniconda3/envs/latentchem_dev/bin/python}"
"$base_python" -m venv --system-site-packages "$project_root/.venv"
"$project_root/.venv/bin/python" -m pip install -r "$project_root/requirements.txt"
"$project_root/.venv/bin/python" -m pip check

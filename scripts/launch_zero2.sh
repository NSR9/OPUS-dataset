#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-production/config.prod.json}"
NUM_GPUS="${NUM_GPUS:-8}"

deepspeed --num_gpus "${NUM_GPUS}" production/train_prod.py --config "${CONFIG_PATH}"

#!/bin/bash

set -eo pipefail

python3 -m pip install -r requirements.txt --break-system-packages
python3 scripts/download_synth_shard.py -o ../synth_local_en
PYTHONPATH=. deepspeed --num_gpus 4 production/train_prod.py --config production/config.paper_baseline_random.json 2>&1 | tee baseline_run.log

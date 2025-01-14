#!/usr/bin/env bash

PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
    $(dirname "$0")/train.py --launcher pytorch ${@:4} --eval bbox

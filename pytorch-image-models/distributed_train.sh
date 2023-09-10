#!/bin/bash
source ~/.venv/bin/activate

# number of cuda of current node
NUM_PROC=$1
# number of nodes
NNODES=$2
# current node number
RANK=$3
USE_CPU=$4

shift
shift
shift
shift

if [ "$USE_CPU" == "cpu" ]; then
    export CUDA_VISIBLE_DEVICES=""
else
    export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5'
fi

torchrun --nproc_per_node=$NUM_PROC --nnodes=$NNODES --node_rank=$RANK \
--rdzv-id=400 --rdzv-backend=c10d --rdzv-endpoint=192.168.10.55:29600  train.py "$@"
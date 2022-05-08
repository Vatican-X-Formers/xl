#!/bin/bash

export OMP_NUM_THREADS=1

if [ -z "$C" ]
then
    C=./configs/small.yaml
fi

echo 'Finding free port'
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo 'Run training...'
python -m torch.distributed.launch --master_port=$PORT --nproc_per_node="$1" fine_tune.py --config_file "$C" $ARGS

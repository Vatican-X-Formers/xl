#!/bin/bash

export OMP_NUM_THREADS=1

# We specify the config through env variable
# The other argument for this script is the number of GPUs we use, it's passed as positional/vanilla argument

if [ -z "$C" ]
then
    C=./configs/small.yaml
fi

# This is a trick to run 2 distributed trainings on a single node. We need to specify the port for the processes to communicate, otherwise they will try to use the same one.
echo 'Finding free port'
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo 'Run training...'
python -m torch.distributed.launch --master_port=$PORT --nproc_per_node="$1" train.py --config_file "$C" $ARGS

#!/bin/bash

# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export OMP_NUM_THREADS=1

ARGS=""

if [ -n "$DEBUG" ]
then
    echo "DEBUG MODE"
    ARGS+="--batch_chunk=4"
    ARGS+=" --debug"
fi

echo 'Finding free port'
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo 'Run training...'
python -m torch.distributed.launch --master_port=$PORT --nproc_per_node="$2" train.py --config_file "$1" $ARGS

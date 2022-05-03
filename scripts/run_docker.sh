docker run --gpus all -it --rm --ipc=host \
    -v /pio/scratch/1/pn/hourglass:/workspace/hourglass \
    -v /pio/scratch/1/pn/checkpoints:/workspace/checkpoints \
    transformer-xl bash

docker run --gpus all -it --rm --ipc=host \
    -v /pio/scratch/1/pn/hourglass:/workspace/hourglass \
    -v /pio/scratch/1/pn/data:/pio/scratch/1/pn/data \
    -v /home/pn/.pdbrc.py:/root/.pdbrc.py \
    transformer-xl bash

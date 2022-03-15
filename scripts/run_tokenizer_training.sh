if [ -z "$SPLIT" ]
then
    SPLIT=text8
fi

echo $SPLIT

python tokenizer.py \
    --corpus_dir /home/pnawrot/piotrek/datasets/text8 \
    --corpus_split $SPLIT \
    --save_dir ./tokenizer_data \
    --tokenizer_type $1 \
    --vocab_size $2 \
    --dropout $3 \
    --pretokenization $4 \
    --algorithm $5


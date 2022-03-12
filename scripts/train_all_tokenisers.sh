for TOKENIZER in unigram wordpiece bpe do
		for VOCAB_SIZE in 5000 50000 do
				echo "Running $TOKENIZER $VOCAB_SIZE"
				python train_tokenizer.py \
						--corpus_dir /home/pnawrot/piotrek/datasets/text8 \
						--corpus_split text8 \
						--save_dir /home/pnawrot/piotrek/datasets/tokenizer/ \
						--tokenizer_type $TOKENIZER \
						--vocab_size $VOCAB_SIZE \
						--dropout 0.0 &
		done
done

echo "Running dropout training"
python train_tokenizer.py \
		--corpus_dir /home/pnawrot/piotrek/datasets/text8 \
		--corpus_split text8 \
		--save_dir /home/pnawrot/piotrek/datasets/tokenizer/ \
		--tokenizer_type bpe \
		--vocab_size 5000 \
		--dropout 0.1 &

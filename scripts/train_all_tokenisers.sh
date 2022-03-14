for TOKENIZER in unigram wordpiece bpe
do
		for VOCAB_SIZE in 5000 50000 
        do
				echo "Running $TOKENIZER $VOCAB_SIZE"
                bash scripts/run_tokenizer_training.sh $TOKENIZER $VOCAB_SIZE 0.0
		done
done

echo "Running dropout training"
bash scripts/run_tokenizer_training.sh bpe 5000 0.1

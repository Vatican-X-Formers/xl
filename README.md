# Hourglass with Dynamic Pooling

Requirements - ask me to do a `pip freeze`

## Files

- benchmark_code
	- Contains independent benchmarks of different parts of the code
- configs
	- By now I have only focused on `text8` dataset. Whenever I decide to switch to another dataset there will be another directory level that would refer to the dataset. I care for not having two files that share the common prefix.  For each dataset I keep 3 configs that refer to scale of the trained model. It could be either small, medium or large.
- utilts
	- Code copied from Nvidia's Transformer XL repo
	- It implements some functions for DDP, logging etc. 
	- I'd recommend to never touch this files ;p
- train.py
- eval.py
- vocabulary.py
	- It implements the most basic way to encode the text dataset from string to tensor of integers
- boundary_creator.py
	- It implements different ways to extract the boundaries from raw text, without processing anything by the model. It's used for topline experiments, it won't be used in the experiments with latent variable modelling the boundaries. 
- data_utils.py
	- It implements the data loader and dataset creation
- hourglass.py
	- It implements the Hourglass model, based on implementation of Transformer XL
- run_exp.sh
	- Uses 2 positional arguments, config file and the number of gpus. Model parameters, training hyper-parameters and the dataset is specified in the .yaml config file

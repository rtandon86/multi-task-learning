# Multi-Task Learning

About 
------------
MTL has proven effective in NLP for incorporating inductive bias. Linguistic hierarchies can be learned using the task hierarchy approach of MTL. The tasks for experiments are selected to enrich the representations linguistically. The tasks are POS tagging, chunking, dependency parsing, and sentence embeddings.

Data  
------------
Penn Treebank: Wall Street Journal Portion
CoNLL-2000
SWAG

Environment Set Up
------------
The main dependencies are:
AllenNLP
PyTorch

To ensure the code runs successfully please install all the packages mentioned in requirements.txt file.

Example Usage
------------
The implementation is based on the AllenNLP library. For an introduction to this library, you should check these tutorials.

An experiment is defined in a json configuration file (see configs/*.json for examples). The configuration file mainly describes the datasets to load, the model to create along with all the hyper-parameters of the model.

Once you have set up your configuration file (and defined custom classes such DatasetReaders if needed), you can simply launch a training with the following command and arguments:

python train.py --config_file_path unarylstm/train_pos_chunk_lm.json --serialization_dir my_training

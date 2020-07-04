# chatbot-nn-t5

Code to train T5 neural network for context-based chatbot.

## DEVELOPMENT STOPPED (4th of July, 2020)

Development is stopped as current version of T5 model and T5-related code 
in Pytorch/HuggingFace Transformers is unfinished yet and
has many problems (for instance
https://github.com/huggingface/transformers/issues/5142,
https://github.com/huggingface/transformers/issues/5349)
which make further development unreasonable at the moment because
of waste of expensive, limited computational resources in vain.

The last pretrained model ("t5-small" model fine-tuned on SQuAD 2.0 dataset):
https://huggingface.co/faust/broken_t5_squad2

pretrained model has a drawback as it generates a lot of "trash" symbols
at the end of a generated answer.
# chatbot-nn-t5

Code to train T5 neural network for context-based question-answering chatbot.
T5 being seq2seq model potentially allows not only to extract answers from text (as a specific substring), 
but also to rephrase or summarize answers from context sentences.

## Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Training](#training)
- [Trained model](#trained-model)

## Prerequisites

Recommended:
* Anaconda 3.x (https://www.anaconda.com/products/individual)

**Remark**: recommended installation method, which is described in this section below, is based on usage of Conda environment. If you don't want to use Anaconda you can install Python 3.6 and all required dependencies listed in [environment.yml](environment.yml) instead, but this approach is not covered in current version of installation instructions.

## Installation

1. Run the following commands to clone the repository and create Conda environment:
    ```bash
    git clone https://github.com/olegfaust/chatbot-nn-t5.git
    cd chatbot-nn-t5
    conda env create -f environment.yml
    ```

2. Run `conda activate chatbot-nn`. 
    * This activates the `chatbot-nn` environment
    * Do this each time you want to train or evaluate model
    * Run `conda deactivate` when you finished your current environment isn't needed anymore

## Training

To train model locally run train.py script.
For example: 

```bash
python3 train.py --input_dir=./data/ --model_name_or_path=t5-small --learning_rate=3e-5 --output_dir=./model/ --gpus=1
```

Use [train.ipynb](notebook/train.ipynb) to train model on Google Colab.

## Trained model

The last trained model ("t5-small" model fine-tuned on SQuAD 2.0 dataset) is available at: 
https://huggingface.co/faust/broken_t5_squad2

This model has a drawback as it generates a lot of "trash" symbols
at the end of a generated answer. Presumably such a situation happens because T5 model in
[HuggingFace Transformers](https://github.com/huggingface/transformers) is unfinished yet and
has many unsolved problems (for instance
https://github.com/huggingface/transformers/issues/5142,
https://github.com/huggingface/transformers/issues/5349) related to EOL symbols and tokenization in general. 
Anyway, if these "trash" symbols are filtered out, model answers are quite adequate. 

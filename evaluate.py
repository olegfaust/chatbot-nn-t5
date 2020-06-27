"""
Runs a model on a single node across multiple gpus.
"""
import os
import colorama
import glob
from train import add_general_args

from argparse import ArgumentParser
from termcolor import colored

from model import T5QaModel


def load_model(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = T5QaModel(hparams)

    # load model from checkpoint
    checkpoints = list(sorted(glob.glob(os.path.join(hparams.output_dir, "chatbot_model-epoch=*.ckpt"),
                                        recursive=True)))
    model = model.load_from_checkpoint(checkpoints[-1])

    return [model.model, model.tokenizer]


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    parent_parser = ArgumentParser(add_help=False)

    # add general arguments
    add_general_args(parent_parser)

    # each LightningModule defines arguments relevant to it
    parser = T5QaModel.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    # ---------------------
    # READ MODEL FROM THE LAST CHECKPOINT
    # ---------------------
    model, tokenizer = load_model(hyperparams)

    # ---------------------
    # RUN MODEL EVALUATION
    # ---------------------

    # On Windows, calling colorama.init() will filter ANSI escape sequences out of any text sent to stdout or stderr,
    # and replace them with equivalent Win32 calls.
    colorama.init()

    while True:
        # receive question
        print(colored('Q:>', "yellow"), end="")
        question = input().strip()
        # if there is no question -> continue
        if not question:
            continue
        # answer question
        input_ids = tokenizer.encode(question, return_tensors="pt")
        res = model.generate(input_ids)
        answer = [tokenizer.decode(x) for x in res]
        # print answer
        print(colored("A:>", "yellow"), colored(answer, "green"))

"""
Runs a model on a single node across multiple gpus.
"""
import os
from argparse import ArgumentParser

import numpy as np
import torch

import pytorch_lightning as pl
from model import T5QaModel

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)


def create_trainer(hparams) -> 'pl.Trainer':
    """
    Create trainer
    """
    # check if specified output directory is empty
    # (it must be empty to avoid possibility to rewrite trained model by mistake)
    if os.path.exists(hparams.output_dir) and os.listdir(hparams.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(hparams.output_dir))

    # prepare callback to save trained model after each epoch
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=hparams.output_dir, prefix="chatbot_model-",
                                                       save_top_k=-1)

    train_params = dict(
        max_epochs=hparams.epochs,
        gpus=hparams.gpus,
        precision=16 if hparams.use_16bit else 32,
        checkpoint_callback=checkpoint_callback,
    )

    if hparams.use_16bit:
        train_params["use_amp"] = hparams.use_16bit

    if hparams.gpus > 1:
        train_params["distributed_backend"] = hparams.distributed_backend

    trainer = pl.Trainer(**train_params)

    return trainer


def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = T5QaModel(hparams)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = create_trainer(hparams)

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


def add_general_args(arg_parser):
    # Data settings
    arg_parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=True,
        help="input data directory",
    )
    arg_parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="output directory where trained model and checkpoints will be saved",
    )

    # GPU settings
    arg_parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='how many gpus to be used'
    )
    # default distributed_backend is set to DDP as it is much faster:
    # as DDP only performs 1 transfer to sync gradients whereas
    # DP performs three GPU transfers for EVERY batch
    arg_parser.add_argument(
        '--distributed_backend',
        type=str,
        default='ddp',
        help='supports four options dp, ddp, ddp2, ddp_spawn'
    )
    arg_parser.add_argument(
        '--use_16bit',
        dest='use_16bit',
        action='store_true',
        help='if true uses 16 bit precision'
    )


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
    # RUN TRAINING
    # ---------------------
    main(hyperparams)

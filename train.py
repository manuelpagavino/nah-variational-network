import os
import logging
import hydra

import lightning as L

from pathlib import Path
from os.path import join

from solver import Solver

logger = logging.getLogger(__name__)

def run(args):
    """Setup trainer + solver and start training."""
    ###############
    ### Trainer ###
    ###############
    
    callbacks = []

    # set up checkpoints
    ckpt_dir_path = join(args.trainer.log_dir, args.trainer.checkpoint_dir)
    os.makedirs(ckpt_dir_path, exist_ok=True)
    ckpt_dir = os.listdir(ckpt_dir_path)
    if ckpt_dir:
        if args.trainer.continue_best:
            ckpt_file = "best.ckpt"
        else:
            ckpt_file = "last.ckpt"
        ckpt_file = [x for x in ckpt_dir if ckpt_file in x][-1]
        ckpt_file_path = join(ckpt_dir_path, ckpt_file)
        if args.trainer.checkpoint and Path(ckpt_file_path).exists():
            load_from = ckpt_file_path
        elif args.trainer.continue_from:
            load_from = args.trainer.continue_from
            args.trainer.keep_history = False
    else:
        args.trainer.checkpoint = False
    
    # add checkpoint callbacks
    callbacks += [
        # save checkpoint every n epochs as last.pt
        L.pytorch.callbacks.ModelCheckpoint(
            monitor="train_loss",
            dirpath=ckpt_dir_path,
            filename="last",
            save_last=True,
            save_top_k=0 if not args.trainer.checkpoint_every else -1, 
            every_n_epochs=args.trainer.checkpoint_every,
        ),
        # save best validation checkpoint as best.pt
        L.pytorch.callbacks.ModelCheckpoint(
            monitor="valid_loss",
            dirpath=ckpt_dir_path,
            filename="best",
            save_last=False,
        )
    ]

    # add callback to log learning rate every epoch
    callbacks += [
        L.pytorch.callbacks.LearningRateMonitor(
        logging_interval='epoch'
        )
    ]

    # setup tensorboard logger
    tb_logger = L.pytorch.loggers.TensorBoardLogger(
        save_dir='.',
        name="",
        version=join(args.trainer.log_dir, args.trainer.tensorboard_dir),
    )

    # setup trainer
    trainer = L.pytorch.Trainer(
        default_root_dir=os.path.join(os.getcwd(), ckpt_dir_path),
        accelerator=args.trainer.accelerator,
        strategy=args.trainer.strategy,
        devices=args.trainer.devices,
        precision=args.trainer.precision,
        callbacks=callbacks,
        logger=tb_logger,
        gradient_clip_val=args.trainer.optimizer.clip_norm if args.trainer.optimizer.clip else None,
        gradient_clip_algorithm="norm",
        check_val_every_n_epoch=args.trainer.valid_every,
        max_epochs=args.trainer.epochs,
    )

    ##############
    ### Solver ###
    ##############

    solver = Solver(args)
    trainer.fit(solver, ckpt_path=load_from if args.trainer.checkpoint else None)


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(args):
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)

    if args.trainer.verbose:
        logger.setLevel(logging.DEBUG)
    logger.info("Experiment: %s", os.getcwd())
    logger.debug(args)

    run(args)

if __name__ == "__main__":
    main()
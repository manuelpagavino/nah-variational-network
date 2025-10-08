import torch
import math 
import time
import json
import hydra 
import logging

import lightning as L

from torch.utils.data import DataLoader
from pathlib import Path
from autoclip.torch import QuantileClip

from methods.vn.vn import VariationalNetwork
from dataloader.dataset import NAHDataset
from utils.loss import CompositeLoss
from utils.misc import (
    worker_init_fn, 
    show_grad_norm, 
    pull_metric, 
    copy_state
    )

logger = logging.getLogger(__name__)
class Solver(L.LightningModule):
    """The training engine."""

    def __init__(self, args):
        super(Solver, self).__init__()
        self.args = args

        # init variational network 
        self.vn = VariationalNetwork(args.methods.vn)

        # init loss
        self.loss_fn = CompositeLoss(args.trainer.loss)

        # history
        self.history = []
        self.history_file = self.args.trainer.history_file

        # best checkpoint
        self.best_state = None
        self.best_file = Path(args.trainer.best_file)

        # hydra 
        self.hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

        self.save_hyperparameters()

    ##################
    ### Dataloders ###
    ##################

    def train_dataloader(self):
        train_dataset = NAHDataset(self.args, label="train")
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.args.trainer.batch_size,
            num_workers=self.args.dataloader.num_workers,  
            pin_memory=self.args.dataloader.pin_memory,
            prefetch_factor=self.args.dataloader.prefetch_factor,
            shuffle=False, # must be False for correct update of esm model in dataset
            worker_init_fn=worker_init_fn, # random seeding for multiple workers
            drop_last=True,
            persistent_workers=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_dataset = NAHDataset(self.args, label="valid")
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=self.args.trainer.batch_size,
            num_workers=self.args.dataloader.num_workers,  
            pin_memory=self.args.dataloader.pin_memory,
            prefetch_factor=self.args.dataloader.prefetch_factor,
            shuffle=False, # must be False for correct update of esm model in dataset
            drop_last=True,
            persistent_workers=True,
        )
        return valid_loader

    #################
    ### Optimizer ###
    #################

    def configure_optimizers(self):
        optim_dict = {}

        # fetch optimizer class from config
        optim_class = getattr(torch.optim, self.args.trainer.optimizer.klass)

        # init optimizer
        optimizer = optim_class(
            params=self.vn.parameters(),
            lr=self.args.trainer.optimizer.lr,
            betas=[self.args.trainer.optimizer.b1, self.args.trainer.optimizer.b2],
        )

        # init autoclip
        if self.args.trainer.optimizer.autoclip:
            optimizer = QuantileClip.as_optimizer(
                optimizer=optimizer, 
                quantile=self.args.trainer.optimizer.clip_quantile, 
                history_length=self.args.trainer.optimizer.clip_history_length, 
                )

        optim_dict |= {"optimizer": optimizer}

        # init scheduler
        if self.args.trainer.scheduler.use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode=self.args.trainer.scheduler.mode,
                factor=self.args.trainer.scheduler.factor,
                patience=self.args.trainer.scheduler.patience,
            )
            optim_dict |= {
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "train_loss"
                    }
                }

        return optim_dict

    def backward(self, loss):
        loss.backward()

        # display global L2-norm of gradients
        if self.args.trainer.optimizer.show_grad_norm:
            show_grad_norm(self.model, self.global_step)

    ###############
    ### Fitting ###
    ###############

    def on_fit_start(self):
        if self.history:
            logger.info("Showing metric history for checkpoint...")
        for epoch, metrics in enumerate(self.history):
            info = " ".join(f"{metric.capitalize()}={value:.5f}" for metric, value in metrics.items())
            logger.info(f"Epoch {epoch + 1}: {info}")

    def common_step(self, batch, batch_idx, valid=False):
        # extract relevant data from batch
        G, q, p, meta = batch.values()
        snr = meta["snr"].float()

        # apply variational network
        q_0 = torch.zeros_like(q)
        q_t_all = self.vn(p, G, q_0, snr, return_intermediate=True)
        
        # compute exponentially-weigthed loss function as in Eq. (32)
        with torch.autograd.set_detect_anomaly(True):
            loss = 0.
            kappa0 = 1e-3
            kappa = kappa0 * (self.global_step + 1)
            for t, q_t in enumerate(q_t_all):
                loss_weight = math.exp(-kappa * (self.args.methods.vn.T - t))
                if loss_weight >= 1e-8:
                    loss += self.loss_fn(estimate=q_t, target=q) * loss_weight

        # log loss in tensorboard
        self.log(
            "train_loss" if not valid else "valid_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # output dict
        outputs = {"loss": loss}

        return outputs

    ################
    ### Training ###
    ################

    def on_train_epoch_start(self):
        self.start = time.time()
        logger.info("-" * 70)
        logger.info("Training...")

    def training_step(self, batch, batch_idx):
        outputs = self.common_step(batch, batch_idx, valid=False)
        return outputs

    def on_train_epoch_end(self):
        train_loss = self.trainer.logged_metrics["train_loss"].item()
        valid_loss = self.trainer.logged_metrics["valid_loss"].item()

        logger.info(
            f"Training Summary | "
            f"Epoch: {self.current_epoch + 1} | "
            f"Time: {time.time() - self.start:.2f}s | "
            f"Train Loss: {train_loss:.5f}"
            )

        best_loss = min(pull_metric(self.history, 'valid') + [valid_loss])
        metrics = {'train': train_loss, 'valid': valid_loss, 'best': best_loss}

        if valid_loss == best_loss:
            # save the model with best valid loss as best state
            logger.info("New best valid loss %.4f", valid_loss)
            self.best_state = copy_state(self.state_dict())

        self.history.append(metrics)
        info = " | ".join(f"{metric.capitalize()} {value:.5f}" for metric, value in metrics.items())
        logger.info("-" * 70)
        logger.info(f"Overall Summary | Epoch {self.current_epoch + 1} | {info}")

        json.dump(self.history, open(self.history_file, "w"), indent=2)

    ##################
    ### Validation ###
    ##################

    def on_validation_epoch_start(self):
        self.start = time.time()
        logger.info("-" * 70)
        logger.info("Validation...")

    def validation_step(self, batch, batch_idx):
        outputs = self.common_step(batch, batch_idx, valid=True)
        return outputs

    def on_validation_epoch_end(self):
        valid_loss = self.trainer.logged_metrics["valid_loss"].item()
        logger.info(
            f"Validation Summary | "
            f"Epoch: {self.current_epoch + 1} | "
            f"Time: {time.time() - self.start:.2f}s | "
            f"Valid Loss: {valid_loss:.5f}"
            )

    ##################
    ### Checkpoint ###
    ##################

    def on_save_checkpoint(self, checkpoint):
        checkpoint["history"] = self.history

    def on_load_checkpoint(self, checkpoint):
        if self.args.trainer.keep_history:
            self.history = checkpoint["history"]
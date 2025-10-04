import torch
import random
import logging

import numpy as np

logger = logging.getLogger(__name__)

def set_nontorch_seeds(seed: int):
    """Set the random seed of all non-torch python packages that require a seed (numpy, random, etc.)."""
    # NOTE: numpy seed must be between 0 and 2**32 - 1, see
    # https://numpy.org/doc/1.14/reference/generated/numpy.random.RandomState.html
    max_seed = 4294967295  # equals 2**32 - 1
    seed = seed % max_seed
    np.random.seed(seed)
    random.seed(seed)

def worker_init_fn(worker_id):
    """Set new random seeds of non-torch libraries for each dataloader worker.

    See https://pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading
    """
    # each worker has a different torch seed (base_seed + worker_id) by default
    seed = torch.utils.data.get_worker_info().seed
    # manually reset seed for non-torch libraries for proper randomization across workers
    set_nontorch_seeds(seed)

def show_grad_norm(model, step):
    """Display global L2-norm of model gradients for step."""
    # get gradients of all trainable parameters
    gradients = [
        p.grad.detach() for p in model.parameters()
        if p.grad is not None and p.requires_grad
    ]
    # compute global L2-norm of gradient vector
    global_norm = 0.
    for grad in gradients:
        grad_norm = grad.data.norm(2)
        global_norm += grad_norm.item() ** 2
    global_norm = global_norm ** 0.5
    logger.info(f"Global L2-norm of gradients in step {step}: {global_norm}")

def pull_metric(history, name):
    """Retrieve metrics from history.json."""
    out = []
    for metrics in history:
        if name in metrics:
            out.append(metrics[name])
    return out

def copy_state(state):
    """Copy model state dict."""
    return {k: v.cpu().clone() for k, v in state.items()}
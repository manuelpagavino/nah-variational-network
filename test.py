import os
import logging
import hydra
import torch
import math

import numpy as np
import pandas as pd

from copy import deepcopy 
from os.path import (
    join, 
    abspath, 
    dirname
    )

from dataloader.esm import EquivalentSourceModel
from methods.benchmarks.ln import LeastNorm 
from methods.benchmarks.cs import CompressiveSensing
from methods.benchmarks.tgv import TotalGeneralizedVariation
from methods.benchmarks.pnn import PointNeuronNetwork
from methods.vn.vn import VariationalNetwork
from utils.plot import (
    plot_src, 
    plot_y_against_x, 
    plot_metric_evolution
    )
from utils.metrics import (
    rmse, 
    ncc, 
    ssim, 
    psnr
    )

logger = logging.getLogger(__name__)

def init_model(args, esm):
    """Initialize model for given method."""
    method = args.tester.method
    src = args.tester.src

    # calc regularization parameter according to Hald
    snr_lambdaa = args.methods[method].snr_lambdaa
    if snr_lambdaa == -1:
        snr_lambdaa = args.methods[method].optimized.snr_lambdaa[src]
    s_max = torch.linalg.norm(torch.from_numpy(esm.G), 2) # largest singular value of G
    p_norm = torch.linalg.norm(torch.from_numpy(esm.p).view(1,-1), 2, -1) # pressure norm
    lambdaa = s_max * 10 ** (-snr_lambdaa / 20) * p_norm / math.sqrt(esm.N) # regularization parameter

    # calc gradient step size 
    tau = 0.5 / (s_max ** 2) # gradient step size
    beta = 2 * s_max ** 2 # Lipschitz constant

    # init model for method
    if method == "ln":
        model = LeastNorm(
            T=args.methods.ln.T,
            method=args.methods.ln.method,
            lambdaa=lambdaa,
            tau=tau,
            rho=args.methods.ln.rho,
            eval_every=args.methods.ln.eval_every,
            )
    elif method == "cs":
        model = CompressiveSensing(
            T=args.methods.cs.T,
            lambdaa=lambdaa, 
            tau=tau,
            eval_every=args.methods.cs.eval_every,
            )
    elif method in ["tv", "tgv"]:
        if method == "tv":
            order = 1
        elif method == "tgv":
            order = 2
        model = TotalGeneralizedVariation(
            order=order, 
            T=args.methods[method].T, 
            lambdaa=lambdaa, 
            tau=tau, 
            rho=args.methods[method].rho, 
            mu=args.methods[method].mu, 
            beta=beta
            )
    elif method == "pnn":
        step_q = args.methods.pnn.step_q
        if step_q == -1:
            step_q = args.methods.pnn.optimized.step_q[src]
        step_x = args.methods.pnn.step_x
        if step_x == -1:
            step_x = args.methods.pnn.optimized.step_x[src]
        step_y = args.methods.pnn.step_y
        if step_y == -1:
            step_y = args.methods.pnn.optimized.step_y[src]
        psi = args.methods.pnn.psi
        if psi == -1:
            psi = args.methods.pnn.optimized.psi[src]
        model = PointNeuronNetwork(
            esm=esm, 
            T=args.methods.pnn.T, 
            step_q=step_q, 
            step_x=step_x, 
            step_y=step_y, 
            step_z=args.methods.pnn.step_z, 
            psi=psi, 
            lambdaa=lambdaa,
            init_ln=args.methods.pnn.init_ln,
            eval_every=args.methods.pnn.eval_every,
            stop_early=args.methods.pnn.stop_early,
            stop_tol=args.methods.pnn.stop_tol,
            )
    elif method == "vn":
        if args.tester.checkpoint_path:
            # init from checkpoint
            package = torch.load(args.tester.checkpoint_path, map_location="cpu", weights_only=False)
            config = package["hyper_parameters"]["args"]["methods"]["vn"]
            state_dict = package["state_dict"]
            state_dict = {k.replace("vn.", ""): v for k, v in state_dict.items()}
            model = VariationalNetwork(args=config)
            model.load_state_dict(state_dict)
        else:
            # init from scratch
            model = VariationalNetwork(args=args.methods.vn)
    
    return model, snr_lambdaa
        
def run(args):
    # extract params
    seed = args.tester.seed
    src = args.tester.src
    method = args.tester.method
    if isinstance(args.tester.freq, (int, float)):
        freqs = [args.tester.freq]
    else:
        freqs = np.arange(*args.tester.freq)
    dist = args.tester.dist
    dist_rec = args.tester.dist_rec
    snr = args.tester.snr
    ensemble = args.tester.ensemble

    if src == "lsp":
        # fixed experimental setup
        freqs = [1150]
        dist = 0.025
        snr = 0

    # make directory
    export_path = os.path.join(src, method)
    os.makedirs(export_path, exist_ok=True)

    # setup pandas MultiIndex DataFrame
    ################################################################
    #                       |   col1    |   col2    |   ....
    # sample1   |   row1    |
    #           |   row2    |
    #           |   ....    |
    # sample2   |   row1    |
    #           |   row2    |
    #           |   ....    |
    ################################################################
    cols = freqs # freqs as columns
    rows0 = np.arange(ensemble) # realizations in ensemble as rows / index 0
    rows1 = ["rmse", "ncc", "ssim", "psnr"] # metrics as rows / index 1
    midx = pd.MultiIndex.from_product([cols, rows0, rows1])
    data_q, data_p, data_v, data_I = [], [], [], [] # lists for data storage

    # init equivalent source model 
    esm = EquivalentSourceModel(args.dataloader.esm.test)

    for i in range(ensemble):
        for freq in freqs:
            logger.info(f"Processing sample {i} for reconstruction frequency of {freq}Hz ...")
            basename = f"src={src}_freq={freq}Hz_dist={dist}m_snr={snr}dB_sample={i}_method={method}"

            # update equivalent source model
            # reset frequency, distance, and snr according to user request
            updated_params = {"freq": float(freq), "dist": dist, "snr": snr}    
            if src == "lsp":
                # adapt esm geometry to match fixed experimental lsp and array setup
                updated_params |= {
                    "lx": 0.825, "ly": 0.6, "Nx": 50, "Ny": 40, "Mx": 12, "My": 9 
                    }
            esm.requested_params.update(updated_params) 
            esm.check_params()
            esm.setup_model(limit_to_nyquist=False)

            # load source patches for plotting
            src_patches = torch.load(join(dirname(abspath(__file__)), "data/src_patches.pt"), weights_only=False)
            q_ref_patches = src_patches[args.tester.src]

            # load source data
            src_data = torch.load(join(dirname(abspath(__file__)), "data/src_data.pt"), weights_only=False)
            if src == "lsp":
                # source data contains pressure measurements
                esm.p = src_data[args.tester.src].numpy()
                q_ref = np.zeros_like(esm.q)
            else:
                # src data contains ground truth source strength
                q_ref = src_data[args.tester.src]
                # radiate sound field for the given source
                esm.q = q_ref
                esm.radiate(seed=seed)
            
            # extract model quantities and add dummy batch and channel dim 
            G = torch.from_numpy(np.expand_dims(esm.G, (0,1))) # propagator
            q_ref = torch.from_numpy(np.expand_dims(esm.q, (0,1))) # equivalent source model
            p = torch.from_numpy(np.expand_dims(esm.p, (0,1))) # microphone array measurements
            meta = esm.current_params # metadata about current model setup
            
            # initialize model for given method
            model, snr_lambdaa = init_model(args, esm)

            # estimate source
            q0 = torch.zeros_like(q_ref)
            if method == "vn":
                with torch.no_grad():
                    q_all = model(p, G, q0, snr=snr_lambdaa, return_intermediate=True)
                q_pred = q_all[-1]

                metrics_all = []
                for t, q_t in enumerate(q_all):
                    hologram_layer_path = join(export_path, "hologram_layer")
                    os.makedirs(hologram_layer_path, exist_ok=True)

                    # plot intermediate VN layer outputs q^t
                    # cannot use same patch multiple times, hence deepcopy
                    title = fr"Source Strength $\mathbf{{q}}^{{{t}}}$ - " + method.upper()
                    plot_src(
                        meta, q_t, method.upper(), patches=deepcopy(q_ref_patches), title=title, 
                        filename=os.path.join(hologram_layer_path, f"hologram_{basename}_layer={t}.pdf")
                        )

                    diffmap_layer_path = join(export_path, "diffmap_layer")
                    os.makedirs(diffmap_layer_path, exist_ok=True)

                    # plot spatial difference map between VN layer outputs q^t and ground truth q_ref
                    diffmap = (q_t.abs() - q_ref.abs()) / q_ref.abs().max()
                    title = fr"Difference Map $\mathbf{{e}}^{{{t}}}$ of $\mathbf{{q}}^{{{t}}}$ vs. $\mathbf{{q}}_\mathrm{{true}}$"
                    plot_src(
                        meta, diffmap, method.upper(), patches=deepcopy(q_ref_patches), title=title, cmap='hot_r',
                        filename=os.path.join(diffmap_layer_path, f"diffmap_{basename}_layer={t}_vs_true.pdf")
                        )

                    # plot spatial difference map between successive VN layer outputs q^t vs. q^t-1
                    if t > 0:
                        diffmap = (q_t.abs() - q_all[t-1].abs()) / q_all[t-1].abs().max()
                        title = fr"Difference Map $\mathbf{{e}}^{{{t}}}$ of $\mathbf{{q}}^{{{t}}}$ vs. $\mathbf{{q}}^{{{t-1}}}$"
                        plot_src(
                            meta, diffmap, method.upper(), patches=deepcopy(q_ref_patches), title=title, cmap='hot_r',
                            filename=os.path.join(diffmap_layer_path, f"diffmap_{basename}_layer={t}_vs_layer={t-1}.pdf")
                            )

                    # store metrics
                    metrics_t = {
                        "rmse": rmse(q_t, q_ref),
                        "ncc": ncc(q_t, q_ref),
                        "ssim": ssim(q_t, q_ref),
                        "psnr": psnr(q_t, q_ref),
                    }
                    metrics_all += [metrics_t]

                metric_vs_layer_path = join(export_path, "metric_vs_layer")
                os.makedirs(metric_vs_layer_path, exist_ok=True)

                # plot evolution of metrics across VN layers
                plot_metric_evolution(
                    metrics_all, method.upper(), 
                    filename=os.path.join(metric_vs_layer_path, f"metrics_iterative_evolution_{basename}.pdf")
                    )
            else:
                q_pred = model(p, G, q0, q_ref=q_ref)
            
            # compute metrics
            metrics = {
                "rmse": rmse(q_pred, q_ref),
                "ncc": ncc(q_pred, q_ref),
                "ssim": ssim(q_pred, q_ref),
                "psnr": psnr(q_pred, q_ref),
            }

            # store metrics for current source estimate
            data_q += [metrics[key] for key in rows1]

            hologram_path = join(export_path, "hologram")
            os.makedirs(hologram_path, exist_ok=True)

            # plot ground truth and save figure
            if src != "lsp":
                plot_src(
                    meta, q_ref, "True", 
                    filename=os.path.join(hologram_path, '_'.join(["ground_truth", f"src={src}"]) + ".pdf")
                    )

            # plot estimate and save figure
            plot_src(
                meta, q_pred, method.upper(), metrics=metrics if src != "lsp" else {}, patches=q_ref_patches, 
                filename=os.path.join(hologram_path, f"hologram_{basename}.pdf")
                )

            # estimate sound field at reconstruction distance
            if dist_rec and src != "lsp":
                # reset model for requested reconstruction distance
                esm.requested_params.update({"dist": dist_rec}) 
                esm.check_params()
                esm.setup_model(limit_to_nyquist=False)
                
                # define fully-determined (M==N) propagator and its normal derivative
                esm.setup_propagator(fully_determined=True)
                esm.setup_propagator_derivatives(fully_determined=True)
                G = torch.from_numpy(np.expand_dims(esm.G, (0,1))) 
                dGdz = torch.from_numpy(np.expand_dims(esm.dGdz.astype(np.complex64), (0,1)))

                # true sound field quantities radiated from source ground truth
                p_ref = G @ q_ref.view(*q_ref.shape[:2],-1,1)
                v_ref = -dGdz @ q_ref.view(*q_ref.shape[:2],-1,1)
                I_ref = 0.5 * (p_ref * v_ref.conj()).real
                I_ref = I_ref.type(dtype=torch.cfloat) # metric computation expects complex dtype despite zero imag

                # reconstructed sound field quantities radiated from source estimate
                p_pred = G @ q_pred.view(*q_pred.shape[:2],-1,1) # pressure
                v_pred = -dGdz @ q_pred.view(*q_pred.shape[:2],-1,1) # velocity
                I_pred = 0.5 * (p_pred * v_pred.conj()).real # intensity
                I_pred = I_pred.type(dtype=torch.cfloat) # metric computation expects complex dtype despite zero imag

                for (x_ref, x_pred, data_x) in [
                    (p_ref, p_pred, data_p), 
                    (v_ref, v_pred, data_v), 
                    (I_ref, I_pred, data_I)
                    ]:
                    # rasterize fully-determined sound field quantities
                    x_ref = x_ref.view(q_ref.shape)
                    x_pred = x_pred.view(q_ref.shape)

                    # compute metrics
                    metrics = {
                        "rmse": rmse(x_pred, x_ref),
                        "ncc": ncc(x_pred, x_ref),
                        "ssim": ssim(x_pred, x_ref),
                        "psnr": psnr(x_pred, x_ref),
                    }

                    # store metrics for current sound field estimate
                    data_x += [metrics[key] for key in rows1]
        
    # plot metrics against frequency
    if src != "lsp" and len(freqs) > 1:
        data_list = [data_q]
        quantity_list = ["source_strength"]
        if dist_rec:
            data_list += [data_p, data_v, data_I]
            quantity_list += ["field_pressure", "field_velocity", "field_intensity"]

        metric_vs_freq_path = join(export_path, "metric_vs_freq")
        os.makedirs(metric_vs_freq_path, exist_ok=True)

        for data, quantity in zip(data_list, quantity_list):
            # convert to pandas dataframe of format
            df = pd.DataFrame(np.array(data), index=midx)
            df = df.unstack(level=0) # correct shape

            # write to csv
            df.to_csv(os.path.join(metric_vs_freq_path, f"metrics_quantity={quantity}_{basename}.csv"), encoding='utf-8')

            # debug: read csv 
            # df_from_csv = pd.read_csv(filename, index_col=[0,1], header=[1])

            # plot metric against frequency        
            for metric in metrics.keys():
                x = freqs
                y = df.loc[(np.arange(ensemble), metric), :] # requested metric for each sample in ensemble
                y = np.array(y.mean(0)) # calculate ensemble-average over all realizations

                # define plot title and axes labels
                titles = {
                    "source_strength": r"\textbf{{Error - Source Strength} $\mathbf{q}$", 
                    "field_pressure": r"\textbf{{Error - Pressure} $\mathbf{p}$", 
                    "field_velocity": r"\textbf{{Error - Velocity} $\mathbf{v}$", 
                    "field_intensity": r"\textbf{{Error - Intensity} $\mathbf{I}$",
                }
                ylabels = {
                    "rmse": r"$\epsilon_{\mathrm{RMSE}}$ [\%]",
                    "ncc": r"$\epsilon_{\mathrm{NCC}}$ [\%]",
                    "psnr": r"$\epsilon_{\mathrm{PSNR}}$ [dB]",
                    "ssim": r"$\epsilon_{\mathrm{SSIM}}$ [\%]",
                }
                xlabel = "Frequency [Hz]"

                # plot and save figure
                plot_y_against_x(
                    y=y, x=x, label=method.upper(), title=titles[quantity], ylabel=ylabels[metric], xlabel=xlabel, 
                    yrange=None if metric == "psnr" else [0, 100], axvline=esm.nyquist_freq, axvlabel="Sampling Limit",
                    filename=os.path.join(metric_vs_freq_path, f"{metric}-vs-freq_quantity={quantity}_dist_rec={str(dist_rec)}m_{basename}.pdf")
                    )
    
@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(args):
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)

    if args.trainer.verbose:
        logger.setLevel(logging.DEBUG)
    logger.info("Experiment: %s", os.getcwd())
    logger.debug(args)
    
    # check for invalid param requests
    assert args.tester.src in ["plate", "piston", "quadrupole", "lsp"]
    assert args.tester.method in ["ln", "cs", "tv", "tgv", "pnn", "vn"]

    run(args)

if __name__ == "__main__":
    main()
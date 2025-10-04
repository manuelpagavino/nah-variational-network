# Data-driven and physics-constrained acoustic holography based on optimizer unrolling

This repository holds supplementary material to the article:

M. Pagavino and F. Zotter, [Data-driven and physics-constrained acoustic holography based on optimizer unrolling](https://doi.org/10.1051/aacus/2025030), Acta Acustica, 2025.

When using this code, please refer to:

```
@article{pagavino_zotter_2025,
	author = {{Pagavino, Manuel} and {Zotter, Franz}},
	title = {Data-driven and physics-constrained acoustic holography based on optimizer unrolling},
	doi = "10.1051/aacus/2025030",
	url = "https://doi.org/10.1051/aacus/2025030",
	journal = {Acta Acustica},
	year = 2025,
	volume = 9,
}
```

# Requirements

To use this code, please install the required dependencies listed in `requirements.yaml`. If you are using the `conda` package manager, you can simply run `conda env create -f requirements.yaml` to install all dependencies in a new environment named `nah-variational-network`. The code was tested with `python=3.13.7` and `pytorch=2.8.0`.

# Training

To retrain the Variational Network, simply run the `train.py` script and adapt the `conf/trainer/trainer.yaml` config according to your needs. Here you can specify settings related to the dataloader (batch size, epochs), optimizer (type, learning rate, gradient clipping), learning rate scheduler, and other training-related settings, such as device, numerical precision, etc. Once you have settled with a suitable configuration, you can start training with:

```
python train.py name=foobar -cn=config
```

**NOTE**: The current release of the code does not yet include the on-the-fly equivalent source generator for online dataset creation. Instead, random dummy data is created as placeholder during training until the online source generator has been made public. 
However, a dataset containing a fixed number of samples drawn from the online source generator is likely to be released in the near future and will replace the dummy data generator. You can always patch in your own dataset or source generator by modifying the code in `dataloader/dataset.py`.

# Testing - Reproduction of Results

To reproduce the results of our article simply run the  `test.py` script and adapt the `conf/tester/tester.yaml` config according to your needs. 

The config flags in `conf/tester/tester.yaml` determine the ...

- `src`: Sound source type
	- `plate`: center-driven plate 
	- `piston`: baffled circular piston  
	- `quadrupole`: quadrupole
	- `lsp`: loudspeaker layout
- `method`: Reconstruction method
	- `ln`: Least Norm (LN)
	- `cs`: Compressive Sensing (CS)
	- `tv`: Total Variation (TV)
	- `tgv`: Total Generalized Variation (TGV)
	- `pnn`: Point Neuron Network (PNN)
	- `vn`: Variational Network (VN - the proposed method)
- `freq`: Sound frequency in $\mathrm{Hz}$. Your can either provide a single frequency or a list `[start, stop, step]` defining a range of frequencies over which to iterate.
- `snr`: Signal-to-Noise Ratio (SNR) of simulated measurements in $\mathrm{dB}$. Default: $28 \thinspace \mathrm{dB}$
- `dist`: Measurement distance in $\mathrm{m}$. Default: $6 \thinspace \mathrm{cm}$ 
- `dist_rec`: Reconstruction distance in $\mathrm{m}$ to reconstruct the sound field of the estimated equivalent source distribution. Default: $3 \thinspace \mathrm{cm}$ 
- `seed`: Use a specific fixed random seed for a fair comparison of methods. Default: $1234$ 
- `ensemble`: Number of random realizations. If you want the ensemble-averaged metrics over multiple random realizations, you need to set `ensemble`$>1$ and use a random seed via `seed=null`. Default: $1$
- `checkpoint_path`: The absolute path to a valid checkpoint containing the model weights of the Variational Network.

Defaults are chosen to match the settings used in our experiments.

You can also modify method-specific parameters in the associated `conf/methods` configs. These include parameters like ...

- `T`: Number of iterations. Default: $20000$
- `snr_lambdaa:` Estimated SNR for computation of the regularization parameter according to Hald's formula in Eq. (25).  Default: $-1$ (optimized setting).

However, it is advised to run the script with default parameters, which correspond to the optimized settings as used in the article.

Once you run the script, the associated figures are exported to the `results` folder, which is organized in the following hierarchical structure:

```bash
results
├─- name # experiment name
    ├── src # source type: plate, piston, quadrupole, lsp
        ├── method # reconstruction method: ln, cs, tv, tgv, pnn, vn
	        ├── hologram # source holograms of Figs. 3-5,7,12
	        ├── hologram_layer # intermediate source holograms of Fig. 9
	        ├── diffmap_layer # spatial difference maps of Fig. 10
	        ├── metric_vs_freq # metrics vs. frequency plots of Fig. 6
			├── metric_vs_layer # metrics vs. layer plots of Fig. 8
```

Here, the folders `hologram_layer`, `diffmap_layer`, and `metric_vs_layer` are only relevant to the `vn`, while `hologram` and `metric_vs_freq` are available to all methods.

The figure filenames hold information about the chosen experimental settings and follow the convention:

```
{folder_name}_src={src}_freq={freq}Hz_dist={dist}m_snr={snr}dB_sample={i}_method={method}.pdf
```

## tl;dr: Reproduction of Results 

To reproduce the results of our article, simply run the following commands and choose your preferred reconstruction methods from:

- `tester.method=ln`: Least Norm (LN)
- `tester.method=cs`: Compressive Sensing (CS)
- `tester.method=tv`: Total Variation (TV)
- `tester.method=tgv`: Total Generalized Variation (TGV)
- `tester.method=pnn`: Point Neuron Network (PNN)
- `tester.method=vn`: Variational Network (VN - the proposed method)

**Section 6.1 Center-driven plate**:

Figures 3,4:
```
python test.py name=foobar tester.src=plate tester.freq=800 tester.snr=28 tester.dist=0.06 tester.method=method_of_choice -cn=config
```

**Section 6.2 Baﬄed piston**:

Figures 3,5:
```
python test.py name=foobar tester.src=piston tester.freq=1000 tester.snr=28 tester.dist=0.06 tester.method=method_of_choice -cn=config
```

Figure 6:
```
python test.py name=foobar tester.src=piston tester.freq="[100, 5100, 100]" tester.snr=28 tester.dist=0.06 tester.dist_rec=0.03 tester.method=method_of_choice -cn=config
```

Optionally use `tester.ensemble=10` and `tester.seed=null` to get the ensemble-averaged metrics over 10 different random realizations.

**Section 6.3 Quadrupole**:

Figures 3,7:
```
python test.py name=foobar tester.src=quadrupole tester.freq=3000 tester.snr=28 tester.dist=0.06 tester.method=method_of_choice -cn=config
```

To introduce additional spatial sparsity to the TV and TGV method as used in the article, just add `methods.tv.mu=1` and `methods.tgv.mu=1` to the above command.

**Section 7.2 Reconstructed loudspeaker layout**:

Figures 12:
```
python test.py name=foobar tester.src=lsp tester.freq=1150 tester.dist=0.025 tester.snr=0 methods.pnn.T=10000 tester.method=method_of_choice -cn=config
```

**Section 6.4 Iterative evolution (piston):**

Figures 8-10:
```
python test.py name=foobar tester.src=piston tester.freq=1000 tester.snr=28 tester.dist=0.06 tester.method=vn -cn=config
```

Only relevant to the Variational Network `tester.method=vn`.

# Checkpoint / Model Weights

A checkpoint containing the model weights of a pre-trained Variational Network will be released soon.

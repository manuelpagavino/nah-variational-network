import random

import numpy as np
class EquivalentSourceModel:
    """The Equivalent Source Method (ESM) according to Section 2."""

    def __init__(self, kwargs):
        super(EquivalentSourceModel, self).__init__()
        # check requested parameter ranges
        self.requested_params = kwargs
        self.check_params()

        # set current parameters
        self.current_params = {}
        self.setup_model()

    def check_params(self):
        """Check requested parameters for critical values."""
        for name, params in self.requested_params.items():

            # turn single value parameter into range
            if not isinstance(params, list) and isinstance(params, (int, float)):
                params = [params] * 2
                self.requested_params[name] = params

            # turn single value list into range
            elif len(params) == 1 and isinstance(params[0], (int, float)):
                params = [params[0]] * 2
                self.requested_params[name] = params

            # check if requested range is valid
            elif len(params) > 2:
                raise ValueError(
                    f"Invalid parameter request! \n"
                    f"Requested range {params} for parameter {name} does not define a valid range of form "
                    f"[lower_bound, upper_bound]!"
                )
            
            # check if all values in range have the same type
            assert all(type(param) == type(params[0]) for param in params)

    def setup_model(self, requested_params=None, limit_to_nyquist=True, fully_determined=False):
        """Randomly select current parameters from requested parameter ranges."""
        # randomly reset all parameters
        if not requested_params:
            requested_params = self.requested_params

        # randomize parameters within requested ranges according to their dtype
        for key, value in requested_params.items():
            if isinstance(value[0], float):
                setattr(self, key, random.uniform(*value))
            elif isinstance(value[0], int):
                setattr(self, key, random.randint(*value))
            elif isinstance(value[0], str):
                setattr(self, key, random.choice(value))

            # set current parameter as class attribute
            self.current_params[key] = getattr(self, key)

        # setup model geometry
        self.setup_geometry(axisymmetric=False)

        # reset frequency
        self.setup_freq(limit_to_nyquist=limit_to_nyquist)

        # setup propagation matrix (Green's functions)
        self.setup_propagator(fully_determined=fully_determined)

    def setup_geometry(self, axisymmetric=False):
        """Set up geometry of equivalent sources and measurement array."""
        # equivalent source setup
        self.N = self.Nx * self.Ny
        if axisymmetric:
            self.x_src = np.linspace(-0.5 * self.lx, 0.5 * self.lx, self.Nx)
            self.y_src = np.linspace(-0.5 * self.ly, 0.5 * self.ly, self.Ny)
            self.X_src, self.Y_src = np.meshgrid(self.x_src, self.y_src)
        else:
            self.x_src = np.linspace(0., self.lx, self.Nx)
            self.y_src = np.linspace(0., self.ly, self.Ny)
            self.X_src, self.Y_src = np.meshgrid(self.x_src, self.y_src)
        self.R_src = np.sqrt(self.X_src ** 2 + self.Y_src ** 2)
        self.q = np.zeros((self.Ny, self.Nx), dtype=np.complex64)

        # microphone array setup
        self.M = self.Mx * self.My
        if axisymmetric:
            self.x_mic = np.linspace(-0.5 * self.lx, 0.5 * self.lx, self.Mx)
            self.y_mic = np.linspace(-0.5 * self.ly, 0.5 * self.ly, self.My)
            self.X_mic, self.Y_mic = np.meshgrid(self.x_mic, self.y_mic)
        else:
            self.x_mic = np.linspace(0., self.lx, self.Mx)
            self.y_mic = np.linspace(0., self.ly, self.My)
            self.X_mic, self.Y_mic = np.meshgrid(self.x_mic, self.y_mic)
        self.R_mic = np.sqrt(self.X_mic ** 2 + self.Y_mic ** 2)
        self.p = np.zeros((self.My, self.Mx), dtype=np.complex64)

    def setup_freq(self, limit_to_nyquist=False, sample_log=False):
        """Randomly sample frequency and optionally enforce Nyquist limit."""
        # determine "Nyquist" sampling frequency 
        dx_mic = self.x_mic[1] - self.x_mic[0]
        dy_mic = self.y_mic[1] - self.y_mic[0]
        self.nyquist_freq = self.c / (2 * np.min([dx_mic, dy_mic]))

        # set lower and upper bound
        freq_min = self.requested_params["freq"][0]
        if limit_to_nyquist:
            freq_max = np.minimum(self.requested_params["freq"][1], self.nyquist_freq) # upper bound = aliasing limit
        else:
            freq_max = self.requested_params["freq"][1] 

        # randomly set frequency within lower and upper bound
        if sample_log:
            # logarithmic sampling favors lower frequencies
            self.freq = np.exp(np.random.uniform(*np.log([freq_min, freq_max]))) 
        else:
            # uniform sampling
            self.freq = np.random.uniform(freq_min, freq_max) 
        self.current_params["freq"] = self.freq

        # reset frequency-related parameters
        self.omega = 2 * np.pi * self.freq # angular frequency
        self.k = self.omega / self.c # wave number

    def setup_propagator(self, fully_determined=False):
        """Set up Green's function propagation matrix for the used geometry."""
        # vectorize and tile
        if fully_determined: # propagator is a fully-determined square matrix (M == N)
            # source surface
            x_src = np.tile(self.X_src.reshape(1, -1), (self.N, 1))
            y_src = np.tile(self.Y_src.reshape(1, -1), (self.N, 1))
            # measurement surface
            x_mic = np.tile(self.X_src.reshape(-1, 1), (1, self.N))
            y_mic = np.tile(self.Y_src.reshape(-1, 1), (1, self.N))

        else: # propagator is an under-determined rectangular matrix (M < N)
            # source surface
            x_src = np.tile(self.X_src.reshape(1, -1), (self.M, 1))
            y_src = np.tile(self.Y_src.reshape(1, -1), (self.M, 1))
            # measurement surface
            x_mic = np.tile(self.X_mic.reshape(-1, 1), (1, self.N))
            y_mic = np.tile(self.Y_mic.reshape(-1, 1), (1, self.N))

        # distances between equivalent sources and microphones
        self.R = np.sqrt((x_mic - x_src) ** 2 + (y_mic - y_src) ** 2 + self.dist ** 2)

        # setup propagation matrix of Green's function
        self.G = np.exp(-1j * self.k * self.R) / (4 * np.pi * self.R)
        self.G *= 1j * self.omega * self.rho
        self.G = self.G.astype(np.complex64)

    def setup_propagator_derivatives(self, fully_determined=False):
        """Compute first- and second-order derivatives of propagation matrix."""
        G = self.G
        self.setup_propagator(fully_determined)

        # first-order spatial derivatives
        self.dGdx = - (1j * self.k + 1 / self.R) * self.X_src.flatten() / self.R * self.G
        self.dGdy = - (1j * self.k + 1 / self.R) * self.Y_src.flatten() / self.R * self.G
        self.dGdz = - self.dist * np.exp(-1j * self.k * self.R) / (4 * np.pi * self.R ** 3) * \
                    (1j * self.k * self.R + 1)

        # second-order spatial derivatives
        self.ddGdxdx = 1 / self.R ** 2 * self.X_src.flatten() ** 2 / self.R ** 2 * self.G - \
                       (1j * self.k + 1 / self.R) * (self.R ** 2 - self.X_src.flatten() ** 2) / \
                       self.R ** 3 * self.G - (1j * self.k + 1 / self.R) ** 2 * self.X_src.flatten() ** 2 / \
                       self.R ** 2 * self.G

        self.ddGdydy = 1 / self.R ** 2 * self.Y_src.flatten() ** 2 / self.R ** 2 * self.G - \
                       (1j * self.k + 1 / self.R) * (self.R ** 2 - self.Y_src.flatten() ** 2) / \
                       self.R ** 3 * self.G - (1j * self.k + 1 / self.R) ** 2 * self.Y_src.flatten() ** 2 \
                       / self.R ** 2 * self.G

        self.ddGdzdz = self.k ** 2 * self.G - (self.R ** 2 - self.dist ** 2) / self.R ** 4 * self.G - \
                       (1j * self.k + 1 / self.R) * self.dist ** 2 / self.R ** 3 * self.G - \
                       (1j * self.k + 1 / self.R) ** 2 * (self.X_src.flatten() ** 2 + self.Y_src.flatten() ** 2) / \
                       self.R ** 2 * self.G

        self.G = G

    def vectorize(self):
        """Vectorize source and measurement matrix."""
        self.q = self.q.reshape(self.N, 1)
        self.p = self.p.reshape(self.M, 1)

    def rasterize(self):
        """Rasterize source and measurement vector."""
        self.q = self.q.reshape(self.Ny, self.Nx)
        self.p = self.p.reshape(self.My, self.Mx)

    def radiate(self, seed=None):
        """Compute sound field of the current source setup."""
        self.vectorize()
        self.p = self.G.dot(self.q)
        self.add_noise(seed=seed)
        self.rasterize()

    def normalize_src(self):
        """Normalize current source."""
        self.q /= np.max(np.abs(self.q))

    def add_noise(self, seed=None):
        """Add noise to measurements with given signal-to-noise ratio (SNR)."""
        # compute noise variance for given SNR
        snr_linear = 10 ** (self.snr / 10)
        p_var = np.var(self.p)
        n_var = p_var / snr_linear

        # use fixed seed 
        if seed:
            np.random.seed(seed)

        # compute complex-valued Gaussian noise
        n = np.random.normal(
            loc=0,
            scale=np.sqrt(2)/2,
            size=(self.p.shape + (2,))
        ).view(np.complex128)[...,0]
        n = n.astype(np.complex64)

        # add noise with given SNR
        self.p += np.sqrt(n_var) * n


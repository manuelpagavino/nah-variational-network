import numpy as np

from torch.utils.data import Dataset

from dataloader.esm import EquivalentSourceModel
class NAHDataset(Dataset):
    """Dataset for Nearfield Acoustic Holgraphy (NAH)."""

    def __init__(self, args, label="training", **kwargs):
        super(NAHDataset, self).__init__(**kwargs)
        self.args = args
        self.label = label
        self.samples_per_epoch = self.args.dataloader[self.label].samples_per_epoch

        # init equivalent source model 
        self.esm = EquivalentSourceModel(self.args.dataloader.esm[self.label])

    def __len__(self):
        """Get Nr. of files per epoch."""
        return self.samples_per_epoch

    def __getitem__(self, index):
        """Get dict containing propagator G, equivalent source q, measurements p, and metadata for random model setup."""
        #np.random.seed(0) # debug: use same fixed seed for each batch

        # reset the equivalent source model
        if not index % self.args.trainer.batch_size:
            # reset entire model for each batch
            self.esm.setup_model(limit_to_nyquist=False)
        else:
            # only reset frequency, distance, and snr within a batch, while keeping model geometry
            requested_params_reduced = {
                key: self.args.dataloader.esm[self.label][key] 
                for key in ["freq", "dist", "snr"]
                }
            self.esm.setup_model(limit_to_nyquist=False, requested_params=requested_params_reduced)

        # create new source
        # NOTE: create random dummy data as placeholder until the 
        #       on-the-fly equivalent source generator has been made public
        dummy_source = np.random.randn(self.esm.Ny, self.esm.Nx, 2).view(np.complex128)
        self.esm.q = dummy_source.astype(np.complex64)

        # radiate sound field given the new source
        self.esm.radiate()
        
        # create output dict and add missing channel dims
        outputs = {
            "G": np.expand_dims(self.esm.G, 0), # propagator
            "q": np.expand_dims(self.esm.q, 0), # equivalent source model
            "p": np.expand_dims(self.esm.p, 0), # microphone array measurements
            "meta": self.esm.current_params, # metadata about current model setup
        }

        return outputs



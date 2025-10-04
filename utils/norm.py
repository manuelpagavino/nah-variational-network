# NOTE:
#############################################################################################
### This implementation of complex-valued instance normalization is based on the          
### code of complex-valued batch normalization of the cplxmodule package                  
### https://github.com/ivannz/cplxmodule/blob/master/cplxmodule/nn/modules/batchnorm.py   
### that was proposed in "Deep Complex Networks - Trabelsi et al., 2017".                 
### We adapted to code to our needs and also added an instance de-normalization           
### to reverse all operations for experimentation.                                        
#############################################################################################

import torch

from torch.nn import init
from cplxmodule import cplx
from cplxmodule.nn import CplxToCplx

def init_instance_norm(num_features):
    """Initialize fixed instance norm with rescaled weight."""
    # instance normalization
    norm = CplxInstanceNorm2d(
        num_features, 
        track_running_stats=False, 
        affine=True
        )
    # rescale weight and disable learning
    norm.weight = torch.nn.Parameter(
        0.1 * norm.weight, requires_grad=False
        )
    norm.bias.requires_grad = False

    return norm
class _CplxInstanceNorm(CplxToCplx):
    """The base class for complex-valued instance normalization layer. Taken from `torch.nn.modules.InstanceNorm` verbatim."""
    def __init__(
            self,
            num_features,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            training=True
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.training = training
        if self.affine:
            self.weight = torch.nn.Parameter(torch.empty(2, 2, 1, num_features)) # add batch dim
            self.bias = torch.nn.Parameter(torch.empty(2, 1, num_features)) # add batch dim
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)


        if self.track_running_stats:
            self.register_buffer('running_mean', torch.empty(2, 1, num_features))
            self.register_buffer('running_var', torch.empty(2, 2, 1, num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

        self.reset_running_stats()
        self.reset_parameters()

        # remember current mean
        # NOTE: Don't register buffer as this saves current mean and var in the state dict, thereby fixing batch size
        #self.register_buffer('current_mean', torch.empty(2, 1, num_features))
        #self.register_buffer('current_var', torch.empty(2, 2, 1, num_features))
        self.current_mean = torch.empty(2, 1, num_features)
        self.current_var = torch.empty(2, 2, 1, num_features)

    def reset_running_stats(self):
        if not self.track_running_stats:
            return

        self.num_batches_tracked.zero_()

        self.running_mean.zero_()
        self.running_var.copy_(torch.eye(2,  2).unsqueeze(-1).unsqueeze(-1)) # add batch dim

    def reset_parameters(self):
        if not self.affine:
            return

        self.weight.data.copy_(torch.eye(2,  2).unsqueeze(-1).unsqueeze(-1)) # add batch dim
        init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input):
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        return self.cplx_instance_norm(input, self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**vars(self))

    def cplx_instance_norm(
            self,
            input,
            weight=None,
            bias=None,
            eps=1e-05,
    ):
        """Applies complex-valued Instance Normalization.

        Arguments
        ---------
        input : complex-valued tensor
            The input complex-valued data is expected to be at least 2d, with
            shape [B, F, ...], where `B` is the batch dimension, `F` -- the
            channels/features, `...` -- the spatial dimensions (if present).
        running_mean : torch.tensor, or None
            The tensor with running mean statistics having shape [2, F]. Ignored
            if explicitly `None`.
        running_var : torch.tensor, or None
            The tensor with running real-imaginary covariance statistics having
            shape [2, 2, F]. Ignored if explicitly `None`.
        weight : torch.tensor, default=None
            The 2x2 weight matrix of the affine transformation of real and
            imaginary parts post normalization. Has shape [2, 2, F] . Ignored
            together with `bias` if explicitly `None`.
        bias : torch.tensor, or None
            The offest (bias) of the affine transformation of real and imaginary
            parts post normalization. Has shape [2, F] . Ignored together with
            `weight` if explicitly `None`.
        training : bool, default=True
            Determines whether to update running feature statistics, if they are
            provided, or use them instead of batch computed statistics. If `False`
            then `running_mean` and `running_var` MUST be provided.
        momentum : float, default=0.1
            The weight in the exponential moving average used to keep track of the
            running feature statistics.
        eps : float, default=1e-05
            The ridge coefficient to stabilise the estimate of the real-imaginary
            covariance.
        Details
        -------
        Has non standard interface for running stats and weight and bias of the
        affine transformation for purposes of improved memory locality (noticeable
        speedup both on host and device computations).
        """
        # check arguments
        assert ((self.running_mean is None and self.running_var is None)
                or (self.running_mean is not None and self.running_var is not None))
        assert ((self.weight is None and self.bias is None)
                or (self.weight is not None and self.bias is not None))

        # stack along the first axis
        x = torch.stack([input.real, input.imag], dim=0)

        # whiten and apply affine transformation
        #z = self.whiten2x2(x, training=training, momentum=momentum, nugget=eps)
        z = self.whiten2x2(x, nugget=eps)

        if self.weight is not None and self.bias is not None:
            shape = 1, x.shape[2], *([1] * (x.dim() - 3))
            weight = self.weight.reshape(2, 2, *shape)
            z = torch.stack([
                z[0] * weight[0, 0] + z[1] * weight[0, 1],
                z[0] * weight[1, 0] + z[1] * weight[1, 1],
                ], dim=0) + self.bias.reshape(2, *shape)

        return cplx.Cplx(z[0], z[1])

    def whiten2x2(self, tensor, nugget=1e-5):
        r"""Solve R M R = I for R and a given 2x2 matrix M = [[a, b], [c, d]].

        Arguments
        ---------
        tensor : torch.tensor
            The input data expected to be at least 3d, with shape [2, B, F, ...],
            where `B` is the batch dimension, `F` -- the channels/features,
            `...` -- the spatial dimensions (if present). The leading dimension
            `2` represents real and imaginary components (stacked).
        training : bool, default=True
            Determines whether to update running feature statistics, if they are
            provided, or use them instead of batch computed statistics. If `False`
            then `running_mean` and `running_cov` MUST be provided.
        running_mean : torch.tensor, or None
            The tensor with running mean statistics having shape [2, F]. Ignored
            if explicitly `None`.
        running_cov : torch.tensor, or None
            The tensor with running real-imaginary covariance statistics having
            shape [2, 2, F]. Ignored if explicitly `None`.
        momentum : float, default=0.1
            The weight in the exponential moving average used to keep track of the
            running feature statistics.
        nugget : float, default=1e-05
            The ridge coefficient to stabilise the estimate of the real-imaginary
            covariance.
        Details
        -------
        Using (tril) L L^T = V seems to 'favour' the first dimension (re), so
        Trabelsi et al. (2018) used explicit 2x2 root of M: such R that M = RR.
        For M = [[a, b], [c, d]] we have the following facts:
            (1) inv M = \frac1{ad - bc} [[d, -b], [-c, a]]
            (2) \sqrt{M} = \frac1{t} [[a + s, b], [c, d + s]]
                for s = \sqrt{ad - bc}, t = \sqrt{a + d + 2 s}
                det \sqrt{M} = t^{-2} (ad + s(d + a) + s^2 - bc) = s
        Therefore `inv \sqrt{M} = [[p, q], [r, s]]`, where
            [[p, q], [r, s]] = \frac1{t s} [[d + s, -b], [-c, a + s]]
        """
        # assume tensor is 2 x B x F x ...
        assert tensor.dim() >= 3
        # tail shape for broadcasting ? x 1 x F x [*1]
        # batch norm
        #tail = 1, tensor.shape[2], *([1] * (tensor.dim() - 3))
        #axes = 1, *range(3, tensor.dim())
        # instance norm
        tail = tensor.shape[1], tensor.shape[2], *([1] * (tensor.dim() - 3))
        axes = tuple(range(3, tensor.dim()))

        # 1. compute batch mean [2 x F] and center the batch

        if self.training or self.running_mean is None:
            mean = tensor.mean(dim=axes)
            if self.running_mean is not None:
                #running_mean += momentum * (mean.data - running_mean) # avoid in-place due to different shaping
                self.running_mean = self.running_mean + self.momentum * (mean.data - self.running_mean)
                #self.current_mean = mean.reshape(2, *tail)
        else:
            mean = self.running_mean
            #self.current_mean = mean.reshape(2, *tail)
        #mean = tensor.mean(dim=axes)

        # Compute mean
        #mean = tensor.mean(dim=axes)
        self.current_mean = mean.reshape(2, *tail)
        #self.current_mean = torch.zeros_like(mean.reshape(2, *tail))
        tensor = tensor - self.current_mean #mean.reshape(2, *tail)

        # 2. per feature real-imaginary 2x2 covariance matrix
        # running_cov = torch.eye(2,  2).unsqueeze(-1).repeat(1, 1, tensor.shape[2])

        if self.training or self.running_var is None:
            # stabilize by a small ridge
            var = (tensor * tensor).mean(dim=axes) + nugget
            cov_uu, cov_vv = var[0], var[1]

            # has to mul-mean here anyway (naïve) : reduction axes shifted left.
            cov_vu = cov_uv = (tensor[0] * tensor[1]).mean([a - 1 for a in axes])
            if self.running_var is not None:
                cov = torch.stack([
                    cov_uu.data, cov_uv.data,
                    cov_vu.data, cov_vv.data,
                ], dim=0).reshape(2, 2, *cov_vu.shape) #.reshape(2, 2, -1)
                #running_cov += momentum * (cov - running_cov)
                self.running_var = self.running_var + self.momentum * (cov - self.running_var)
        else:
            cov_uu, cov_uv, cov_vu, cov_vv = self.running_var.reshape(4, -1)

        # Compute covariances
        #var = (tensor * tensor).mean(dim=axes) + nugget
        #cov_uu, cov_vv = var[0], var[1]
        # has to mul-mean here anyway (naïve) : reduction axes shifted left.
        #cov_vu = cov_uv = (tensor[0] * tensor[1]).mean([a - 1 for a in axes])

        # 3. get R = [[p, q], [r, s]], with E R c c^T R^T = R M R = I
        # (unsure if intentional, but the inv-root in Trabelsi et al. (2018) uses
        # numpy `np.sqrt` instead of `K.sqrt` so grads are not passed through
        # properly, i.e. constants, [complex_standardization](bn.py#L56-57).
        sqrdet = torch.sqrt(cov_uu * cov_vv - cov_uv * cov_vu) # this is the denominator of the inverse
        # torch.det uses svd, so may yield -ve machine zero

        denom = sqrdet * torch.sqrt(cov_uu + 2 * sqrdet + cov_vv) # this is the denominator of the square root
        p, q = (cov_vv + sqrdet) / denom, -cov_uv / denom
        r, s = -cov_vu / denom, (cov_uu + sqrdet) / denom

        # forward matrix
        p_fw, q_fw = (cov_uu + sqrdet) / denom * sqrdet, cov_uv / denom * sqrdet
        r_fw, s_fw = cov_vu / denom * sqrdet , (cov_vv + sqrdet) / denom * sqrdet
        self.current_var = torch.cat([p_fw, q_fw, r_fw, s_fw], dim=0).reshape(2, 2, *p_fw.shape)

        # 4. apply Q to x (manually)
        out = torch.stack([
            tensor[0] * p.reshape(tail) + tensor[1] * r.reshape(tail),
            tensor[0] * q.reshape(tail) + tensor[1] * s.reshape(tail),
            ], dim=0)
        return out # , torch.cat([p, q, r, s], dim=0).reshape(2, 2, *p.shape)

    def instance_norm_inverse(self, tensor):
        """Instance denormalization to reverse all operations."""
        # stack along the first axis
        tensor = torch.stack([tensor.real, tensor.imag], dim=0)
        # invert re-scaling and re-shifting
        if self.weight is not None and self.bias is not None:
            shape = 1, tensor.shape[2], *([1] * (tensor.dim() - 3))
            weight = self.weight.reshape(2, 2, *shape)
            tensor -= self.bias.reshape(2, *shape)
            denom = (weight[0, 0] * weight[1, 1] - weight[0, 1] * weight[1, 0])
            tensor = torch.stack([
                tensor[0] * weight[1, 1] / denom + tensor[1] * (-weight[0, 1] / denom),
                tensor[0] * (-weight[1, 0] / denom) + tensor[1] * weight[0, 0] / denom,
                ], dim=0)


        # invert current mean and current cov
        # prouces correct inverse up to a tolerance of 1e-7
        p, q, r, s = self.current_var[0,0,...], self.current_var[1,0,...], self.current_var[0,1,...], self.current_var[1,1,...]
        tail = (*p.shape, 1, 1)
        tensor = torch.stack([
            tensor[0] * p.reshape(tail) + tensor[1] * r.reshape(tail),
            tensor[0] * q.reshape(tail) + tensor[1] * s.reshape(tail),
            ], dim=0)

        tensor += self.current_mean

        return cplx.Cplx(tensor[0], tensor[1])
class CplxInstanceNorm2d(_CplxInstanceNorm):
    """Complex-valued Instance Norm for 2D data. See torch.nn.InstanceNorm2d for details."""
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

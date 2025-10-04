import torch
import math 

def create_activation_params(args):
    """Create Radial Basis Function (RBF) activation weights."""
    x_0 = torch.linspace(-1, 1, args.num_weights)

    if args.init_type == "random":
        r1 = - 1 * args.init_scale
        r2 = 1 * args.init_scale
        w =  (r1 - r2) * torch.rand(
            args.num_filters,
            args.num_weights,
            dtype=torch.complex64) + r2
    elif args.init_type == "linear":
        w_0 = args.init_scale * x_0
    elif args.init_type == "sign":
        w_0 = args.init_scale * torch.sign(x_0)
    elif args.init_type == "relu":
        w_0 = args.init_scale * torch.maximum(x_0, torch.as_tensor(0.))
    elif args.init_type == "student-t":
        alpha = 100
        w_0 = args.init_scale * torch.sqrt(torch.as_tensor(alpha)) \
                * x_0 / (1 + 0.5 * alpha * x_0 ** 2)
    else:
        raise ValueError(f"Requested init_type {args.init_type} not defined!")

    if args.init_type != "random":
        w = torch.tile(w_0, (args.num_filters, 1))
        w = torch.complex(w, w)
        
    return w

def create_filter(args, multiplier=1):
    """Create kernels for convolutional filters."""
    k = torch.randn(
        args.out_channels * multiplier,
        args.in_channels,
        args.kernel_size,
        args.kernel_size,
        dtype=torch.complex64
    ) / torch.sqrt(torch.as_tensor(args.kernel_size ** 2 * args.in_channels))

    if args.zero_mean:
        k = zero_mean(k, dim=(1, 2, 3))

    if args.norm:
        k, _ = normalize(k, norm_max=args.norm_max, dim=(1, 2, 3))

    if args.kaiming_uniform:
        k = kaiming_uniform_cplx(k, a=math.sqrt(5))
    return k

def normalize(x, norm_max=1.0, dim=(1,2,3)):
    """Peak magnitude normalization."""
    norm = torch.maximum(
        torch.sqrt(
            torch.sum((x * torch.conj(x)).real, dim=dim, keepdim=True) / norm_max
        ), torch.as_tensor(1.)
    )
    x /= norm
    return x, norm

def zero_mean(x, dim=(1,2,3)):
    """Force zero mean."""
    y = x - torch.mean(x, dim=dim, keepdim=True)
    return y

def kaiming_uniform_cplx(x, a=0.0, mode="fan_in", nonlinearity="leaky_relu"):
    """Complex-valued Kaiming uniform initialization of input tensor."""
    a = math.sqrt(1 + 2 * a * a)
    re = torch.nn.init.kaiming_uniform_(x.real, a=a, mode=mode, nonlinearity=nonlinearity)
    im = torch.nn.init.kaiming_uniform_(x.imag, a=a, mode=mode, nonlinearity=nonlinearity)
    return torch.complex(re, im)



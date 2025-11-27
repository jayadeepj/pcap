import torch


class GaussianNoise:

    def __init__(self, device, mean=0., std=0.5):
        self.device = device
        self.std = std  # SNR = mu(signal)/std(noise); hence std of 0.1 indicates SNR of 10
        self.mean = mean

    def __call__(self, raw):
        """ Add noise relative to the mean of the signal"""
        assert raw.dim() == 1
        relative_std = raw.mean().item() * self.std
        return raw + torch.randn(raw.size(), device=self.device) * relative_std + self.mean


def l2_norm(point1, point2):
    """ Given 2  coordinate tensors, return the L2 dist
        2 points each of 4 x 3 gives 4 x 1 result
    """
    if point1.shape != point1.shape:
        raise ValueError("Invalid shape of tensors to calculate L2 Norm")

    cdist = torch.cdist(point1, point2, p=2.0)
    return torch.diagonal(cdist, 0)


def gen_fd_spacing_idx(d, device):
    """
    Gen spacing indexes depending on the dimensions d
    Represents the left and right in each dimensions for finite differencing.

    E.g.,

    d == 1: = torch.tensor([[-1],
                            [1]])

    d == 2: = torch.tensor([[-1, 0],
                            [0, -1],
                            [0, 1],
                            [1, 0]])

    d == 3: = torch.tensor([[-1, 0, 0],
                            [0, -1, 0],
                            [0, 0, -1],
                            [0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0]])

    """
    spacing_idx = torch.zeros(2 * d, d, dtype=torch.int, device=device)
    start, col_idx = -1, 0
    for idx, row in enumerate(spacing_idx):
        row[col_idx] = start
        col_idx = 0 if col_idx == d - 1 else col_idx + 1
        start = start * -1 if col_idx == d - 1 else start

    return spacing_idx


def approx_gradient_nd(func, x, spacing, device):
    """ Gradient Approximation using Central Differences; requires 2d function evaluations
    Inputs: function to be evaluated, input and the spacing dx.
    Generalises to any number of dimensions

    In x Shape: torch.Size([n, d])
    func signature : torch.Size([n * 2 * d , d]) => torch.Size([n * 2 * d])
    Out Shape : torch.Size([n, d])

    n: No of particles in case of svgd
    d: No of dimensions

    Note: Central Differences are more accurate, but need extra 'd' evaluations compared to fwd/bwd differences
    if d=2, there will be 4 evaluations for every point; hence 4n in total
    """

    n_particles, d = x.shape[0], x.shape[1]
    n_evals = 2 * d

    spacing_idx = gen_fd_spacing_idx(d, device=device)

    rep_x = x.repeat_interleave(n_evals, dim=0)
    _spacing = spacing * spacing_idx
    rep_spacing = _spacing.repeat(n_particles, 1)
    assert rep_spacing.shape == rep_x.shape, "Shape Mismatch"

    x_eval = rep_spacing + rep_x
    assert x_eval.shape == torch.Size([n_particles * n_evals, d]), "Shape Mismatch"

    fx = func(x_eval)
    assert fx.shape == torch.Size([n_particles * n_evals, 1])

    split_fx = torch.split(fx, split_size_or_sections=n_evals)

    def _grad_input_mat(fxi):
        """ The corners in the matrix (except the rhombus/rhomboid shape in the center) need not be evaluated.
         Hence, replace others with zeros."""
        grad_in_mat_shape = tuple(3 for _ in range(d))
        grad_in_mat = torch.zeros(grad_in_mat_shape).to(device)

        for idx, row in enumerate(spacing_idx.tolist()):
            row_plus_1 = tuple(_ + 1 for _ in row)
            grad_in_mat[row_plus_1] = fxi[idx]

        return grad_in_mat

    grad_input_mat = tuple(_grad_input_mat(fxi) for fxi in split_fx)
    grad_input_mat_n = torch.cat(grad_input_mat)

    spacing_shape = tuple(spacing for _ in range(d))
    grad_all_n = torch.gradient(grad_input_mat_n, spacing=spacing_shape)

    def _extract_center_elements(_t, _d):
        search_dim = tuple(1 for _ in range(_d))
        center = tuple(c[search_dim].unsqueeze(0)
                       for c in torch.split(_t, split_size_or_sections=3))
        return torch.cat(center, dim=0)

    grad = torch.cat(tuple(_extract_center_elements(single_dim, d).unsqueeze(1)
                           for single_dim in grad_all_n), dim=1)
    assert grad.shape == x.shape
    return grad


def approx_gradient_1d(func, x, spacing):
    """
    Note: Works only for d == 1
    Gradient Approximation using Central Differences; requires 2d function evaluations
    Inputs: function to be evaluated, input and the spacing dx ,

    In x Shape: n x no_params
    func signature : 3n x no_params => n
    Out Shape : n x no_params

    Note: Central Differences are more accurate, but need extra 'd' evaluations compared to fwd/bwd differences
    if d=1, there will be 3 evaluations for every point; hence 3n in total
    """

    # Calculate prob gradient by finite differences
    x_bwd = x.detach().clone() - spacing
    x_fwd = x.detach().clone() + spacing

    # attaching the fwd & bwd inputs sequentially to enable parallelization
    bwd_x_fwd = torch.cat((x_bwd, x, x_fwd), dim=0)
    eval_fun = func(bwd_x_fwd)

    fwd, cen, bwd = torch.split(eval_fun.unsqueeze(1), split_size_or_sections=x.shape[0])
    gradient_in = torch.cat((fwd, cen, bwd), dim=1)

    gradient_mat = torch.gradient(gradient_in, spacing=spacing, dim=1)

    # extract the central element
    grad = gradient_mat[0][:, 1:2]
    assert grad.shape == x.shape

    return grad


def approx_gradient_2d(func, x, spacing, device):
    """
    Note: Works only for d == 1

    Gradient Approximation using Central Differences; requires 2d function evaluations
    Inputs: function to be evaluated, input and the spacing dx ,

    In x Shape: torch.Size([n, d])
    func signature : torch.Size([n * 2 * d , d]) => torch.Size([n * 2 * d])
    Out Shape : torch.Size([n, d])

    n: No of particles in case of svgd
    d: No of dimensions

    Note: Central Differences are more accurate, but need extra 'd' evaluations compared to fwd/bwd differences
    if d=2, there will be 4 evaluations for every point; hence 4n in total
    """

    n_particles, d = x.shape[0], x.shape[1]
    n_evals = 2 * d
    spacing_idx = torch.tensor([[-1, 0],
                                [0, -1],
                                [0, 1],
                                [1, 0]], device=device)

    rep_x = x.repeat_interleave(n_evals, dim=0)
    _spacing = spacing * spacing_idx
    rep_spacing = _spacing.repeat(n_particles, 1)

    assert rep_spacing.shape == rep_x.shape, "Shape Mismatch"
    x_eval = rep_spacing + rep_x
    assert x_eval.shape == torch.Size([n_particles * n_evals, d]), "Shape Mismatch"

    fx = func(x_eval)
    split_fx = torch.split(fx, split_size_or_sections=n_evals)

    def _grad_input_mat(fxi):
        grad_in_mat = torch.zeros(d + 1, d + 1).to(device)
        for idx, [p, q] in enumerate(spacing_idx.tolist()):
            grad_in_mat[1 + p, 1 + q] = fxi[idx]

        return grad_in_mat

    grad_input_mat = tuple(_grad_input_mat(fxi) for fxi in split_fx)
    grad_input_mat_n = torch.stack(grad_input_mat)
    grad_all = torch.gradient(grad_input_mat_n, spacing=spacing, dim=[1, 2])
    grad = torch.cat(tuple(mat[:, 1, 1].unsqueeze(1) for mat in grad_all), dim=1)

    assert grad.shape == x.shape
    return grad


def mmd(x, y, sigma=0.1):
    """ Find the Maximum Mean Discrepancy MMD between 2 samples. MMD is a distance between feature means in RKHS
     The sigma doesn't matter much as both x * y are projected on to the RKHS with the same sigma
     This logic is taken from Thomas Viehmann
     https://torchdrift.org/notebooks/note_on_mmd.html
     """
    n, d1 = x.shape
    m, d2 = y.shape
    assert d1 == d2
    xy = torch.cat([x.detach(), y.detach()], dim=0)
    dists = torch.cdist(xy, xy, p=2.0)

    # use RBF Kernel
    k = torch.exp((-1 / (2 * sigma ** 2)) * dists ** 2) + torch.eye(n + m).to(x.device) * 1e-5
    k_x = k[:n, :n]
    k_y = k[n:, n:]
    k_xy = k[:n, n:]
    mmd = k_x.sum() / (n * (n - 1)) + k_y.sum() / (m * (m - 1)) - 2 * k_xy.sum() / (n * m)
    return mmd

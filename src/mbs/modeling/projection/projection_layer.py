import torch
import torch.nn as nn


def gaussian_random_matrix(out_dim, in_dim, device, dtype, seed=None):
    """
    Generate a Gaussian random projection matrix with entries ~ N(0, 1/out_dim).

    Parameters
    ----------
    out_dim : int
        Number of output dimensions (rows).

    in_dim : int
        Number of input features (columns).

    device : torch.device or None
        Device to allocate the matrix.

    dtype : torch.dtype
        Data type of the matrix.

    seed : int or None
        Optional seed for reproducibility.

    Returns
    -------
    matrix : torch.Tensor of shape (out_dim, in_dim)
        Dense Gaussian random matrix.
    """
    if seed is not None:
        torch.manual_seed(seed)
    std = torch.tensor(1.0 / out_dim, dtype=dtype, device=device).sqrt()
    return torch.empty(out_dim, in_dim, device=device, dtype=dtype).normal_(mean=0.0, std=std)


def sparse_random_projection_matrix_torch(n_components, n_features, density="auto", device=None, dtype=torch.float32, seed=None):
    """
    Generate a sparse random projection matrix based on Achlioptas/Li et al.

    Non-zero entries are drawn from:
        -sqrt(s)/sqrt(n_components) or +sqrt(s)/sqrt(n_components) with equal prob,
    where s = 1 / density.

    Parameters
    ----------
    n_components : int
        Number of rows (target projection dimensions).

    n_features : int
        Number of columns (original input features).

    density : float or 'auto', default='auto'
        Fraction of non-zero entries per row.
        'auto' sets it to 1 / sqrt(n_features), as recommended by Li et al.

    device : torch.device or None
        Device to store the tensor on.

    dtype : torch.dtype, default=torch.float32
        Tensor data type.

    seed : int or None
        Optional seed for reproducibility.

    Returns
    -------
    matrix : torch.Tensor of shape (n_components, n_features)
        Dense tensor representing the sparse projection matrix.
    """
    if seed is not None:
        torch.manual_seed(seed)

    if density == 'auto':
        density = torch.tensor(1.0 / n_features, device=device, dtype=dtype)
    elif not (0.0 < float(density) <= 1.0):
        raise ValueError("density must be in (0, 1] or 'auto'")
    else:
        density = torch.tensor(density, device=device, dtype=dtype)

    s = 1.0 / density
    scale = s.sqrt() / torch.tensor(n_components, dtype=dtype, device=device).sqrt()

    binom = torch.distributions.Binomial(total_count=n_features, probs=density)
    nonzeros_per_row = binom.sample([n_components]).to(torch.int32)

    rows, cols, vals = [], [], []

    for i, nnz in enumerate(nonzeros_per_row.tolist()):
        col_idx = torch.randperm(n_features, device=device)[:nnz]
        sign = torch.randint(0, 2, (nnz,), device=device) * 2 - 1
        rows.append(torch.full((nnz,), i, dtype=torch.int32, device=device))
        cols.append(col_idx)
        vals.append(sign.to(dtype) * scale)

    row_idx = torch.cat(rows)
    col_idx = torch.cat(cols)
    values = torch.cat(vals)

    sparse_proj = torch.sparse_coo_tensor(
        indices=torch.stack([row_idx, col_idx]),
        values=values,
        size=(n_components, n_features),
        device=device,
        dtype=dtype
    )

    return sparse_proj.to_dense()

class ProjectionLayer(nn.Module):
    """
    Linear layer initialized with random projection matrix (Gaussian or Sparse).

    Parameters
    ----------
    in_features : int
        Input feature dimension.

    out_features : int
        Output feature dimension (projection size).

    init : str, default="gaussian"
        Initialization strategy: "gaussian" or "sparse".

    density : float or 'auto', default=0.01
        Used only if init == "sparse". Controls fraction of non-zeros.

    bias : bool, default=True
        Whether to use a bias term.

    freeze : bool, default=False
        If True, weight and bias parameters will not require gradients.

    seed : int or None
        Optional random seed for reproducibility.

    device : torch.device or None
        Device to initialize the layer on.

    dtype : torch.dtype, default=torch.float32
        Tensor data type.
    """
    def __init__(
        self,
        in_features,
        out_features,
        W=None,
        init="gaussian",
        density=0.01,
        bias=True,
        freeze=False,
        seed=None,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)

        if W is None:
            if init == "gaussian":
                W = gaussian_random_matrix(out_features, in_features, device=device, dtype=dtype, seed=seed)
            elif init == "sparse":
                W = sparse_random_projection_matrix_torch(out_features, in_features, density=density, device=device, dtype=dtype, seed=seed)
            else:
                raise ValueError(f"Unknown init type: {init}. Choose 'gaussian' or 'sparse'.")
        else:
            print("Using provided projector weights for ProjectionLayer initialization.")

        with torch.no_grad():
            self.linear.weight.copy_(W)
            if bias:
                self.linear.bias.zero_()

        if freeze:
            self.linear.weight.requires_grad = False
            if bias:
                self.linear.bias.requires_grad = False

    def forward(self, x):
        return self.linear(x)
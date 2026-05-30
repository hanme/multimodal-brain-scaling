from typing import Optional, Literal

import torch
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


def create_loss_func(loss_func, **kwargs):        
    mixup = kwargs.get('mixup', 0.)
    cutmix = kwargs.get('cutmix', 0.)
    label_smoothing = kwargs.get('label_smoothing', 0.)
    
    match loss_func:
        
        case 'cross_entropy':
            assert mixup == 0. and cutmix == 0., \
                "Mixup and Cutmix not supported with Cross Entropy loss"
            return nn.CrossEntropyLoss()
        
        case 'soft_target_cross_entropy':    
            return SoftTargetCrossEntropy()
        
        case 'label_smoothing_cross_entropy' :
            assert mixup == 0. and cutmix == 0., \
                "Mixup and Cutmix not supported with Label Smoothing Cross Entropy loss"
            assert label_smoothing > 0., \
                "Label Smoothing Cross Entropy loss requires label_smoothing > 0."
            return LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        
        case 'multiclass_bce':
            return nn.BCEWithLogitsLoss()
        
        case 'mse':
            return nn.MSELoss()
        
        case 'mae':
            return nn.L1Loss()
        
        case 'cka':
            return CKALoss()
        
        case 'logcka':
            return CKALoss(log=True)

        case _:
            raise NotImplementedError(f"Loss function '{loss_func}' is not implemented")
    


class CenteredKernelAlignmentTorch(nn.Module):
    """
    Centered Kernel Alignment (CKA) in PyTorch (GPU-friendly).

    Supports:
      - 'linear':
          - unbiased=False → fast, standard (biased) linear CKA
          - unbiased=True  → unbiased HSIC-based linear CKA
      - 'rbf':
          kernel CKA using biased HSIC (unbiased not implemented)

    X, Y are expected to be (n_samples, ...) and are flattened along feature dims.
    """

    def __init__(
        self,
        kernel_type: Literal["linear", "rbf"] = "linear",
        sigma: Optional[float] = None,
        unbiased: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.unbiased = unbiased
        self.eps = eps

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Batch sizes must match along axis 0: {X.shape[0]} vs {Y.shape[0]}"
            )

        # Flatten feature dims
        X = X.view(X.shape[0], -1)
        Y = Y.view(Y.shape[0], -1)

        if self.kernel_type == "linear":
            if self.unbiased:
                return self._unbiased_linear_cka(X, Y)
            else:
                # Standard (biased) fast linear CKA
                Xc = X - X.mean(dim=0, keepdim=True)
                Yc = Y - Y.mean(dim=0, keepdim=True)
                return self._fast_linear_cka(Xc, Yc)

        elif self.kernel_type == "rbf":
            if self.unbiased:
                raise ValueError(
                    "Unbiased CKA is only implemented for the linear kernel."
                )
            return self._kernel_cka(X, Y)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")

    # ---------- Biased linear CKA (fast form) ----------

    def _fast_linear_cka(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Fast linear CKA using Frobenius norms:

            CKA(X, Y) = ||Yᵀ X||_F² / (||Xᵀ X||_F · ||Yᵀ Y||_F)

        Assumes X and Y are already centered across samples.
        """
        num = self._matrix_inner_product(X, Y) ** 2
        denom = self._matrix_inner_product(X, X) * self._matrix_inner_product(Y, Y)
        if denom <= self.eps:
            return X.new_tensor(0.0)
        return num / denom

    @staticmethod
    def _matrix_inner_product(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Frobenius norm of Yᵀ X:
            <X, Y> = ||Yᵀ X||_F
        """
        M = Y.t() @ X
        return torch.norm(M, p="fro")

    # ---------- Unbiased linear CKA via unbiased HSIC ----------

    def _unbiased_linear_hsic(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Unbiased HSIC estimator (Song et al.) for linear kernel.

        X: [n, d_x]
        Y: [n, d_y]
        Returns: scalar HSIC estimate.
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Batch sizes must match: {X.shape[0]} vs {Y.shape[0]}")
        n = X.shape[0]
        if n < 4:
            raise ValueError(f"Unbiased HSIC requires at least 4 samples, got n={n}")

        device = X.device
        dtype = X.dtype

        # Linear kernel Gram matrices
        K = X @ X.t()   # [n, n]
        L = Y @ Y.t()   # [n, n]

        # Zero out diagonals
        K = K.clone()
        L = L.clone()
        K.fill_diagonal_(0.0)
        L.fill_diagonal_(0.0)

        ones = torch.ones(n, 1, device=device, dtype=dtype)

        # term1 = tr(K L) = sum_ij K_ij L_ij
        term1 = (K * L).sum()

        # term2 = (1^T K 1)(1^T L 1)
        K1 = K @ ones   # [n, 1]
        L1 = L @ ones   # [n, 1]
        term2 = K1.sum() * L1.sum()

        # term3 = 1^T K L 1 = sum_i K1_i L1_i
        term3 = (K1 * L1).sum()

        coef = 1.0 / (n * (n - 3.0))
        hsic = coef * (
            term1
            + term2 / ((n - 1.0) * (n - 2.0))
            - 2.0 * term3 / (n - 2.0)
        )
        return hsic

    def _unbiased_linear_cka(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Unbiased linear CKA:

            CKA_unb(X, Y) =
                HSIC_unb(X, Y) / sqrt(HSIC_unb(X, X) HSIC_unb(Y, Y))
        """
        hsic_xy = self._unbiased_linear_hsic(X, Y)
        hsic_xx = self._unbiased_linear_hsic(X, X)
        hsic_yy = self._unbiased_linear_hsic(Y, Y)

        prod = hsic_xx * hsic_yy
        if prod <= 0:
            return X.new_tensor(0.0)
        denom = torch.sqrt(prod) + self.eps
        return hsic_xy / denom

    # ---------- Kernel CKA (RBF, biased HSIC) ----------

    def _get_kernel(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute kernel matrix K(X, Y). Currently supports only RBF.
        """
        if Y is None:
            Y = X

        if self.kernel_type == "rbf":
            if self.sigma is None:
                # median heuristic on X
                n = X.shape[0]
                if n > 1:
                    # pairwise distances via cdist
                    # To avoid huge memory, you may want batch versions in practice.
                    dists = torch.cdist(X, X, p=2.0)
                    i, j = torch.triu_indices(n, n, offset=1, device=X.device)
                    vals = dists[i, j]
                    if vals.numel() > 0:
                        self.sigma = float(vals.median().item())
                    else:
                        self.sigma = 1.0
                else:
                    self.sigma = 1.0

            sq_dists = torch.cdist(X, Y, p=2.0) ** 2
            K = torch.exp(-sq_dists / (2.0 * (self.sigma ** 2)))
            return K

        raise ValueError(f"Unsupported kernel type for _get_kernel: {self.kernel_type}")

    @staticmethod
    def _center_gram_matrix(K: torch.Tensor) -> torch.Tensor:
        """
        Center a Gram matrix K using:
            K_c = H K H,  with H = I - 1/n · 11ᵀ
        """
        n = K.shape[0]
        device = K.device
        dtype = K.dtype
        H = torch.eye(n, device=device, dtype=dtype) - torch.ones(
            (n, n), device=device, dtype=dtype
        ) / n
        return H @ K @ H

    def _hsic_xy_biased(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Biased HSIC(X, Y) = sum(K_c ⊙ L_c).
        """
        K = self._get_kernel(X)
        L = self._get_kernel(Y)
        Kc = self._center_gram_matrix(K)
        Lc = self._center_gram_matrix(L)
        return (Kc * Lc).sum()

    def _hsic_self_biased(self, X: torch.Tensor) -> torch.Tensor:
        """
        Biased HSIC(X, X).
        """
        K = self._get_kernel(X)
        Kc = self._center_gram_matrix(K)
        return (Kc * Kc).sum()

    def _kernel_cka(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Kernel CKA using biased HSIC.
        """
        hsic_xy = self._hsic_xy_biased(X, Y)
        hsic_xx = self._hsic_self_biased(X)
        hsic_yy = self._hsic_self_biased(Y)

        prod = hsic_xx * hsic_yy
        if prod <= 0:
            return X.new_tensor(0.0)
        denom = torch.sqrt(prod) + self.eps
        return hsic_xy / denom     

class CenteredKernelAlignment(nn.Module):
    """
    Centered Kernel Alignment (CKA) implementation with fast linear kernel option.
    Uses direct computation for linear kernel and Gram matrices for other kernels.
    Returns raw CKA value.
    """
    def __init__(self, 
                 kernel_type: Literal['linear', 'rbf'] = 'linear',
                 sigma: Optional[float] = None):
        super(CenteredKernelAlignment, self).__init__()
        self.kernel_type = kernel_type
        self.sigma = sigma

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        # Input validation
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Batch sizes don't match: {X.shape[0]} vs {Y.shape[0]}")

        # Reshape inputs
        X = X.view(X.shape[0], -1)
        Y = Y.view(Y.shape[0], -1)

        """
        For the linear kernel case, centering the data before computing the Gram matrix
        is equivalent to centering the Gram matrix itself. Here's why:

        In the kernel-based approach:
        1. We compute K = XX^T (Gram matrix)
        2. We center it using H = I - 1/n: HKH = H(XX^T)H
        
        In the fast implementation:
        1. We center X directly: X_c = HX = X - μ_x
        2. Then compute X_c^T X_c
        
        These are equivalent because:
        H(XX^T)H = (HX)(HX)^T = X_c X_c^T

        This is why we can center the data first in the fast implementation
        and still get the same result as centering the Gram matrix.
        """
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)

        # Use fast implementation for linear kernel
        if self.kernel_type == 'linear':
            return self._fast_linear_cka(X, Y)
        else:
            return self._kernel_cka(X, Y)

    def _fast_linear_cka(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Fast implementation of linear CKA using Frobenius norm.
        
        For linear kernel, CKA can be computed directly using centered data:
        CKA(X,Y) = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)

        This is equivalent to the HSIC-based formulation because:
        1. For linear kernel, HSIC(X,Y) = Tr(KHLH) where K = XX^T, L = YY^T
        2. When data is centered (H applied), this becomes Tr((X_c X_c^T)(Y_c Y_c^T))
        3. Using trace properties: Tr(X_c X_c^T Y_c Y_c^T) = ||Y_c^T X_c||_F^2
        
        The denominator follows the same principle, making the implementations equivalent.
        """
        return self._matrix_inner_product(X, Y)**2 / (self._matrix_inner_product(X, X) * self._matrix_inner_product(Y, Y))

    @staticmethod
    def _matrix_inner_product(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute inner product between matrices using Frobenius norm:
        <X,Y> = Tr(X^T Y) = ||X^T Y||_F
        
        This is used in the fast implementation because:
        1. For centered data, HSIC(X,Y) = Tr(X_c X_c^T Y_c Y_c^T)
        2. This equals ||Y_c^T X_c||_F^2 by trace properties
        """
        return torch.norm(torch.matmul(Y.t(), X), p='fro')

    def _get_kernel(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute kernel matrix.
        For non-linear kernels, we must use the full kernel formulation
        as the fast implementation only works for linear kernel.
        """
        if Y is None:
            Y = X

        if self.kernel_type == 'rbf':
            if self.sigma is None:
                # Heuristic: median distance between points
                if X.shape[0] > 1:
                    self.sigma = torch.median(torch.pdist(X))
                else:
                    self.sigma = 1.0
                    
            diff = X.unsqueeze(1) - Y.unsqueeze(0)
            sq_dist = torch.sum(diff ** 2, dim=-1)
            return torch.exp(-sq_dist / (2 * self.sigma ** 2))
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")

    def _center_gram_matrix(self, K: torch.Tensor) -> torch.Tensor:
        """
        Center the Gram matrix using centering matrix H = I - 1/n.
        
        For the linear kernel case, this is equivalent to centering the data first.
        For non-linear kernels, we must center the kernel matrix explicitly.
        """
        n = K.shape[0]
        unit = torch.ones(n, n, device=K.device)
        I = torch.eye(n, device=K.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)

    def _hsic(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute Hilbert-Schmidt Independence Criterion.
        This is the unbiased estimator of HSIC used in CKA.
        """
        K = self._center_gram_matrix(self._get_kernel(X))
        L = self._center_gram_matrix(self._get_kernel(Y))
        return torch.sum(K * L)

    def _kernel_cka(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute CKA using HSIC for non-linear kernels.
        HSIC is used because it measures the dependence between 
        feature maps in the kernel's implicit feature space.
        """
        hsic_xy = self._hsic(X, Y)
        hsic_xx = self._hsic(X, X)
        hsic_yy = self._hsic(Y, Y)
        
        # Add small epsilon for numerical stability
        epsilon = 1e-8
        return hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + epsilon)
    
    

class CKALoss(nn.Module):
    """
    Converts CKA similarity into a loss value.
    Can return either 1-CKA or -log(CKA) as the loss.
    
    Args:
        kernel_type: Type of kernel to use for CKA computation ('linear' or 'rbf')
        sigma: Bandwidth parameter for RBF kernel (only used if kernel_type='rbf')
        log: If True, returns -log(CKA), otherwise returns 1-CKA
    """
    def __init__(self,
                 kernel_type: Literal['linear', 'rbf'] = 'linear',
                 sigma: Optional[float] = None,
                 log: bool = False):
        super(CKALoss, self).__init__()
        self.cka = CenteredKernelAlignment(kernel_type=kernel_type, sigma=sigma)
        # self.cka = CenteredKernelAlignmentTorch(kernel_type=kernel_type, sigma=sigma)
        self.log = log
        if self.log:
            self.__class__.__name__ = f'CKALogLoss'
        else:
            self.__class__.__name__ = f'CKALoss'
        
    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute CKA-based loss between X and Y.
        
        Args:
            X: First input tensor
            Y: Second input tensor
            
        Returns:
            torch.Tensor: Loss value. 
                If self.log=True: returns -log(CKA)
                If self.log=False: returns 1-CKA
        """
        cka_value = self.cka(X, Y)
        
        # Add small epsilon for numerical stability with log
        epsilon = 1e-8
        
        if self.log:
            return torch.log(1-cka_value + epsilon)
        else:
            return 1.0 - cka_value
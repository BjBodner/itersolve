from __future__ import annotations

import torch

from itersolve.torch_jacobi import get_random_diagonally_dominant_matrix


def test_nothing():
    dim = 3000
    conditioning_factor = 5000
    sparsity = 0.99
    lr = 0.5
    torch.manual_seed(1)
    A = get_random_diagonally_dominant_matrix(
        dim, conditioning_factor, sparsity=sparsity
    )

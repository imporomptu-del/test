"""Reuse recognized NumPy quadratic-form execution graphs, not approximations.

The tracker uses float64 C-contiguous residuals and inverse covariance. NumPy
1.x's optimized einsum uses tensordot followed by a C einsum reduction; newer
NumPy uses two matmul contractions. These are not numerically interchangeable.
Unrecognized contraction plans/layouts keep the original public einsum path.
"""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=64)
def execution_plan(count, dimension):
    residual = np.empty((count, dimension), np.float64)
    covariance = np.empty((dimension, dimension), np.float64)
    path = np.einsum_path("ni,ij,nj->n", residual, covariance, residual, optimize=True)[0]
    if dimension not in (2, 4):
        return "reference", path
    try:
        _, contractions = np.einsum_path("ni,ij,nj->n", residual, covariance, residual,
            optimize=path, einsum_call=True)
        if contractions == [
            ((1, 0), {"i"}, "ij,ni->jn", ["nj", "jn"], True),
            ((1, 0), {"j"}, "jn,nj->n", ["n"], False),
        ]:
            return "tensordot_c_einsum", path
        if contractions == [
            ((1, 0), "ij,ni->jn", ["nj", "jn"]),
            ((1, 0), "jn,nj->n", ["n"]),
        ]:
            return "batched_matmul", path
    except (TypeError, ValueError, IndexError):
        pass
    return "reference", path


def mahalanobis_values(residuals, inverse_covariance):
    count, dimension = residuals.shape
    mode, path = execution_plan(count, dimension)
    supported = (residuals.dtype == np.float64 and inverse_covariance.dtype == np.float64
        and residuals.flags.c_contiguous and inverse_covariance.flags.c_contiguous
        and inverse_covariance.shape == (dimension, dimension))
    if supported and mode == "tensordot_c_einsum":
        product = np.tensordot(inverse_covariance, residuals, axes=((0,), (1,)))
        return np.einsum("jn,nj->n", product, residuals, optimize=False)
    if supported and mode == "batched_matmul":
        product = inverse_covariance.T @ residuals.T
        return np.matmul(product.T[:, None, :], residuals[:, :, None]).reshape(count)
    return np.einsum("ni,ij,nj->n", residuals, inverse_covariance, residuals, optimize=path)

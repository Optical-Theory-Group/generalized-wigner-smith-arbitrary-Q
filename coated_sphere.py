"""Module containing functions for calculating scattering coefficients for
a coated sphere"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import scipy.optimize
import functools
import scipy.integrate
from tqdm import tqdm
from typing import Any, Callable
import quadpy
import bessel
import utils

# -----------------------------------------------------------------------------
# a coefficient
# -----------------------------------------------------------------------------


def get_A(
    k: complex,
    r_core: float,
    n_core_func: Callable[[complex], complex],
    n_coating_func: Callable[[complex], complex],
    n_background: complex,
    n: int,
) -> complex:
    """Calculate A_n, which is used in calculating the scattering coefficients
    of a coated sphere"""

    n_core = n_core_func(k)
    n_coating = n_coating_func(k)
    m1 = n_core / n_background
    m2 = n_coating / n_background

    # Get the refractive index first. The sphere is assumed to be in air
    # m = get_n_silver_drude(k)
    x = k * r_core * n_background
    # y = k * r_coating * n_background

    psi_m1x = bessel.get_psi(n, m1 * x)
    psi_m2x = bessel.get_psi(n, m2 * x)
    dpsi_m1x = bessel.get_dpsi(n, m1 * x)
    dpsi_m2x = bessel.get_dpsi(n, m2 * x)
    chi_m2x = bessel.get_chi(n, m2 * x)
    dchi_m2x = bessel.get_dchi(n, m2 * x)

    top = m2 * psi_m2x * dpsi_m1x - m1 * dpsi_m2x * psi_m1x
    bottom = m2 * chi_m2x * dpsi_m1x - m1 * dchi_m2x * psi_m1x
    out = top / bottom
    return out


def get_a(
    k: complex,
    r_core: float,
    r_coating: float,
    n_core_func: Callable[[complex], complex],
    n_coating_func: Callable[[complex], complex],
    n_background: complex,
    n: int,
) -> complex:
    """Get the an coefficient (from Mischenko) for a silver sphere with
    wavenumber k and radius r"""

    # n_core = n_core_func(k)
    n_coating = n_coating_func(k)

    # m1 = n_core / n_background
    m2 = n_coating / n_background

    # Get the refractive index first. The sphere is assumed to be in air
    # m = get_n_silver_drude(k)
    # x = k * r_core * n_background
    y = k * r_coating * n_background

    # Work out all the bessel functions
    A = get_A(k, r_core, n_core_func, n_coating_func, n_background, n)

    psi_y = bessel.get_psi(n, y)
    dpsi_y = bessel.get_dpsi(n, y)
    psi_m2y = bessel.get_psi(n, m2 * y)
    dpsi_m2y = bessel.get_dpsi(n, m2 * y)
    xi_y = bessel.get_xi(n, y)
    dxi_y = bessel.get_dxi(n, y)
    chi_m2y = bessel.get_chi(n, m2 * y)
    dchi_m2y = bessel.get_dchi(n, m2 * y)

    top = psi_y * (dpsi_m2y - A * dchi_m2y) - m2 * dpsi_y * (
        psi_m2y - A * chi_m2y
    )
    bottom = xi_y * (dpsi_m2y - A * dchi_m2y) - m2 * dxi_y * (
        psi_m2y - A * chi_m2y
    )
    out = -top / bottom
    return out


def get_da_dnb(
    k: complex,
    r_core: float,
    r_coating: float,
    n_core_func: Callable[[complex], complex],
    n_coating_func: Callable[[complex], complex],
    n_background: complex,
    n: int,
    dnb: float = 1e-8,
) -> complex:
    """Calculate A_n, which is used in calculating the scattering coefficients
    of a coated sphere"""
    low = get_a(
        k, r_core, r_coating, n_core_func, n_coating_func, n_background - dnb / 2, n
    )
    high = get_a(
        k, r_core, r_coating, n_core_func, n_coating_func, n_background + dnb / 2, n
    )
    diff = (high - low) / dnb
    return diff


def get_da_dk(
    k: complex,
    r_core: float,
    r_coating: float,
    n_core_func: Callable[[complex], complex],
    n_coating_func: Callable[[complex], complex],
    n_background: complex,
    n: int,
    dk: float = 1e-8,
) -> complex:
    """Calculate A_n, which is used in calculating the scattering coefficients
    of a coated sphere"""
    low = get_a(
        k - dk / 2, r_core, r_coating, n_core_func, n_coating_func, n_background, n
    )
    high = get_a(
        k + dk / 2, r_core, r_coating, n_core_func, n_coating_func, n_background, n
    )
    diff = (high - low) / dk
    return diff


def get_ws_a_k(
    k: complex,
    r_core: float,
    r_coating: float,
    n_core_func: Callable[[complex], complex],
    n_coating_func: Callable[[complex], complex],
    n_background: complex,
    n: int,
    dk: float = 1e-4,
) -> complex:
    a = get_a(k, r_core, r_coating, n_core_func, n_coating_func, n_background, n)
    da_dk = get_da_dk(
        k, r_core, r_coating, n_core_func, n_coating_func, n_background, n, dk
    )
    ws = -1j * 1 / a * da_dk
    return ws


def get_ws_a_nb(
    k: complex,
    r_core: float,
    r_coating: float,
    n_core_func: Callable[[complex], complex],
    n_coating_func: Callable[[complex], complex],
    n_background: complex,
    n: int,
    dnb: float = 1e-4,
) -> complex:
    a = get_a(k, r_core, r_coating, n_core_func, n_coating_func, n_background, n)
    da_dnb = get_da_dnb(
        k, r_core, r_coating, n_core_func, n_coating_func, n_background, n, dnb
    )
    ws = -1j * 1 / a * da_dnb
    return ws


# def get_dA_dnb(
#     k: complex,
#     r_core: float,
#     n_core_func: Callable[[complex], complex],
#     n_coating: complex,
#     n_background: complex,
#     n: int,
# ) -> complex:
#     """Calculate A_n, which is used in calculating the scattering coefficients
#     of a coated sphere"""

#     n_core = n_core_func(k)
#     m1 = n_core / n_background
#     m2 = n_coating / n_background

#     # Get the refractive index first. The sphere is assumed to be in air
#     # m = get_n_silver_drude(k)
#     x = k * r_core * n_background
#     # y = k * r_coating * n_background

#     psi_m1x = bessel.get_psi(n, m1 * x)
#     psi_m2x = bessel.get_psi(n, m2 * x)
#     dpsi_m1x = bessel.get_dpsi(n, m1 * x)
#     dpsi_m2x = bessel.get_dpsi(n, m2 * x)
#     d2psi_m1x = bessel.get_d2psi(n, m1 * x)
#     d2psi_m2x = bessel.get_d2psi(n, m2 * x)
#     chi_m2x = bessel.get_chi(n, m2 * x)
#     dchi_m2x = bessel.get_dchi(n, m2 * x)
#     d2chi_m2x = bessel.get_d2chi(n, m2 * x)

#     top = m2 * psi_m2x * dpsi_m1x - m1 * dpsi_m2x * psi_m1x
#     dtop = (
#         1
#         / n_background**2
#         * (
#             -n_coating * psi_m2x * dpsi_m1x
#             - m2 * dpsi_m2x * dpsi_m1x * n_coating * x
#             - m2 * psi_m2x * d2psi_m1x * n_core * x
#             + n_core * dpsi_m2x * psi_m1x
#             + m1 * d2psi_m2x * psi_m1x * n_coating * x
#             + m1 * dpsi_m2x * dpsi_m1x * n_core * x
#         )
#     )

#     bottom = m2 * chi_m2x * dpsi_m1x - m1 * dchi_m2x * psi_m1x
#     dbottom = (
#         1
#         / n_background**2
#         * (
#             -n_coating * chi_m2x * dpsi_m1x
#             - m2 * dchi_m2x * dpsi_m1x * n_coating * x
#             - m2 * chi_m2x * d2psi_m1x * n_core * x
#             + n_core * dchi_m2x * psi_m1x
#             + m1 * d2chi_m2x * psi_m1x * n_coating * x
#             + m1 * dchi_m2x * dpsi_m1x * n_core * x
#         )
#     )

#     out = (dtop * bottom - top * dbottom) / bottom**2
#     return out


# def get_a_top(
#     k: complex,
#     r_core: float,
#     r_coating: float,
#     n_core_func: Callable[[complex], complex],
#     n_coating: complex,
#     n_background: complex,
#     n: int,
# ) -> complex:
#     """Get the an coefficient (from Mischenko) for a silver sphere with
#     wavenumber k and radius r"""

#     # n_core = n_core_func(k)
#     # m1 = n_core / n_background
#     m2 = n_coating / n_background

#     # Get the refractive index first. The sphere is assumed to be in air
#     # m = get_n_silver_drude(k)
#     # x = k * r_core * n_background
#     y = k * r_coating * n_background

#     # Work out all the bessel functions
#     A = get_A(k, r_core, n_core_func, n_coating, n_background, n)

#     psi_y = bessel.get_psi(n, y)
#     dpsi_y = bessel.get_dpsi(n, y)
#     psi_m2y = bessel.get_psi(n, m2 * y)
#     dpsi_m2y = bessel.get_dpsi(n, m2 * y)
#     chi_m2y = bessel.get_chi(n, m2 * y)
#     dchi_m2y = bessel.get_dchi(n, m2 * y)

#     top = psi_y * (dpsi_m2y - A * dchi_m2y) - m2 * dpsi_y * (
#         psi_m2y - A * chi_m2y
#     )
#     out = top
#     return out


# def get_da_top_dnb(
#     k: complex,
#     r_core: float,
#     r_coating: float,
#     n_core_func: Callable[[complex], complex],
#     n_coating: complex,
#     n_background: complex,
#     n: int,
# ) -> complex:
#     """Get the an coefficient (from Mischenko) for a silver sphere with
#     wavenumber k and radius r"""

#     # n_core = n_core_func(k)
#     # m1 = n_core / n_background
#     m2 = n_coating / n_background

#     # Get the refractive index first. The sphere is assumed to be in air
#     # m = get_n_silver_drude(k)
#     # x = k * r_core * n_background
#     y = k * r_coating * n_background

#     # Work out all the bessel functions
#     A = get_A(k, r_core, n_core_func, n_coating, n_background, n)

#     psi_y = bessel.get_psi(n, y)
#     dpsi_y = bessel.get_dpsi(n, y)
#     psi_m2y = bessel.get_psi(n, m2 * y)
#     dpsi_m2y = bessel.get_dpsi(n, m2 * y)
#     chi_m2y = bessel.get_chi(n, m2 * y)
#     dchi_m2y = bessel.get_dchi(n, m2 * y)

#     top = psi_y * (dpsi_m2y - A * dchi_m2y) - m2 * dpsi_y * (
#         psi_m2y - A * chi_m2y
#     )
#     out = top
#     return out


# def get_a_bottom(
#     k: complex,
#     r_core: float,
#     r_coating: float,
#     n_core_func: Callable[[complex], complex],
#     n_coating: complex,
#     n_background: complex,
#     n: int,
# ) -> complex:
#     """Get the an coefficient (from Mischenko) for a silver sphere with
#     wavenumber k and radius r"""

#     # n_core = n_core_func(k)
#     # m1 = n_core / n_background
#     m2 = n_coating / n_background

#     # Get the refractive index first. The sphere is assumed to be in air
#     # m = get_n_silver_drude(k)
#     # x = k * r_core * n_background
#     y = k * r_coating * n_background

#     # Work out all the bessel functions
#     A = get_A(k, r_core, n_core_func, n_coating, n_background, n)

#     psi_m2y = bessel.get_psi(n, m2 * y)
#     dpsi_m2y = bessel.get_dpsi(n, m2 * y)
#     xi_y = bessel.get_xi(n, y)
#     dxi_y = bessel.get_dxi(n, y)
#     chi_m2y = bessel.get_chi(n, m2 * y)
#     dchi_m2y = bessel.get_dchi(n, m2 * y)

#     bottom = xi_y * (dpsi_m2y - A * dchi_m2y) - m2 * dxi_y * (
#         psi_m2y - A * chi_m2y
#     )
#     out = bottom
#     return out

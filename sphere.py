"""Module containing functions for calculating scattering coefficients"""

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


def get_a(
    k: complex,
    r: float,
    n_particle_func: Callable[[complex], complex],
    n_background: complex,
    n: int,
) -> complex:
    """Get the an coefficient (from Mischenko) for a silver sphere with
    wavenumber k and radius r"""

    n_particle = n_particle_func(k)
    m = n_particle / n_background

    # Get the refractive index first. The sphere is assumed to be in air
    # m = get_n_silver_drude(k)
    x = k * r * n_background

    # Work out all the bessel functions
    psi_x = bessel.get_psi(n, x)
    dpsi_x = bessel.get_dpsi(n, x)
    psi_mx = bessel.get_psi(n, m * x)
    dpsi_mx = bessel.get_dpsi(n, m * x)
    xi_x = bessel.get_xi(n, x)
    dxi_x = bessel.get_dxi(n, x)

    a = -(m * psi_mx * dpsi_x - psi_x * dpsi_mx) / (
        m * psi_mx * dxi_x - xi_x * dpsi_mx
    )
    return a

def get_b(
    k: complex,
    r: float,
    n_particle_func: Callable[[complex], complex],
    n_background: complex,
    n: int,
) -> complex:
    """Get the an coefficient (from Mischenko) for a silver sphere with
    wavenumber k and radius r"""

    n_particle = n_particle_func(k)
    m = n_particle / n_background

    # Get the refractive index first. The sphere is assumed to be in air
    # m = get_n_silver_drude(k)
    x = k * r * n_background

    # Work out all the bessel functions
    psi_x = bessel.get_psi(n, x)
    dpsi_x = bessel.get_dpsi(n, x)
    psi_mx = bessel.get_psi(n, m * x)
    dpsi_mx = bessel.get_dpsi(n, m * x)
    xi_x = bessel.get_xi(n, x)
    dxi_x = bessel.get_dxi(n, x)

    b = -(psi_mx * dpsi_x - m*psi_x * dpsi_mx) / (
        psi_mx * dxi_x - m*xi_x * dpsi_mx
    )
    return b


def get_a_numerator(
    k: complex,
    r: float,
    n_particle_func: Callable[[complex], complex],
    n_background: complex,
    n: int,
) -> tuple[complex, complex]:
    """Get the an coefficient (from Mischenko) for a silver sphere with
    wavenumber k and radius r"""
    n_particle = n_particle_func(k)
    m = n_particle / n_background

    # Get the refractive index first. The sphere is assumed to be in air
    # m = get_n_silver_drude(k)
    x = k * r * n_background

    # Work out all the bessel functions
    psi_x = bessel.get_psi(n, x)
    dpsi_x = bessel.get_dpsi(n, x)
    psi_mx = bessel.get_psi(n, m * x)
    dpsi_mx = bessel.get_dpsi(n, m * x)

    return m * psi_mx * dpsi_x - psi_x * dpsi_mx


# def get_da_numerator_dk(
#     k: complex, r: float, m_func: complex, n: int
# ) -> tuple[complex, complex]:
#     """Derivative of the numerator with respect to k"""
#     m = m_func(k)

#     # Get the refractive index first. The sphere is assumed to be in air
#     # m = get_n_silver_drude(k)
#     x = k * r

#     # Work out all the bessel functions
#     psi_x = bessel.get_psi(n, x)
#     psi_mx = bessel.get_psi(n, m * x)
#     dpsi_x = bessel.get_dpsi(n, x)
#     dpsi_mx = bessel.get_dpsi(n, m * x)
#     d2psi_x = bessel.get_d2psi(n, x)
#     d2psi_mx = bessel.get_d2psi(n, m * x)

#     term_one = r * (m**2 - 1) * dpsi_mx * dpsi_x
#     term_two = r * m * (psi_mx * d2psi_x - psi_x * d2psi_mx)
#     return term_one + term_two


def get_da_numerator_dr(
    k: complex,
    r: float,
    n_particle_func: Callable[[complex], complex],
    n_background: complex,
    n: int,
) -> tuple[complex, complex]:
    """Derivative of the numerator with respect to the radius of the sphere"""
    n_particle = n_particle_func(k)
    m = n_particle / n_background

    # Get the refractive index first. The sphere is assumed to be in air
    # m = get_n_silver_drude(k)
    x = k * r * n_background

    # Work out all the bessel functions
    psi_x = bessel.get_psi(n, x)
    psi_mx = bessel.get_psi(n, m * x)
    dpsi_x = bessel.get_dpsi(n, x)
    dpsi_mx = bessel.get_dpsi(n, m * x)
    d2psi_x = bessel.get_d2psi(n, x)
    d2psi_mx = bessel.get_d2psi(n, m * x)

    term_one = k * (m**2 - 1) * dpsi_mx * dpsi_x
    term_two = k * m * (psi_mx * d2psi_x - psi_x * d2psi_mx)
    return term_one + term_two


def get_da_numerator_dnb(
    k: complex,
    r: float,
    n_particle_func: Callable[[complex], complex],
    n_background: complex,
    n: int,
) -> tuple[complex, complex]:
    """Derivative of a numerator with respect to the outside refractive index

    m = n_particle / n_background
    """
    n_particle = n_particle_func(k)
    m = n_particle / n_background

    # Get the refractive index first. The sphere is assumed to be in air
    # m = get_n_silver_drude(k)
    x = k * r * n_background

    # Work out all the bessel functions
    psi_x = bessel.get_psi(n, x)
    psi_mx = bessel.get_psi(n, m * x)
    dpsi_x = bessel.get_dpsi(n, x)
    dpsi_mx = bessel.get_dpsi(n, m * x)
    # d2psi_x = bessel.get_d2psi(n, x)
    d2psi_mx = bessel.get_d2psi(n, m * x)

    term_one = psi_mx * dpsi_x * (-n_particle / n_background**2)
    term_two = m * dpsi_mx * dpsi_x * (-x * n_particle / n_background**2)
    term_three = psi_x * d2psi_mx * (x * n_particle / n_background**2)
    return term_one + term_two + term_three


def get_a_denominator(
    k: complex,
    r: float,
    n_particle_func: Callable[[complex], complex],
    n_background: complex,
    n: int,
) -> tuple[complex, complex]:
    """Get the an coefficient (from Mischenko) for a silver sphere with
    wavenumber k and radius r"""
    n_particle = n_particle_func(k)
    m = n_particle / n_background

    # Get the refractive index first. The sphere is assumed to be in air
    # m = get_n_silver_drude(k)
    x = k * r * n_background

    # Work out all the bessel functions
    psi_mx = bessel.get_psi(n, m * x)
    dpsi_mx = bessel.get_dpsi(n, m * x)
    xi_x = bessel.get_xi(n, x)
    dxi_x = bessel.get_dxi(n, x)

    return m * psi_mx * dxi_x - xi_x * dpsi_mx


# def get_da_denominator_dk(
#     k: complex,
#     r: float,
#     n_particle_func: Callable[[complex], complex],
#     n_background: complex,
#     n: int,
# ) -> tuple[complex, complex]:
#     """Get the an coefficient (from Mischenko) for a silver sphere with
#     wavenumber k and radius r"""
#     m = m_func(k)

#     # Get the refractive index first. The sphere is assumed to be in air
#     # m = get_n_silver_drude(k)
#     x = k * r

#     # Work out all the bessel functions
#     xi_x = bessel.get_xi(n, x)
#     psi_mx = bessel.get_psi(n, m * x)
#     dxi_x = bessel.get_dxi(n, x)
#     dpsi_mx = bessel.get_dpsi(n, m * x)
#     d2xi_x = bessel.get_d2xi(n, x)
#     d2psi_mx = bessel.get_d2psi(n, m * x)

#     term_one = r * (m**2 - 1) * dpsi_mx * dxi_x
#     term_two = r * m * (psi_mx * d2xi_x - xi_x * d2psi_mx)
#     return term_one + term_two


def get_da_denominator_dnb(
    k: complex,
    r: float,
    n_particle_func: Callable[[complex], complex],
    n_background: complex,
    n: int,
) -> tuple[complex, complex]:
    """Get the an coefficient (from Mischenko) for a silver sphere with
    wavenumber k and radius r"""
    n_particle = n_particle_func(k)
    m = n_particle / n_background

    # Get the refractive index first. The sphere is assumed to be in air
    # m = get_n_silver_drude(k)
    x = k * r * n_background

    # Work out all the bessel functions
    xi_x = bessel.get_xi(n, x)
    psi_mx = bessel.get_psi(n, m * x)
    dxi_x = bessel.get_dxi(n, x)
    dpsi_mx = bessel.get_dpsi(n, m * x)
    # d2xi_x = bessel.get_d2xi(n, x)
    d2psi_mx = bessel.get_d2psi(n, m * x)

    term_one = psi_mx * dxi_x * (-n_particle / n_background**2)
    term_two = m * dpsi_mx * dxi_x * (-x * n_particle / n_background**2)
    term_three = xi_x * d2psi_mx * (-x * n_particle / n_background**2)
    return term_one + term_two + term_three


# -----------------------------------------------------------------------------
# b coefficient




# -----------------------------------------------------------------------------
# Wigner Smith functions
# -----------------------------------------------------------------------------
def get_ws_a_dk(k: complex, r: float, m_func: complex, n: int) -> complex:
    fp = get_da_numerator_dk(k, r, m_func, n)
    f = get_a_numerator(k, r, m_func, n)
    gp = get_da_denominator_dk(k, r, m_func, n)
    g = get_a_denominator(k, r, m_func, n)
    ws = fp / f - gp / g
    return -1j * ws


def get_ws_a_dr(k: complex, r: float, m_func: complex, n: int) -> complex:
    fp = get_da_numerator_dr(k, r, m_func, n)
    f = get_a_numerator(k, r, m_func, n)
    gp = get_da_denominator_dr(k, r, m_func, n)
    g = get_a_denominator(k, r, m_func, n)
    ws = fp / f - gp / g
    return -1j * ws


def get_ws_a_dk_drude(
    k: complex, r: float, n: int, m_func: callable, dk: float = 1e-9
) -> complex:
    func_numerator = functools.partial(
        get_a_numerator, r=r, m_func=m_func, n=n
    )
    fp = utils.get_derivative_dk(func_numerator, k, dk)
    f = get_a_numerator(k, r, m_func, n)

    func_denominator = functools.partial(
        get_a_denominator, r=r, m_func=m_func, n=n
    )
    gp = utils.get_derivative_dk(func_denominator, k, dk)
    g = get_a_denominator(k, r, m_func, n)

    ws = fp / f - gp / g
    return -1j * ws


def get_ws_a_dnb(
    k: complex,
    r: float,
    n_particle_func: Callable[[complex], complex],
    n_background: complex,
    n: int,
) -> complex:
    fp = get_da_numerator_dnb(k, r, n_particle_func, n_background, n)
    f = get_a_numerator(k, r, n_particle_func, n_background, n)

    gp = get_da_denominator_dnb(k, r, n_particle_func, n_background, n)
    g = get_a_denominator(k, r, n_particle_func, n_background, n)

    ws = fp / f - gp / g
    return -1j * ws

def get_cross_section(
    k: complex,
    r: float,
    n_particle_func: Callable[[complex], complex],
    n_background: complex,
    max_multipole_order: int = 7,
) -> complex:
    """Recursive computation of a"""

    cs = 0
    for i in range(1, max_multipole_order + 1):
        new_a = get_a(k, r, n_particle_func, n_background, i)
        new_b = get_b(k, r, n_particle_func, n_background, i)
        cs += (2 * i + 1) * np.real(new_a + new_b)

    return cs * 2 * np.pi / k**2

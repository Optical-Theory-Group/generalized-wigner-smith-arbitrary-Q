"""Module for calculating various Bessel functions, derived functions
and their derivatives"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import scipy.optimize
import functools
import scipy.integrate
from tqdm import tqdm
from typing import Any
import quadpy

# -----------------------------------------------------------------------------
# Bessel functions and derivatives
# -----------------------------------------------------------------------------


def get_j(n: int, x: complex) -> complex:
    """Spherical Bessel function j_n"""
    return scipy.special.spherical_jn(n, x)


def get_dj(n: int, x: complex) -> complex:
    """Derivative of spherical Bessel function j'_n"""
    return scipy.special.spherical_jn(n, x, derivative=True)


def get_d2j(n: int, x: complex) -> complex:
    """Second derivative of spherical Bessel function j''

    To calculate j'' we use the recurrence relation
    j' = j_n-1 - (n+1)/z * j_n,
    for n=0
    j' = -j_1(x)

    which can be differentiated to give
    j'' = j'_n-1 + (n+1)/z^2 * j_n - (n+1)/z * j'_n
    for n=0
    j'' = -j'_1(x)
    """
    if n == 0:
        d2j = -get_dj(1, x)
    else:
        j = get_j(n, x)
        dj = get_dj(n, x)
        dj_minus = get_dj(n - 1, x)
        d2j = dj_minus + (n + 1) / x**2 * j - (n + 1) / x * dj
    return d2j


def get_y(n: int, x: complex) -> complex:
    """Spherical Bessel function y_n"""
    return scipy.special.spherical_yn(n, x)


def get_dy(n: int, x: complex) -> complex:
    """Derivative of spherical Bessel function y'_n"""
    return scipy.special.spherical_yn(n, x, derivative=True)


def get_d2y(n: int, x: complex) -> complex:
    """Second derivative of spherical Bessel function y''

    To calculate y'' we use the recurrence relation
    y' = y_n-1 - (n+1)/z * y_n,
    for n=0
    y' = -y_1(x)

    which can be differentiated to give
    y'' = y'_n-1 + (n+1)/z^2 * y_n - (n+1)/z * y'_n
    for n=0
    y'' = -y'_1(x)
    """
    if n == 0:
        d2y = -get_dy(1, x)
    else:
        y = get_y(n, x)
        dy = get_dy(n, x)
        dy_minus = get_dy(n - 1, x)
        d2y = dy_minus + (n + 1) / x**2 * y - (n + 1) / x * dy
    return d2y


def get_h(n: int, x: complex) -> complex:
    """Spherical Hankel function h = j + iy"""
    return get_j(n, x) + 1j * get_y(n, x)


def get_dh(n: int, x: complex) -> complex:
    """Derivative of spherical Hankel function h' = j' + iy'"""
    return get_dj(n, x) + 1j * get_dy(n, x)


def get_d2h(n: int, x: complex) -> complex:
    """Second derivative of spherical Hankel function h'' = j'' + iy''"""
    return get_d2j(n, x) + 1j * get_d2y(n, x)


def get_h2(n: int, x: complex) -> complex:
    """Spherical Hankel function h = j + iy"""
    return get_j(n, x) - 1j * get_y(n, x)


def get_dh2(n: int, x: complex) -> complex:
    """Derivative of spherical Hankel function h' = j' + iy'"""
    return get_dj(n, x) - 1j * get_dy(n, x)


def get_d2h2(n: int, x: complex) -> complex:
    """Second derivative of spherical Hankel function h'' = j'' + iy''"""
    return get_d2j(n, x) - 1j * get_d2y(n, x)


def get_psi(n: int, x: complex) -> complex:
    """psi(x) = x*j(x)"""
    return x * get_j(n, x)


def get_dpsi(n: int, x: complex) -> complex:
    """psi' = x*j' + j"""
    return x * get_dj(n, x) + get_j(n, x)


def get_d2psi(n: int, x: complex) -> complex:
    """psi'' = x*j'' + 2j'"""
    return x * get_d2j(n, x) + 2 * get_dj(n, x)


def get_xi(n: int, x: complex) -> complex:
    """xi(x) = x*h(x)"""
    return x * get_h(n, x)


def get_dxi(n: int, x: complex) -> complex:
    """xi' = x*h' + h"""
    return x * get_dh(n, x) + get_h(n, x)


def get_d2xi(n: int, x: complex) -> complex:
    """xi'' = x*h'' + 2h'"""
    return x * get_d2h(n, x) + 2 * get_dh(n, x)


def get_xi2(n: int, x: complex) -> complex:
    """xi(x) = x*h(x)"""
    return x * get_h2(n, x)


def get_dxi2(n: int, x: complex) -> complex:
    """xi' = x*h' + h"""
    return x * get_dh2(n, x) + get_h2(n, x)


def get_d2xi2(n: int, x: complex) -> complex:
    """xi'' = x*h'' + 2h'"""
    return x * get_d2h2(n, x) + 2 * get_dh2(n, x)


def get_chi(n: int, x: complex) -> complex:
    """chi(x) = -x*y(x)"""
    return -x * get_y(n, x)


def get_dchi(n: int, x: complex) -> complex:
    """chi' = -y - zy'"""
    return -get_y(n, x) - x * get_dy(n, x)


def get_d2chi(n: int, x: complex) -> complex:
    """chi'' = -x*y'' - 2y'"""
    return -x * get_d2y(n, x) - 2 * get_dy(n, x)

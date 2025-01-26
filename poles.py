import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import scipy.optimize
import functools
import scipy.integrate
from tqdm import tqdm
from typing import Any, Callable
import sphere
import quadpy
import coated_sphere
import multilayer

def find_pole_a_multilayer(
    pole_guess: complex,
    r_list: float,
    n_list: list[Callable[[complex], complex]],
    n: int,
) -> complex:
    def func(k_guess: np.ndarray) -> float:
        """Get the an coefficient (from Mischenko) for a silver sphere with
        wavenumber k and radius r"""
        k = k_guess[0] + 1j * k_guess[1]
        a = multilayer.get_a(k, r_list, n_list, n)
        return np.abs(1 / a)

    out = scipy.optimize.minimize(
        func,
        np.array([pole_guess.real, pole_guess.imag]),
        method="Nelder-Mead",
    )
    pole = out.x[0] + 1j * out.x[1]
    return pole

def find_zero_a_multilayer(
    zero_guess: complex,
    r_list: float,
    n_list: list[Callable[[complex], complex]],
    n: int,
) -> complex:
    def func(k_guess: np.ndarray) -> float:
        """Get the an coefficient (from Mischenko) for a silver sphere with
        wavenumber k and radius r"""
        k = k_guess[0] + 1j * k_guess[1]
        a = multilayer.get_a(k, r_list, n_list, n)
        return np.abs(a)

    out = scipy.optimize.minimize(
        func,
        np.array([zero_guess.real, zero_guess.imag]),
        method="Nelder-Mead",
    )
    zero = out.x[0] + 1j * out.x[1]
    return zero


def find_pole_cs_multilayer(
    pole_guess: complex,
    r_list: float,
    n_list: list[Callable[[complex], complex]],
) -> complex:
    def func(k_guess: np.ndarray) -> float:
        """Get the an coefficient (from Mischenko) for a silver sphere with
        wavenumber k and radius r"""
        k = k_guess[0] + 1j * k_guess[1]
        a = multilayer.get_cross_section(k, r_list, n_list)
        return np.abs(1 / a)

    out = scipy.optimize.minimize(
        func,
        np.array([pole_guess.real, pole_guess.imag]),
        method="Nelder-Mead",
    )
    pole = out.x[0] + 1j * out.x[1]
    return pole

def find_zero_cs_multilayer(
    zero_guess: complex,
    r_list: float,
    n_list: list[Callable[[complex], complex]],
) -> complex:
    def func(k_guess: np.ndarray) -> float:
        """Get the an coefficient (from Mischenko) for a silver sphere with
        wavenumber k and radius r"""
        k = k_guess[0] + 1j * k_guess[1]
        a = multilayer.get_cross_section(k, r_list, n_list)
        return np.abs(a)

    out = scipy.optimize.minimize(
        func,
        np.array([zero_guess.real, zero_guess.imag]),
        method="Nelder-Mead",
    )
    zero = out.x[0] + 1j * out.x[1]
    return zero


# Pole finding
def find_pole_a(
    pole_guess: complex,
    r: float,
    n_particle_func: Callable[[complex], complex],
    n_background: complex,
    n: int,
) -> complex:
    def func(k_guess: np.ndarray) -> float:
        """Get the an coefficient (from Mischenko) for a silver sphere with
        wavenumber k and radius r"""
        k = k_guess[0] + 1j * k_guess[1]
        a = sphere.get_a(k, r, n_particle_func, n_background, n)
        return np.abs(1 / a)

    out = scipy.optimize.minimize(
        func,
        np.array([pole_guess.real, pole_guess.imag]),
        method="Nelder-Mead",
    )
    pole = out.x[0] + 1j * out.x[1]
    return pole


def find_pole_a_coated(
    pole_guess: complex,
    r_core: float,
    r_coating: float,
    n_core_func: Callable[[complex], complex],
    n_coating_func: Callable[[complex], complex],
    n_background: complex,
    n: int,
) -> complex:
    def func(k_guess: np.ndarray) -> float:
        """Get the an coefficient (from Mischenko) for a silver sphere with
        wavenumber k and radius r"""
        k = k_guess[0] + 1j * k_guess[1]
        a = coated_sphere.get_a(
            k, r_core, r_coating, n_core_func, n_coating_func, n_background, n
        )
        return np.abs(1 / a)

    out = scipy.optimize.minimize(
        func,
        np.array([pole_guess.real, pole_guess.imag]),
        method="Nelder-Mead",
    )
    pole = out.x[0] + 1j * out.x[1]
    return pole


def find_pole_a_v2(
    pole_guess: complex, r: float, m_func: complex, n: int
) -> complex:
    def func(k_guess: np.ndarray) -> float:
        """Get the an coefficient (from Mischenko) for a silver sphere with
        wavenumber k and radius r"""
        k = k_guess[0] + 1j * k_guess[1]
        return np.abs(sphere.get_a_denominator(k, r, m_func, n))

    out = scipy.optimize.minimize(
        func,
        np.array([pole_guess.real, pole_guess.imag]),
        method="Nelder-Mead",
    )
    pole = out.x[0] + 1j * out.x[1]
    return pole





def get_residue(
    function,
    pole: complex,
    radius: float = 1e-1,
    scheme: Any = None,
    degree: int = 10,
) -> complex:
    """Find the pole of a function numerically using a contour integral.

    This is not particularly optimized, but should do the job. Make sure the
    radius is small enough so that only 1 pole is contained within."""

    # We convert the integral to polar coordinates
    # int f(z) dz = int f(pole + R*e^it) * i * R * e^it
    def integrand(t):
        arg = pole + radius * np.exp(1j * t)
        return 1j * radius * function(arg) * np.exp(1j * t)

    if scheme is None:
        scheme = quadpy.c1.gauss_legendre(degree)

    points = scheme.points
    weights = scheme.weights

    polar = (1.0 + points) * np.pi

    # Main loop for integral calculation
    integral = 0.0
    for point, weight in zip(polar, weights):
        value = integrand(point)
        integral += value * weight

    integral *= np.pi
    residue = integral / (2 * np.pi * 1j)

    return residue

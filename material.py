"""Module containing material properties, e.g. refractive indices"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.special
import scipy.optimize
import functools
import scipy.integrate
from tqdm import tqdm
from typing import Any
import quadpy

# Fundamental constants
C = 299792458
HBAR = 1.054572669125102e-34
MU0 = 1.25663706127e-6
EPS0 = 1 / (C * C * MU0)
EV = 1.602176634e-19


# Drude model functions
def get_eps_r_silver_drude(k: complex) -> complex:
    """Drude model for permittivity of silver

    Based on the paper
    Optical dielectric function of silver
    """

    w = k * C

    # Define empirical parameters. These come from the paper
    eps_inf = 5
    w_p = 8.9 * EV / HBAR
    gamma = 1 / (17e-15)

    eps_r = eps_inf - w_p**2 / (w**2 + 1j * w * gamma)
    return eps_r


def get_n_silver_drude(k: complex) -> complex:
    """Drude model for the refractive index of silver"""
    return np.sqrt(get_eps_r_silver_drude(k))


# Drude model functions
def get_eps_r_gold_drude_lorentz(k: complex) -> complex:
    """Drude model for permittivity of silver

    Based on the paper
    Optical dielectric function of silver
    """

    w = k * C

    # Define empirical parameters. These come from the paper
    # Optical properties of metallic films for vertical-cavity optoelectronic
    # devices
    f0 = 0.76
    f_array = np.array([0.024, 0.010, 0.071, 0.601, 4.384])
    g0 = 0.053 * EV / HBAR
    g_array = np.array([0.241, 0.345, 0.870, 2.494, 2.214]) * EV / HBAR
    w_array = np.array([0.415, 0.830, 2.969, 4.304, 13.32]) * EV / HBAR
    w_p = 9.03 * EV / HBAR

    # Free term
    # eps_inf = 9.84
    eps_r_free = 1 - f0 * w_p**2 / (w**2 + 1j * w * g0)

    # Bound term
    eps_r_bound = 0
    for fi, gi, wi in zip(f_array, g_array, w_array):
        eps_r_bound += fi * w_p**2 / (wi**2 - w**2 - 1j * w * gi)

    eps_r = eps_r_free + eps_r_bound
    return eps_r


# Drude model functions
def get_eps_r_gold_drude(k: complex) -> complex:
    """Drude model for permittivity of silver

    Based on the paper
    Optical dielectric function of silver
    """

    w = k * C

    # Define empirical parameters. These come from the paper
    eps_inf = 9.84
    w_p = 9.03 * EV / HBAR
    gamma = 1 / (9.3e-15)

    eps_r = eps_inf - w_p**2 / (w**2 + 1j * w * gamma)
    return eps_r


def get_n_gold_drude(k: complex) -> complex:
    """Drude model for the refractive index of silver"""
    return np.sqrt(get_eps_r_gold_drude(k))


def get_n_gold_drude_lorentz(k: complex) -> complex:
    """Drude model for the refractive index of silver"""
    return np.sqrt(get_eps_r_gold_drude_lorentz(k))


# def get_dn_silver_drude(k: complex) -> complex:
#     """Get the derivative of n"""
#     prefactor = 0.5 / get_n_silver_drude(k)

#     w = k * C

#     w_p = 8.9 * EV / HBAR
#     gamma = 1 / (17e-15)

#     deps = C * w_p**2 * (2 * w + 1j * gamma) / (w**2 + 1j * w * gamma) ** 2
#     return prefactor * deps


# def get_dn_silver_drude_numerical(k: complex, dk: float = 1e-5) -> complex:
#     """Get the derivative of n"""
#     k_low = k - dk / 2
#     k_high = k + dk / 2
#     f_low = get_n_silver_drude(k_low)
#     f_high = get_n_silver_drude(k_high)
#     return (f_high - f_low) / dk


def get_n_silica(k: complex) -> complex:
    """Refractive index of silica"""
    x = 2 * np.pi / k
    x *= 1e6
    n = np.lib.scimath.sqrt(
        1
        + 0.6961663 / (1 - (0.0684043 / x) ** 2)
        + 0.4079426 / (1 - (0.1162414 / x) ** 2)
        + 0.8974794 / (1 - (9.896161 / x) ** 2)
    )
    return n


def get_n_water() -> complex:
    data = np.genfromtxt("/home/nbyrnes/code/resonance/water_hale.txt")
    real = data[1:170, :]
    imag = data[171:, :]
    real[:, 0] *= 1e-6
    imag[:, 0] *= 1e-6
    real_spline = scipy.interpolate.CubicSpline(real[:, 0], real[:, 1])
    imag_spline = scipy.interpolate.CubicSpline(imag[:, 0], imag[:, 1])

    def n(k: complex) -> complex:
        l = 2 * np.pi / np.real(k)
        ri = real_spline(l) + 1j * imag_spline(l)
        return ri

    return n


def get_n_water_interpolate() -> complex:
    data = np.genfromtxt("/home/nbyrnes/code/resonance/water_hale.txt")
    real = data[1:170, :][:21]
    imag = data[171:, :][:21]
    comp = np.column_stack((real[:, 0] * 1e-6, real[:, 1] + 1j * imag[:, 1]))
    comp[:,0] = 2*np.pi/comp[:,0]
    func = np.polynomial.polynomial.Polynomial.fit(
        np.real(comp[:, 0]), np.real(comp[:, 1]), 3
    )
    return func


def get_n_water_sellmeier(k: complex) -> complex:
    """Refractive index of silica"""
    x = 2 * np.pi / k
    x *= 1e6
    n = np.lib.scimath.sqrt(
        1
        + 0.75831 / (1 - (np.sqrt(0.01007) / x) ** 2)
        + 0.08495 / (1 - (np.sqrt(8.91377) / x) ** 2)
    )
    return n


if __name__ == "__main__":
    n_data = get_n_water()
    n_sell = get_n_water_sellmeier
    n_int = get_n_water_interpolate()

    n_dats = []
    n_sells = []
    n_interp = []

    lams = np.linspace(0.2e-6, 1.7e-6, 10**3)
    for l in lams:
        k = 2 * np.pi / l
        n_dats.append(n_data(k))
        n_sells.append(n_sell(k))
        n_interp.append(n_int(k))

    fig, ax = plt.subplots()
    ax.plot(lams, n_dats)
    ax.plot(lams, n_interp)

"""Module containing functions for calculating scattering coefficients for
multilayer spheres"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import scipy.optimize
import functools
import scipy.integrate
from tqdm import tqdm
from typing import Any, Callable
import bessel

# -----------------------------------------------------------------------------
# a coefficient
# -----------------------------------------------------------------------------


def get_a(
    k: complex,
    r_list: list[float],
    n_func_list: list[Callable[[complex], complex]],
    multipole_order: int,
    verbose: bool = False,
) -> complex:
    """Recursive computation of a"""

    num_layers = len(r_list)

    num_n_funcs = len(n_func_list)
    if num_n_funcs - num_layers != 1:
        raise ValueError("Incorrect array lengths")

    # Base case
    if num_layers == 0:
        return 0

    r = r_list[-1]

    n_out = n_func_list[-1](k)
    k_out = k * n_out
    y = k_out * r

    n_in = n_func_list[-2](k)
    k_in = k * n_in
    x = k_in * r

    m = n_in / n_out

    psi_x = bessel.get_psi(multipole_order, x)
    dpsi_x = bessel.get_dpsi(multipole_order, x)
    psi_y = bessel.get_psi(multipole_order, y)
    dpsi_y = bessel.get_dpsi(multipole_order, y)
    xi_x = bessel.get_xi(multipole_order, x)
    dxi_x = bessel.get_dxi(multipole_order, x)
    xi_y = bessel.get_xi(multipole_order, y)
    dxi_y = bessel.get_dxi(multipole_order, y)
    a = m * xi_x * dpsi_y - dxi_x * psi_y
    b = m * psi_x * dpsi_y - dpsi_x * psi_y
    c = m * xi_x * dxi_y - dxi_x * xi_y
    d = m * psi_x * dxi_y - dpsi_x * xi_y

    # Reduce list lengths
    new_r_list = r_list[:-1]
    new_n_func_list = n_func_list[:-1]
    lower = get_a(k, new_r_list, new_n_func_list, multipole_order)
    upper = -(a * lower + b) / (c * lower + d)
    if verbose:
        print(f"lower: {lower}")
        print(f"upper: {upper}")
    return upper

def get_ainv(
    k: complex,
    r_list: list[float],
    n_func_list: list[Callable[[complex], complex]],
    multipole_order: int,
    verbose: bool = False,
) -> complex:
    return 1/(get_a(k, r_list, n_func_list, multipole_order, verbose))


def get_b(
    k: complex,
    r_list: list[float],
    n_func_list: list[Callable[[complex], complex]],
    multipole_order: int,
) -> complex:
    """Recursive computation of a"""

    num_layers = len(r_list)

    num_n_funcs = len(n_func_list)
    if num_n_funcs - num_layers != 1:
        raise ValueError("Incorrect array lengths")

    # Base case
    if num_layers == 0:
        return 0

    r = r_list[-1]

    n_out = n_func_list[-1](k)
    k_out = k * n_out
    y = k_out * r

    n_in = n_func_list[-2](k)
    k_in = k * n_in
    x = k_in * r

    m = n_out / n_in

    psi_x = bessel.get_psi(multipole_order, x)
    dpsi_x = bessel.get_dpsi(multipole_order, x)
    psi_y = bessel.get_psi(multipole_order, y)
    dpsi_y = bessel.get_dpsi(multipole_order, y)
    xi_x = bessel.get_xi(multipole_order, x)
    dxi_x = bessel.get_dxi(multipole_order, x)
    xi_y = bessel.get_xi(multipole_order, y)
    dxi_y = bessel.get_dxi(multipole_order, y)
    a = m * xi_x * dpsi_y - dxi_x * psi_y
    b = m * psi_x * dpsi_y - dpsi_x * psi_y
    c = m * xi_x * dxi_y - dxi_x * xi_y
    d = m * psi_x * dxi_y - dpsi_x * xi_y

    # Reduce list lengths
    new_r_list = r_list[:-1]
    new_n_func_list = n_func_list[:-1]
    lower = get_b(k, new_r_list, new_n_func_list, multipole_order)
    upper = -(a * lower + b) / (c * lower + d)

    return upper


def get_extinction_cross_section(
    k: complex,
    r_list: list[float],
    n_func_list: list[Callable[[complex], complex]],
) -> complex:
    """Recursive computation of a"""

    x = np.real(k) * n_func_list[-1](np.real(k)) * r_list[-1]
    x = np.real(x)
    stop_index = int(np.floor(x + 4.05 * x**0.33333 + 2.0) + 1)

    cs = 0
    for i in range(1, stop_index + 1):
        new_a = get_a(k, r_list, n_func_list, i)
        new_b = get_b(k, r_list, n_func_list, i)
        # cs += (2 * i + 1) * (np.abs(new_a) ** 2 + np.abs(new_b) ** 2)
        cs += (2 * i + 1) * np.real(new_a + new_b)
        # print(f"{i}: {(2 * i + 1) * np.real(new_a + new_b)}")
        # print(f"cs: {cs}")
    return - cs * 2 * np.pi / k**2

def get_scattering_cross_section(
    k: complex,
    r_list: list[float],
    n_func_list: list[Callable[[complex], complex]],
) -> complex:
    """Recursive computation of a"""

    x = np.real(k) * n_func_list[-1](np.real(k)) * r_list[-1]
    x = np.real(x)
    stop_index = int(np.floor(x + 4.05 * x**0.33333 + 2.0) + 1)

    cs = 0
    for i in range(1, stop_index + 1):
        new_a = get_a(k, r_list, n_func_list, i)
        new_b = get_b(k, r_list, n_func_list, i)
        cs += (2 * i + 1) * (np.abs(new_a) ** 2 + np.abs(new_b) ** 2)
        # print(f"{i}: {(2 * i + 1) * np.real(new_a + new_b)}")
        # print(f"cs: {cs}")
    return cs * 2 * np.pi / k**2



def get_da_dk(
    k: complex,
    r_list: list[float],
    n_func_list: list[Callable[[complex], complex]],
    multipole_order: int,
    dk: float = 1e-8,
) -> complex:
    """Calculate A_n, which is used in calculating the scattering coefficients
    of a coated sphere"""
    low = get_a(k - dk / 2, r_list, n_func_list, multipole_order)
    high = get_a(k + dk / 2, r_list, n_func_list, multipole_order)
    diff = (high - low) / dk
    return diff

def get_dainv_dk(
    k: complex,
    r_list: list[float],
    n_func_list: list[Callable[[complex], complex]],
    multipole_order: int,
    dk: float = 1e-8,
) -> complex:
    """Calculate A_n, which is used in calculating the scattering coefficients
    of a coated sphere"""
    low = get_ainv(k - dk / 2, r_list, n_func_list, multipole_order)
    high = get_ainv(k + dk / 2, r_list, n_func_list, multipole_order)
    diff = (high - low) / dk
    return diff


def get_da_dnb_bulk(
    k: complex,
    r_list: list[float],
    n_func_list: list[Callable[[complex], complex]],
    multipole_order: int,
    dnb: float = 1e-8,
) -> complex:
    """Calculate A_n, which is used in calculating the scattering coefficients
    of a coated sphere"""
    n_background_old = n_func_list[-1]
    def n_background_lower(k): return n_background_old(k) - dnb/2    
    def n_background_upper(k): return n_background_old(k) + dnb/2
    n_func_list_lower = n_func_list[:-1] + [n_background_lower]
    n_func_list_upper = n_func_list[:-1] + [n_background_upper]
    low = get_a(k, r_list, n_func_list_lower, multipole_order)
    high = get_a(k, r_list, n_func_list_upper, multipole_order)
    diff = (high - low) / dnb
    return diff

def get_dainv_dnb_bulk(
    k: complex,
    r_list: list[float],
    n_func_list: list[Callable[[complex], complex]],
    multipole_order: int,
    dnb: float = 1e-8,
) -> complex:
    """Calculate A_n, which is used in calculating the scattering coefficients
    of a coated sphere"""
    n_background_old = n_func_list[-1]
    def n_background_lower(k): return n_background_old(k) - dnb/2    
    def n_background_upper(k): return n_background_old(k) + dnb/2
    n_func_list_lower = n_func_list[:-1] + [n_background_lower]
    n_func_list_upper = n_func_list[:-1] + [n_background_upper]
    low = get_ainv(k, r_list, n_func_list_lower, multipole_order)
    high = get_ainv(k, r_list, n_func_list_upper, multipole_order)
    diff = (high - low) / dnb
    return diff


def get_da_dnb_surface(
    k: complex,
    r_list: list[float],
    n_func_list: list[Callable[[complex], complex]],
    multipole_order: int,
    dnb: float = 1e-8,
) -> complex:
    """Calculate A_n, which is used in calculating the scattering coefficients
    of a coated sphere"""
    n_background_old = n_func_list[-1]
    def n_layer_lower(k): return n_background_old(k) - dnb/2    
    def n_layer_upper(k): return n_background_old(k) + dnb/2
    n_func_list_lower = n_func_list[:-2] + [n_layer_lower, n_background_old]
    n_func_list_upper = n_func_list[:-2] + [n_layer_upper, n_background_old]
    low = get_a(k, r_list, n_func_list_lower, multipole_order)
    high = get_a(k, r_list, n_func_list_upper, multipole_order)
    diff = (high - low) / dnb
    return diff

def get_da_dr_surface(
    k: complex,
    r_list: list[float],
    n_func_list: list[Callable[[complex], complex]],
    multipole_order: int,
    dr: float = 1e-15,
) -> complex:
    """Calculate A_n, which is used in calculating the scattering coefficients
    of a coated sphere"""
    r_core, r_coating, r_layer = r_list
    r_list_upper = [r_core, r_coating, r_layer + dr/2]
    r_list_lower=  [r_core, r_coating, r_layer - dr/2]

    low = get_a(k, r_list_lower, n_func_list, multipole_order)
    high = get_a(k, r_list_upper, n_func_list, multipole_order)
    diff = (high - low) / dr
    return diff

def get_da_dr_outer(
    k: complex,
    r_list: list[float],
    n_func_list: list[Callable[[complex], complex]],
    multipole_order: int,
    dr: float = 1e-15,
) -> complex:
    """Calculate A_n, which is used in calculating the scattering coefficients
    of a coated sphere"""
    r_outer = r_list[-1]

    r_list_upper = r_list[:-1] + [r_outer + dr/2]
    r_list_lower = r_list[:-1] + [r_outer - dr/2]

    low = get_a(k, r_list_lower, n_func_list, multipole_order)
    high = get_a(k, r_list_upper, n_func_list, multipole_order)
    diff = (high - low) / dr
    return diff


def get_dainv_dr_surface(
    k: complex,
    r_list: list[float],
    n_func_list: list[Callable[[complex], complex]],
    multipole_order: int,
    dr: float = 1e-15,
) -> complex:
    """Calculate A_n, which is used in calculating the scattering coefficients
    of a coated sphere"""
    r_core, r_coating, r_layer = r_list
    r_list_upper = [r_core, r_coating, r_layer + dr/2]
    r_list_lower=  [r_core, r_coating, r_layer - dr/2]

    low = get_ainv(k, r_list_lower, n_func_list, multipole_order)
    high = get_ainv(k, r_list_upper, n_func_list, multipole_order)
    diff = (high - low) / dr
    return diff


def get_ws_a_k(
    k: complex,
    r_list: list[float],
    n_func_list: list[Callable[[complex], complex]],
    multipole_order: int,
    dk: float = 1e-4,
) -> complex:
    a = get_a(k, r_list, n_func_list, multipole_order)
    da_dk = get_da_dk(k, r_list, n_func_list, multipole_order, dk)
    ws = -1j * 1 / a * da_dk
    return ws

def get_ws_ainv_k(
    k: complex,
    r_list: list[float],
    n_func_list: list[Callable[[complex], complex]],
    multipole_order: int,
    dk: float = 1e-4,
) -> complex:
    a = get_ainv(k, r_list, n_func_list, multipole_order)
    da_dk = get_dainv_dk(k, r_list, n_func_list, multipole_order, dk)
    ws = -1j * 1 / a * da_dk
    return ws


def get_ws_a_nb_bulk(
    k: complex,
    r_list: list[float],
    n_func_list: list[Callable[[complex], complex]],
    multipole_order: int,
    dnb: float = 1e-8,
) -> complex:
    a = get_a(k, r_list, n_func_list, multipole_order)
    da_dnb = get_da_dnb_bulk(k, r_list, n_func_list, multipole_order, dnb)
    ws = -1j * 1 / a * da_dnb
    return ws

def get_ws_ainv_nb_bulk(
    k: complex,
    r_list: list[float],
    n_func_list: list[Callable[[complex], complex]],
    multipole_order: int,
    dnb: float = 1e-8,
) -> complex:
    a = get_ainv(k, r_list, n_func_list, multipole_order)
    da_dnb = get_dainv_dnb_bulk(k, r_list, n_func_list, multipole_order, dnb)
    ws = -1j * 1 / a * da_dnb
    return ws


def get_ws_a_nb_surface(
    k: complex,
    r_list: list[float],
    n_func_list: list[Callable[[complex], complex]],
    multipole_order: int,
    dnb: float = 1e-8,
) -> complex:
    a = get_a(k, r_list, n_func_list, multipole_order)
    da_dnb = get_da_dnb_surface(k, r_list, n_func_list, multipole_order, dnb)
    ws = -1j * 1 / a * da_dnb
    return ws

def get_ws_a_r_surface(
    k: complex,
    r_list: list[float],
    n_func_list: list[Callable[[complex], complex]],
    multipole_order: int,
    dr: float = 1e-15,
) -> complex:
    a = get_a(k, r_list, n_func_list, multipole_order)
    da_dr = get_da_dr_surface(k, r_list, n_func_list, multipole_order, dr)
    ws = -1j * 1 / a * da_dr
    return ws

def get_ws_a_r_outer(
    k: complex,
    r_list: list[float],
    n_func_list: list[Callable[[complex], complex]],
    multipole_order: int,
    dr: float = 1e-15,
) -> complex:
    a = get_a(k, r_list, n_func_list, multipole_order)
    da_dr = get_da_dr_outer(k, r_list, n_func_list, multipole_order, dr)
    ws = -1j * 1 / a * da_dr
    return ws


def get_ws_ainv_r_surface(
    k: complex,
    r_list: list[float],
    n_func_list: list[Callable[[complex], complex]],
    multipole_order: int,
    dr: float = 1e-15,
) -> complex:
    a = get_ainv(k, r_list, n_func_list, multipole_order)
    da_dr = get_dainv_dr_surface(k, r_list, n_func_list, multipole_order, dr)
    ws = -1j * 1 / a * da_dr
    return ws

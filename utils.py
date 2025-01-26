import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import scipy.optimize
import functools
import scipy.integrate
from tqdm import tqdm
from typing import Any
import quadpy

# Generic derivative function
def get_derivative_dk(func: callable, k: complex, dk: float = 1e-6) -> complex:
    """Get the derivative of n"""
    k_low = k - dk / 2
    k_high = k + dk / 2
    f_low = func(k_low)
    f_high = func(k_high)
    return (f_high - f_low) / dk
"""Module for purely random reward functions."""
from scipy.stats import norm, binom


def _random_reward():
    low_normal = norm(loc=2, scale=0.5)
    high_normal = norm(loc=5, scale=1)
    trial = binom.rvs(n=1, p=0.3)
    return trial * low_normal.rvs() + (~trial) * high_normal.rvs()

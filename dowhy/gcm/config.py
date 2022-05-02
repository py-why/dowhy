import os

import numpy as np

show_progress_bars = True
default_n_jobs = -1

# Smallest possible value. This is used in various algorithm for numerical stability.
EPS = os.getenv("DOWHY_EPSILON_VALUE", np.finfo(np.float64).eps)


def enable_progress_bars():
    global show_progress_bars
    show_progress_bars = True


def disable_progress_bars():
    global show_progress_bars
    show_progress_bars = False


def set_default_n_jobs(n_jobs: int) -> None:
    global default_n_jobs
    default_n_jobs = n_jobs

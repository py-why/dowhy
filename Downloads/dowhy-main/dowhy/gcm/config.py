show_progress_bars = True
default_n_jobs = -1


def enable_progress_bars():
    global show_progress_bars
    show_progress_bars = True


def disable_progress_bars():
    global show_progress_bars
    show_progress_bars = False


def set_default_n_jobs(n_jobs: int) -> None:
    global default_n_jobs
    default_n_jobs = n_jobs

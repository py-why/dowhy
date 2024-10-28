from datetime import datetime

import matplotlib.pyplot as plt

SMALL_SIZE = 8
MEDIUM_SIZE = 26
BIGGER_SIZE = 30


def plot_treatment_outcome(treatment, outcome, time_var):
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    fig, ax = plt.subplots()
    tline = ax.plot(time_var, treatment, "o", label="Treatment")
    oline = ax.plot(time_var, outcome, "r^", label="Outcome")

    ax.legend(loc="upper left", bbox_to_anchor=(1.04, 1))
    plt.xlabel("Time")
    fig.set_size_inches(8, 6)
    fig.savefig("obs_data" + datetime.now().strftime("%H-%M-%S") + ".png", bbox_inches="tight")


def plot_causal_effect(estimate, treatment, outcome):
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    fig, ax = plt.subplots()
    x_min = 0
    x_max = max(treatment)
    y_min = estimate.params["intercept"]
    y_max = y_min + estimate.value * (x_max - x_min)
    ax.scatter(treatment, outcome, c="gray", marker="o", label="Observed data")
    ax.plot([x_min, x_max], [y_min, y_max], c="black", ls="solid", lw=4, label="Causal variation")
    ax.set_ylim(0, max(outcome))
    ax.set_xlim(0, x_max)

    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    ax.text(
        10.8,
        1,
        r"DoWhy estimate $\rho$ (slope) = " + str(round(estimate.value, 2)),
        ha="right",
        va="bottom",
        size=20,
        bbox=bbox_props,
    )
    ax.legend(loc="upper left")
    plt.xlabel("Treatment")
    plt.ylabel("Outcome")

    fig.set_size_inches(8, 6)
    fig.savefig("effect" + datetime.now().strftime("%H-%M-%S") + ".png", bbox_inches="tight")

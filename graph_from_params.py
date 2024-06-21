import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import pathlib
from time import time

import pysysid.signal_generators as sg

import p_model


if __name__ == "__main__":
    data_path = pathlib.Path("comparison_data_ga_armax")

    data_file_exp = pathlib.Path("comparison_data_ga_armax/hybrid_20240429T170605.csv")

    df_exp = pd.read_csv(data_file_exp, sep=", ")

    t = np.array(df_exp["t"])[:-1]

    v_cmd_m = np.array(df_exp["cmd_voltage_motor"])[:-1]
    v_cmd_b = np.array(df_exp["cmd_voltage_brake"])[:-1]
    tau_u = np.array(df_exp["torque"])[:-1]

    i_m = np.array(df_exp["current_motor"])[:-1]
    i_b = np.array(df_exp["current_brake"])[:-1]
    omega = np.array(df_exp["omega"])[:-1]

    data_file_armax = pathlib.Path(
        "comparison_data_ga_armax/hybfit_armax_run_20240429T170605.csv"
    )

    df_armax = pd.read_csv(data_file_armax, sep=",")

    i_m_armax = np.array(df_armax["current_motor"])[:-1]
    i_b_armax = np.array(df_armax["current_brake"])[:-1]
    omega_armax = np.array(df_armax["omega"])[:-1]

    data_file_ga = pathlib.Path(
        "comparison_data_ga_armax/hybfit_ga_20240429T170605.csv"
    )

    df_ga = pd.read_csv(data_file_ga, sep=",")

    i_m_ga = np.array(df_ga["current_motor"])
    i_b_ga = np.array(df_ga["current_brake"])
    omega_ga = np.array(df_ga["omega"])

    # Plot predicted trajectories
    fig, ax = plt.subplots(3, 2, constrained_layout=True, sharex=True)

    ax[0][0].set_ylabel(r"$v_{m}(t)$ (V)")
    ax[0][0].plot(t, v_cmd_m, color="C0")

    ax[1][0].set_ylabel(r"$v_{b}(t)$ (V)")
    ax[1][0].plot(t, v_cmd_b, color="C0")

    ax[2][0].set_xlabel(r"$t$ (s)")
    ax[2][0].set_ylabel(r"$\tau_{d}(t)$ (Nm)")
    ax[2][0].plot(t, tau_u, color="C0")

    ax[0][1].set_ylabel(r"$i_{m}(t)$ (A)")
    ax[0][1].plot(t, i_m, label="Experimental Data", color="C0")

    ax[1][1].set_ylabel(r"$i_{b}(t)$ (A)")
    ax[1][1].plot(t, i_b, color="C0")

    ax[2][1].set_xlabel(r"$t$ (s)")
    ax[2][1].set_ylabel(r"$\omega(t)$ (rad/s)")
    ax[2][1].plot(t, omega, color="C0")

    # armax
    ax[0][1].plot(t, i_m_armax, "--", label="ARMAX", color="C1")
    ax[1][1].plot(t, i_b_armax, "--", color="C1")
    ax[2][1].plot(t, omega_armax, "--", color="C1")

    # ga
    ax[0][1].plot(t, i_m_ga, "--", label="GA", color="C2")
    ax[1][1].plot(t, i_b_ga, "--", color="C2")
    ax[2][1].plot(t, omega_ga, "--", color="C2")

    ax[0][1].legend(loc="upper right")
    for a in np.ravel(ax):
        a.grid(linestyle="--")

    # Plot errors
    fig, ax = plt.subplots(3, 1, constrained_layout=True, sharex=True)
    # armax
    ax[0].set_ylabel(r"$\Delta i_{m}(t)$ (A)")
    ax[0].plot(t, i_m - i_m_armax, label="ARMAX", color="C1")

    ax[1].set_ylabel(r"$\Delta i_{b}(t)$ (A)")
    ax[1].plot(t, i_b - i_b_armax, color="C1")

    ax[2].set_xlabel(r"$t$ (s)")
    ax[2].set_ylabel(r"$\Delta \omega(t)$ (rad/s)")
    ax[2].plot(t, omega - omega_armax, color="C1")

    # ga
    ax[0].plot(t, i_m - i_m_ga, label="GA", color="C2")
    ax[0].legend(loc="upper right")

    ax[1].plot(t, i_b - i_b_ga, color="C2")

    ax[2].plot(t, omega - omega_ga, color="C2")

    for a in np.ravel(ax):
        a.grid(linestyle="--")

    plt.show()

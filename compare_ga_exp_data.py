from typing import Dict
from time import time
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pysysid.genetic as genetic
import pysysid.pm2i as pm2i
import pysysid.signal_generators as sg


def _main():

    SHOW_ANG_VEL = True

    data_path = pathlib.Path("data/2024_04_22")

    data_files = list(pathlib.Path.glob(data_path, "hybrid_2*"))

    df_hybrid = pd.read_csv(data_files[-1], sep=", ")

    df_hybrid = df_hybrid.head(1000)

    t = np.array(df_hybrid["t"])

    v_cmd_m = np.array(df_hybrid["cmd_voltage_motor"])
    v_cmd_b = np.array(df_hybrid["cmd_voltage_brake"])
    tau_u = np.array(df_hybrid["torque"])

    i_m = np.array(df_hybrid["current_motor"])
    i_b = np.array(df_hybrid["current_brake"])
    theta = np.array(df_hybrid["theta"])
    omega = np.array(df_hybrid["omega"])

    fig, ax = plt.subplots(3, 2, constrained_layout=True, sharex=True)

    ax[0][0].set_xlabel(r"$t$ (s)")
    ax[0][0].set_ylabel(r"$v_{m}(t)$ (V)")
    ax[0][0].plot(t, v_cmd_m, label=r"Motor Voltage Command", color="C0")
    ax[0][0].legend(loc="upper right")

    ax[1][0].set_xlabel(r"$t$ (s)")
    ax[1][0].set_ylabel(r"$v_{b}(t)$ (V)")
    ax[1][0].plot(t, v_cmd_b, label=r"Brake Voltage Command", color="C0")
    ax[1][0].legend(loc="upper right")

    ax[2][0].set_xlabel(r"$t$ (s)")
    ax[2][0].set_ylabel(r"$\tau_{d}(t)$ (Nm)")
    ax[2][0].plot(t, tau_u, label=r"Load Torque", color="C0")
    ax[2][0].legend(loc="upper right")

    ax[0][1].set_xlabel(r"$t$ (s)")
    ax[0][1].set_ylabel(r"$i_{m}(t)$ (A)")
    ax[0][1].plot(t, i_m, label=r"Motor current", color="C0")
    ax[0][1].legend(loc="upper right")

    ax[1][1].set_xlabel(r"$t$ (s)")
    ax[1][1].set_ylabel(r"$i_{b}(t)$ (A)")
    ax[1][1].plot(t, i_b, label=r"Brake Current", color="C0")
    ax[1][1].legend(loc="upper right")

    if SHOW_ANG_VEL:
        ax[2][1].set_xlabel(r"$t$ (s)")
        ax[2][1].set_ylabel(r"$\omega(t)$ (rad/s)")
        ax[2][1].plot(t, omega, label=r"Angular Velocity", color="C0")
        ax[2][1].legend(loc="upper right")
    else:
        ax[2][1].set_xlabel(r"$t$ (s)")
        ax[2][1].set_ylabel(r"$\theta(t)$ (rad)")
        ax[2][1].plot(t, theta, label=r"Angular Position", color="C0")
        ax[2][1].legend(loc="upper right")

    data_files = list(pathlib.Path.glob(data_path, "hybfit_*"))

    df_fit = pd.read_csv(data_files[-1], sep=",")

    df_fit = df_fit.head(1000)

    t = np.array(df_fit["t"])

    v_cmd_m = np.array(df_fit["cmd_voltage_motor"])
    v_cmd_b = np.array(df_fit["cmd_voltage_brake"])
    tau_u = np.array(df_fit["torque"])

    i_m = np.array(df_fit["current_motor"])
    i_b = np.array(df_fit["current_brake"])
    theta = np.array(df_fit["theta"])
    omega = np.array(df_fit["omega"])

    ax[0][0].set_xlabel(r"$t$ (s)")
    ax[0][0].set_ylabel(r"$v_{m}(t)$ (V)")
    ax[0][0].plot(t, v_cmd_m, label=r"Id. Motor Voltage Command", color="C1")
    ax[0][0].legend(loc="upper right")

    ax[1][0].set_xlabel(r"$t$ (s)")
    ax[1][0].set_ylabel(r"$v_{b}(t)$ (V)")
    ax[1][0].plot(t, v_cmd_b, label=r"Id. Brake Voltage Command", color="C1")
    ax[1][0].legend(loc="upper right")

    ax[2][0].set_xlabel(r"$t$ (s)")
    ax[2][0].set_ylabel(r"$\tau_{d}(t)$ (Nm)")
    ax[2][0].plot(t, tau_u, label=r"Id. Load Torque", color="C1")
    ax[2][0].legend(loc="upper right")

    ax[0][1].set_xlabel(r"$t$ (s)")
    ax[0][1].set_ylabel(r"$i_{m}(t)$ (A)")
    ax[0][1].plot(t, i_m, label=r"Id.  Motor current", color="C1")
    ax[0][1].legend(loc="upper right")

    ax[1][1].set_xlabel(r"$t$ (s)")
    ax[1][1].set_ylabel(r"$i_{b}(t)$ (A)")
    ax[1][1].plot(t, i_b, label=r"Id. Brake Current", color="C1")
    ax[1][1].legend(loc="upper right")

    if SHOW_ANG_VEL:
        ax[2][1].set_xlabel(r"$t$ (s)")
        ax[2][1].set_ylabel(r"$\omega(t)$ (rad/s)")
        ax[2][1].plot(t, omega, label=r"Id. Angular Velocity", color="C1")
        ax[2][1].legend(loc="upper right")
    else:
        ax[2][1].set_xlabel(r"$t$ (s)")
        ax[2][1].set_ylabel(r"$\theta(t)$ (rad)")
        ax[2][1].plot(t, theta, label=r"Id. Angular Position", color="C1")
        ax[2][1].legend(loc="upper right")

    for a in np.ravel(ax):
        a.grid(linestyle="--")

    plt.show()


if __name__ == "__main__":
    _main()

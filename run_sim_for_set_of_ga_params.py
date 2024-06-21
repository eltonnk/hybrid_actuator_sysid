import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import pathlib
from time import time

import pysysid.signal_generators as sg

import p_model


if __name__ == "__main__":
    data_path = pathlib.Path("data/2024_06_04")

    data_files = list(pathlib.Path.glob(data_path, "ga_*"))

    df_params = pd.read_csv(data_files[-1], sep=",")

    dict_params_ga = df_params.iloc[[-1]].to_dict("list")

    dict_params_v2 = {}
    for key, val in dict_params_ga.items():
        dict_params_v2[key] = val[0]

    dict_params_ga = dict_params_v2

    FixedNHAPMG = p_model.generate_ha_pmg_fixed_brake_n(
        brake_n=2,
        brake_rho=10,
        gear_ratio_n=-1,
    )

    integration_method = "Radau"
    # Used to indicate to stop simulating dynamic responmse of chromosomes when
    # the parameters the chromosome define create an unstable system. Naturally the
    # unstable system takes more time to simulate and thus it is not necessary
    # to let scipy.integrate.solve_ivp compute indifinetly
    integration_timeout = 10000.0

    data_file_exp = pathlib.Path("comparison_data_ga_armax/hybrid_20240429T170605.csv")

    df_exp = pd.read_csv(data_file_exp, sep=", ")

    t = np.array(df_exp["t"])

    v_cmd_m = np.array(df_exp["cmd_voltage_motor"])
    v_cmd_b = np.array(df_exp["cmd_voltage_brake"])
    tau_u = np.array(df_exp["torque"])

    i_m = np.array(df_exp["current_motor"])
    i_b = np.array(df_exp["current_brake"])
    omega = np.array(df_exp["omega"])

    dt_data = t[1] - t[0]
    f_samp = 1 / dt_data
    t_end = t[-1]

    # None if continous, float if discrete
    motor_dt = None

    # v_m_sg = sg.PrbsGenerator(amplitude=0.75, offset=0.1, min_period=0.005, seed=4)
    # v_b_sg = sg.PrbsGenerator(amplitude=0.07, offset=0.07, min_period=0.05, seed=5)

    v_m_sg = sg.InterpolatedSignalGenerator(
        signal=v_cmd_m, f_samp=f_samp, time_arr=t, kind="zero"
    )

    v_b_sg = sg.InterpolatedSignalGenerator(
        signal=v_cmd_b, f_samp=f_samp, time_arr=t, kind="zero"
    )

    tau_u_sg = sg.InterpolatedSignalGenerator(
        signal=tau_u, f_samp=f_samp, time_arr=t, kind="quadratic"
    )

    input_gen = sg.InputGenerator([v_m_sg, v_b_sg, tau_u_sg])

    ha_pmg = FixedNHAPMG(dt=motor_dt, **dict_params_ga)

    x0 = [
        i_m[0],
        i_b[0],
        omega[0],
        0,  # np.power(
        #     ha_pmg.brake_A / (ha_pmg.brake_beta + ha_pmg.brake_gamma),
        #     1.0 / ha_pmg.brake_n,
        # ),
        (np.array(df_exp["theta"]))[0],
        0.0,
        0.0,
    ]

    ha_pm2i = ha_pmg.generate_process_model_to_integrate()

    start_time = time()

    sol_t, sol_u, sol_x, sol_y = ha_pm2i.integrate(
        compute_u_from_t=input_gen.value_at_t,
        dt_data=dt_data,
        t_end=t_end,
        x0=x0,
        method=integration_method,
        timeout=integration_timeout,
    )

    v_m_fit = sol_u[0, :]
    v_b_fit = sol_u[1, :]
    tau_u_fit = sol_u[2, :]

    i_m_fit = sol_y[0, :]
    i_b_fit = sol_y[1, :]
    omega_fit = sol_y[2, :]

    theta_fit = sol_x[4, :]

    dict_fit = {
        "t": sol_t,
        "cmd_voltage_motor": v_m_fit,
        "cmd_voltage_brake": v_b_fit,
        "current_motor": i_m_fit,
        "current_brake": i_b_fit,
        "torque": tau_u_fit,
        "theta": theta_fit,
        "omega": omega_fit,
    }

    df_fit = pd.DataFrame(dict_fit)

    output_file_name = data_file_exp.name.replace("hybrid", "hybfit_ga")

    df_fit.to_csv(pathlib.Path(f"data/{output_file_name}"), index=False)

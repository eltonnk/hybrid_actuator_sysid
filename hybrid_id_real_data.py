from time import time
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pysysid.genetic as genetic
import pysysid.signal_generators as sg

import p_model


def _main():
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
    integration_timeout = 500.0

    data_path = pathlib.Path("data")

    data_files = list(pathlib.Path.glob(data_path, "hybrid_*"))

    df_hybrid = pd.read_csv(data_files[-1], sep=", ")

    df_hybrid = df_hybrid.head(1000)

    t = np.array(df_hybrid["t"])

    v_cmd_m = np.array(df_hybrid["cmd_voltage_motor"])
    v_cmd_b = np.array(df_hybrid["cmd_voltage_brake"])
    tau_u = np.array(df_hybrid["torque"])

    i_m = np.array(df_hybrid["current_motor"])
    i_b = np.array(df_hybrid["current_brake"])
    omega = np.array(df_hybrid["omega"])

    f_samp = 800

    dt_data = 1 / f_samp
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

    fig, ax = plt.subplots(3, 2, constrained_layout=True, sharex=True)

    ax[0][0].set_ylabel(r"$v_{m}(t)$ (V)")
    ax[0][0].plot(t, v_cmd_m, label=r"Motor Voltage Command", color="C0")
    ax[0][0].legend(loc="upper right")

    ax[1][0].set_ylabel(r"$v_{b}(t)$ (V)")
    ax[1][0].plot(t, v_cmd_b, label=r"Brake Voltage Command", color="C0")
    ax[1][0].legend(loc="upper right")

    ax[2][0].set_xlabel(r"$t$ (s)")
    ax[2][0].set_ylabel(r"$\tau_{d}(t)$ (Nm)")
    ax[2][0].plot(t, tau_u, label=r"Load Torque", color="C0")
    ax[2][0].legend(loc="upper right")

    ax[0][1].set_ylabel(r"$i_{m}(t)$ (A)")
    ax[0][1].plot(t, i_m, label=r"Motor current", color="C0")
    ax[0][1].legend(loc="upper right")

    ax[1][1].set_ylabel(r"$i_{b}(t)$ (A)")
    ax[1][1].plot(t, i_b, label=r"Brake Current", color="C0")
    ax[1][1].legend(loc="upper right")

    ax[2][1].set_xlabel(r"$t$ (s)")
    ax[2][1].set_ylabel(r"$\omega(t)$ (rad/s)")
    ax[2][1].plot(t, omega, label=r"Angular Velocity", color="C0")
    ax[2][1].legend(loc="upper right")

    X = np.block(
        [
            [
                t.reshape(1, len(t)).T,
                v_cmd_m.reshape(1, len(t)).T,
                v_cmd_b.reshape(1, len(t)).T,
                tau_u.reshape(1, len(t)).T,
            ]
        ]
    )

    y = np.block(
        [
            [
                i_m.reshape(1, len(t)).T,
                i_b.reshape(1, len(t)).T,
                omega.reshape(1, len(t)).T,
            ]
        ]
    )

    # initial_estimate_motor_params = {
    #     "brake_alpha_0": 0.3,  # multiplied by hysteresis variable z that directly affects the torque, so lets give it a reasonable value
    #     "brake_alpha_1": 0.0005,  # same as above, multiplied by the current, so let's keep it lower
    #     "brake_c_0": 3e-4,  # lets set the brake's viscous damping to the twice the motor's value
    #     "brake_c_1": 3e-2,  # let's assume this is quite high, since the brake should be able to produce high torque with little energy
    #     "brake_R": 0.4,  # datasheet b6 (further refined by looking at current curves)
    #     "brake_L": 0.06,  # assuming the brake's electrical time constant is 3 times as big as the motor's electrical time constant
    #     "brake_A": 1,  # read that this parameter is superfluous and can be set to any values, other values in the hysteresis model wil adapt
    #     "brake_gamma": 1e-5,  # TODO: no clue
    #     "brake_beta": 1e1,  # TODO : no clue
    #     "brake_J": 2.6e-6,  # datasheet b6
    #     "motor_K": 0.14132959412,  # datasheet s23
    #     "motor_R": 1.18,  # datasheet s23
    #     "motor_L": 48e-4,  # taking a guess (further refined by looking at current curves)
    #     "motor_B": 1.534e-4,  # since mechanical time constant should be 100 times bigger than electrical time constant, after computation, this is the result
    #     "motor_J": 7.8e-6,  # lets guess 3 times the value of the brake
    #     "m_driver_K": 10.0,  # total guess
    #     "m_driver_w0": 400.0,  # total guess
    #     "b_driver_K": 12.0,  # total guess
    #     "b_driver_w0": 400.0,  # total guess
    # }

    # initial_estimate_motor_params = {  # taken from data/2024/02/17 final results
    #     "brake_alpha_0": 1.2339353154072616,
    #     "brake_alpha_1": 0.0010693706580100638,
    #     "brake_c_0": 0.0004190006356950035,
    #     "brake_c_1": 0.0732595810249624,
    #     "brake_R": 0.3955930327689322,
    #     "brake_L": 0.1068959139843728,
    #     "brake_A": 4.610578608300889,
    #     "brake_gamma": 1.7227481301816273e-05,
    #     "brake_beta": 10.305724441215693,
    #     "brake_J": 2.621242268913648e-06,
    #     "motor_K": 0.03732217072017984,  # removed minus sign here since K should not in reality be negative
    #     "motor_R": 3.4438397496107624,
    #     "motor_L": 0.0009122877775185453,
    #     "motor_B": 0.00031917518081070364,
    #     "motor_J": 1.1813857058653292e-05,
    #     "m_driver_K": 29.59490847582887,
    #     "m_driver_w0": 1046.5049874823897,
    #     "b_driver_K": 30.91307395296483,
    #     "b_driver_w0": 619.6931263293621,
    # }

    # initial_estimate_motor_params = {  # taken from data/2024/02/19 final results
    #     "brake_alpha_0": 5.059667665133325,
    #     "brake_alpha_1": 0.002141523317609087,
    #     "brake_c_0": 5.528604998845135e-05,  # removed minus sign here since K should not in reality be negative, not the end of the world tho as the overall viscous dampin is still positivie when taking into account motor_B
    #     "brake_c_1": 0.05044635522488383,
    #     "brake_R": 0.11891534817272101,
    #     "brake_L": 0.3315230514644945,
    #     "brake_A": 21.194444372727652,
    #     "brake_gamma": 2.1664753578513543e-05,
    #     "brake_beta": 6.628561482358766,
    #     "brake_J": 3.989540772182108e-07,
    #     "motor_K": 0.0597118146190081,
    #     "motor_R": 6.609190951178942,
    #     "motor_L": 0.0011858915701260524,
    #     "motor_B": 0.00024238248265826752,
    #     "motor_J": 8.193926035377125e-06,
    #     "m_driver_K": 112.31277860021181,
    #     "m_driver_w0": 1828.0202163462457,
    #     "b_driver_K": 68.57974389049103,
    #     "b_driver_w0": 86.06737959062832,
    # }

    initial_estimate_motor_params = {  # taken from data/2024/04/22 final results, no real improvement in data/2024/04/23
        "brake_alpha_0": 5.428588835220193,
        "brake_alpha_1": 0.31216617745322374,  # +2 order of magnitude
        "brake_c_0": 6.149313958976718e-05,
        "brake_c_1": 4.851956197499061,
        "brake_R": 0.23557436771200618e-04,  # reduced by 4 orders of manitude so that the velocity gets bigger
        "brake_L": 0.39630143516762095,
        "brake_A": 38.5371889457379,
        "brake_gamma": 8.4287303450129,
        "brake_beta": 3.937180406202165,
        "brake_J": 4.576149461086817e-08,
        "motor_K": 0.05750198166512045,
        "motor_R": 0.2323028766229216,  # removed minus sign
        "motor_L": 0.001905319604181519,
        "motor_B": 0.00030674931757235983,
        "motor_J": 7.781590715413564e-06,
        "m_driver_K": 248.1153642758801,
        "m_driver_w0": 3007.8410972798915,
        "b_driver_K": 90.02908602792685,
        "b_driver_w0": 62.5671510115732,
    }

    ha_pmg = FixedNHAPMG(dt=motor_dt, **initial_estimate_motor_params)

    x0 = [
        i_m[0],
        i_b[0],
        omega[0],
        0,  # np.power(
        #     ha_pmg.brake_A / (ha_pmg.brake_beta + ha_pmg.brake_gamma),
        #     1.0 / ha_pmg.brake_n,
        # ),
        (np.array(df_hybrid["theta"]))[0],
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

    v_cmd_m_sim = sol_u[0, :]
    v_cmd_b_sim = sol_u[1, :]
    tau_u_sim = sol_u[2, :]

    i_m_sim = sol_y[0, :]
    i_b_sim = sol_y[1, :]
    omega_sim = sol_y[2, :]

    z_sim = sol_x[3, :]

    sol_y_detailed = np.zeros(shape=(4, sol_t.shape[0]))

    for i, t in enumerate(sol_t):
        sol_y_detailed[:, i] = ha_pmg.compute_output_detailled(
            t, sol_x[:, i : i + 1], sol_u[:, i : i + 1]
        ).ravel()

    tau_m_elec_sim = sol_y_detailed[0, :]
    tau_m_damp_sim = sol_y_detailed[1, :]
    tau_b_hyst_sim = sol_y_detailed[2, :]
    tau_b_damp_sim = sol_y_detailed[3, :]

    ax[0][0].set_ylabel(r"$v_{m}(t)$ (V)")
    ax[0][0].plot(sol_t, v_cmd_m_sim, label=r"Sim. Motor Voltage Input", color="C2")
    ax[0][0].legend(loc="upper right")

    ax[1][0].set_ylabel(r"$v_{b}(t)$ (V)")
    ax[1][0].plot(sol_t, v_cmd_b_sim, label=r"Sim. Brake Voltage Input", color="C2")
    ax[1][0].legend(loc="upper right")

    ax[2][0].set_xlabel(r"$t$ (s)")
    ax[2][0].set_ylabel(r"$\tau_{d}(t)$ (Nm)")
    ax[2][0].plot(sol_t, tau_u_sim, label=r"Sim. Load Torque", color="C2")
    ax[2][0].legend(loc="upper right")

    ax[0][1].set_ylabel(r"$i_{m}(t)$ (A)")
    ax[0][1].plot(sol_t, i_m_sim, label=r"Sim. Motor current", color="C2")
    ax[0][1].legend(loc="upper right")

    ax[1][1].set_ylabel(r"$i_{b}(t)$ (A)")
    ax[1][1].plot(sol_t, i_b_sim, label=r"Sim. Brake Current", color="C2")
    ax[1][1].legend(loc="upper right")

    ax[2][1].set_xlabel(r"$t$ (s)")
    ax[2][1].set_ylabel(r"$\theta(t)$ (rad)")
    ax[2][1].plot(sol_t, omega_sim, label=r"Sim. Angular Velocity", color="C2")
    ax[2][1].legend(loc="upper right")

    print(f"Total example sim time: {time() - start_time}")

    fig2, ax2 = plt.subplots(2, 2, constrained_layout=True, sharex=True)

    ax2[0][0].set_xlabel(r"$t$ (s)")
    ax2[0][0].set_ylabel(r"$\tau_{m}(t)$ (N)")
    ax2[0][0].plot(sol_t, tau_m_elec_sim, label=r"Sim. Elec. Motor Torque", color="C2")
    ax2[0][0].legend(loc="upper right")

    ax2[0][1].set_xlabel(r"$t$ (s)")
    ax2[0][1].set_ylabel(r"$\tau_{m}(t)$ (N)")
    ax2[0][1].plot(sol_t, tau_m_damp_sim, label=r"Sim. Damp. Motor Torque", color="C2")
    ax2[0][1].legend(loc="upper right")

    ax2[1][0].set_xlabel(r"$t$ (s)")
    ax2[1][0].set_ylabel(r"$\tau_{m}(t)$ (N)")
    ax2[1][0].plot(sol_t, tau_b_hyst_sim, label=r"Sim. Hyst. Brake Torque", color="C2")
    ax2[1][0].legend(loc="upper right")

    ax2[1][1].set_xlabel(r"$t$ (s)")
    ax2[1][1].set_ylabel(r"$\tau_{m}(t)$ (N)")
    ax2[1][1].plot(sol_t, tau_b_damp_sim, label=r"Sim. Damp. Brake Torque", color="C2")
    ax2[1][1].legend(loc="upper right")

    # fig2, ax2 = plt.subplots(1, 1)

    # ax2.set_xlabel(r"$t$ (s)")
    # ax2.set_ylabel(r"$z(t)$ (V)")
    # ax2.plot(sol_t, z_sim, label=r"Hysteresis variable", color="C2")
    # ax2.legend(loc="upper right")

    plt.show()

    # default range value
    range_var = 0.7

    range_dict = {}
    for key, value in initial_estimate_motor_params.items():
        range_dict[key] = range_var

    # adjust here for parameters that we are pretty sure are close to the real value
    range_dict["brake_R"] = 0.9
    range_dict["brake_L"] = 0.9
    range_dict["motor_R"] = 0.9
    range_dict["motor_L"] = 0.9
    range_dict["m_driver_K"] = 0.9
    range_dict["m_driver_w0"] = 0.9
    range_dict["b_driver_K"] = 0.9
    range_dict["b_driver_w0"] = 0.9

    chromosome_parameter_ranges = {}
    for key, value in initial_estimate_motor_params.items():
        chromosome_parameter_ranges[key] = (
            (1 - range_dict[key]) * value,
            (1 + range_dict[key]) * value,
        )

    genetic_algo_regressor = genetic.Genetic(
        process_model=FixedNHAPMG,
        dt=motor_dt,
        compute_u_from_t=input_gen.value_at_t,
        n_chromosomes=200,
        replace_with_best_ratio=0.01,
        can_terminate_after_index=10,
        ratio_max_error_for_termination=0.2,
        seed=2,
        chromosome_parameter_ranges=chromosome_parameter_ranges,
        n_jobs=8,
        integration_method=integration_method,
        integration_timeout=integration_timeout,
        save_progess=True,
    )

    genetic_algo_regressor.fit(X, y, n_iter=50, x0=x0)

    best_fit_params = genetic_algo_regressor._elite_chromosome

    best_chromosome_motor_params = genetic_algo_regressor._gen_chromosome_dict(
        best_fit_params
    )

    print(f"{best_chromosome_motor_params=}")

    motor_fit = FixedNHAPMG(dt=motor_dt, **best_chromosome_motor_params)

    motor_pm2i_fit = motor_fit.generate_process_model_to_integrate()

    sol_t_fit, sol_u_fit, sol_x_fit, sol_y_fit = motor_pm2i_fit.integrate(
        compute_u_from_t=input_gen.value_at_t, dt_data=dt_data, t_end=t_end, x0=x0
    )

    v_m_fit = sol_u_fit[0, :]
    v_b_fit = sol_u_fit[1, :]
    tau_u_fit = sol_u_fit[2, :]

    i_m_fit = sol_y_fit[0, :]
    i_b_fit = sol_y_fit[1, :]
    omega_fit = sol_y_fit[2, :]

    theta_fit = sol_x_fit[4, :]

    dict_fit = {
        "t": sol_t_fit,
        "cmd_voltage_motor": v_m_fit,
        "cmd_voltage_brake": v_b_fit,
        "current_motor": i_m_fit,
        "current_brake": i_b_fit,
        "torque": tau_u_fit,
        "theta": theta_fit,
        "omega": omega_fit,
    }

    df_fit = pd.DataFrame(dict_fit)

    output_file_name = data_files[-1].name.replace("hybrid", "hybfit")

    df_fit.to_csv(pathlib.Path(f"data/{output_file_name}"), index=False)

    ax[0][0].set_ylabel(r"$v_{m}(t)$ (V)")
    ax[0][0].plot(
        sol_t_fit, v_m_fit, label=r"Motor Voltage Input - Best Fit", color="C1"
    )
    ax[0][0].legend(loc="upper right")

    ax[1][0].set_ylabel(r"$v_{b}(t)$ (V)")
    ax[1][0].plot(
        sol_t_fit, v_b_fit, label=r"Brake Voltage Input - Best Fit", color="C1"
    )
    ax[1][0].legend(loc="upper right")

    ax[2][0].set_xlabel(r"$t$ (s)")
    ax[2][0].set_ylabel(r"$\tau_{d}(t)$ (Nm)")
    ax[2][0].plot(sol_t_fit, tau_u_fit, label=r"Load Torque - Best Fit", color="C1")
    ax[2][0].legend(loc="upper right")

    ax[0][1].set_ylabel(r"$i_{m}(t)$ (A)")
    ax[0][1].plot(sol_t_fit, i_m_fit, label=r"Motor current - Best Fit", color="C1")
    ax[0][1].legend(loc="upper right")

    ax[1][1].set_ylabel(r"$i_{b}(t)$ (A)")
    ax[1][1].plot(sol_t_fit, i_b_fit, label=r"Brake Current - Best Fit", color="C1")
    ax[1][1].legend(loc="upper right")

    ax[2][1].set_xlabel(r"$t$ (s)")
    ax[2][1].set_ylabel(r"$\omega(t)$ (rad)")
    ax[2][1].plot(
        sol_t_fit, omega_fit, label=r"Angular Velocity - Best Fit", color="C1"
    )
    ax[2][1].legend(loc="upper right")

    plt.show()

    original_params = np.array(list(initial_estimate_motor_params.values()))


if __name__ == "__main__":
    _main()

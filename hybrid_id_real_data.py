from typing import Dict
from time import time
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pysysid.genetic as genetic
import pysysid.pm2i as pm2i
import pysysid.signal_generators as sg


class HybridActuatorPMG(pm2i.ProcessModelGenerator):
    def __init__(self, dt: float = None, **kwargs: Dict[str, float]):
        super().__init__(dt, **kwargs)
        self.brake_alpha_0 = self.params["brake_alpha_0"]
        self.brake_alpha_1 = self.params["brake_alpha_1"]

        self.brake_c_0 = self.params["brake_c_0"]
        self.brake_c_1 = self.params["brake_c_1"]

        self.brake_R = self.params["brake_R"]
        self.brake_L = self.params["brake_L"]

        self.brake_A = self.params["brake_A"]
        self.brake_gamma = self.params["brake_gamma"]
        self.brake_beta = self.params["brake_beta"]

        self.brake_J = self.params["brake_J"]

        self.brake_n = self.params["brake_n"]

        self.brake_rho = self.params["brake_rho"]

        self.motor_K = self.params["motor_K"]

        self.motor_R = self.params["motor_R"]
        self.motor_L = self.params["motor_L"]

        self.motor_B = self.params["motor_B"]
        self.motor_J = self.params["motor_J"]

        self.gear_ratio_n = self.params["gear_ratio_n"]

        self.m_driver_K = self.params["m_driver_K"]
        self.m_driver_w0 = self.params["m_driver_w0"]

        self.b_driver_K = self.params["b_driver_K"]
        self.b_driver_w0 = self.params["b_driver_w0"]

        # process model constants
        self.l11 = -self.motor_R / self.motor_L
        self.l12 = self.gear_ratio_n * self.motor_K / self.motor_L
        self.l13 = 1.0 / self.motor_L

        self.l21 = -self.brake_R / self.brake_L
        self.l22 = 1.0 / self.brake_L

        self.J = self.gear_ratio_n**2 * self.motor_J + self.brake_J
        self.l31 = -self.gear_ratio_n * self.motor_K / self.J
        self.l32 = -self.gear_ratio_n**2 * self.motor_B / self.J

        self.alpha_1_R_b = self.brake_alpha_1 * self.brake_R
        self.c_1_R_b = self.brake_c_1 * self.brake_R

        self.l33 = -self.brake_alpha_0 / self.J
        self.l34 = -self.alpha_1_R_b / self.J
        self.l37 = -self.brake_c_0 / self.J
        self.l38 = -self.c_1_R_b / self.J

        self.l39 = 1.0 / self.J

        self.brake_n_1 = self.brake_n - 1

        driver_voltage_cmd_to_current_cmd_ratio = 0.5

        self.l61 = -self.m_driver_w0
        self.l62 = -self.m_driver_K * self.m_driver_w0
        self.l63 = -driver_voltage_cmd_to_current_cmd_ratio * self.l62

        self.l71 = -self.b_driver_w0
        self.l72 = -self.b_driver_K * self.b_driver_w0
        self.l73 = -driver_voltage_cmd_to_current_cmd_ratio * self.l72

    def check_valid_brake_voltage(self, v_cmd_b: float):
        if v_cmd_b < 0:
            raise ValueError("Brake nput voltage should always be positive.")

    def _omega_dot(self, total_state: np.ndarray, tau_u: float) -> float:
        # hybrid actuator process model
        i_m = total_state[0, 0]
        i_b = total_state[1, 0]
        omega = total_state[2, 0]
        z = total_state[3, 0]

        return (
            self.l31 * i_m
            + self.l32 * omega
            + (self.l33 + self.l34 * i_b) * z
            + (self.l37 + self.l38 * i_b) * omega
            + self.l39 * tau_u
        )

    def compute_state_derivative(
        self, t: float, total_state: np.ndarray, total_input: np.ndarray
    ) -> np.ndarray:
        # hybrid actuator process model
        i_m = total_state[0, 0]
        i_b = total_state[1, 0]
        omega = total_state[2, 0]
        z = total_state[3, 0]
        # theta
        v_m = total_state[5, 0]
        v_b = total_state[6, 0]

        # Hybrid actuator inputs
        v_cmd_m = total_input[0, 0]
        v_cmd_b = total_input[1, 0]
        # Hand inputs
        tau_u = total_input[2, 0]

        self.check_valid_brake_voltage(v_cmd_b)

        abs_z = np.abs(z)
        abs_z_n_1 = np.power(abs_z, self.brake_n_1)
        abs_z_n = abs_z_n_1 * abs_z

        return np.array(
            [
                [self.l11 * i_m + self.l12 * omega + self.l13 * v_m],
                [self.l21 * i_b + self.l22 * v_b],
                [self._omega_dot(total_state, tau_u)],
                [
                    -self.brake_gamma * np.abs(omega) * z * abs_z_n_1
                    - self.brake_beta * omega * abs_z_n
                    + self.brake_A * omega
                ],
                [omega],
                [self.l61 * v_m + self.l62 * i_m + self.l63 * v_cmd_m],
                [self.l71 * v_b + self.l72 * i_b + self.l73 * v_cmd_b],
            ]
        )

    def _alpha(self, i_b: float) -> float:
        return self.brake_alpha_0 + self.alpha_1_R_b * i_b

    def _c(self, i_b: float) -> float:
        return self.brake_c_0 + self.c_1_R_b * i_b

    def compute_output(
        self, t: float, total_state: np.ndarray, total_input: np.ndarray
    ) -> np.ndarray:
        # hybrid actuator process model
        i_m = total_state[0, 0]
        i_b = total_state[1, 0]
        theta = total_state[4, 0]

        return np.array(
            [
                [i_m],
                [i_b],
                [theta],
            ]
        )

    def param_inequality_constraint(params: Dict[str, float]) -> np.ndarray:

        h_x = []
        # all params should be positive. since inequality constraint should be
        # of the form h(x) <= 0, we multiply every constraint by -1
        for key, val in params.items():
            # only exception is for beta and gamma, where we have beta + gamma >= 0,
            # and gamma - beta >= 0. thus,
            if key == "brake_beta":
                continue
            if key == "brake_gamma":
                continue
            h_x.append(-1.0 * val)

        h_x.append(-params["brake_gamma"] - params["brake_beta"])
        h_x.append(-params["brake_gamma"] + params["brake_beta"])

        return np.array(h_x)

    def _h1(self, z: float, omega: float) -> float:
        return self.brake_A - np.power(np.tanh(self.brake_rho * z), self.brake_n) * (
            self.brake_beta
            + self.brake_gamma * np.tanh(self.brake_rho * omega * z)
            - self.brake_gamma
            * self.brake_rho
            * omega
            * z
            * (1 - np.power(np.tanh(self.brake_rho * omega * z), 2))
        )

    def _h2(self, z: float, omega: float) -> float:
        return -self.brake_n * np.power(
            np.tanh(self.brake_rho * z), self.brake_n - 1
        ) * (
            self.brake_rho * z * (1 - np.power(np.tanh(self.brake_rho * z), 2))
            + np.tanh(self.brake_rho * z)
        ) * (
            self.brake_beta + self.brake_gamma * np.tanh(self.brake_rho * omega * z)
        ) - self.brake_gamma * self.brake_rho * omega * np.power(
            np.tanh(self.brake_rho * z), self.brake_n
        ) * (
            1 - np.power(np.tanh(self.brake_rho * omega * z), 2)
        )

    def compute_df_dx(self, t: float, total_state: np.ndarray, total_input: np.ndarray):
        # i_m = total_state[0]
        i_b = total_state[1]
        omega = total_state[2]
        z = total_state[3]
        # theta = total_state[4]
        # v_m = total_state[5]
        # v_b = total_state[6]

        return np.array(
            [
                [self.l11, 0, self.l12, 0, 0, self.l13, 0],
                [0, self.l21, 0, 0, 0, 0, self.l22],
                [
                    -self.gear_ratio_n * self.motor_K / self.J,
                    -(self.brake_alpha_1 * z + self.brake_c_1 * omega) / self.J,
                    -(self.gear_ratio_n**2 * self.motor_B + self._c(i_b)) / self.J,
                    -self._alpha(i_b) / self.J,
                    0,
                    0,
                    0,
                ],
                [0, 0, self._h1(z, omega), self._h2(z, omega), 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [self.l62, 0, 0, 0, 0, self.l61, 0],
                [0, self.l72, 0, 0, 0, 0, self.l71],
            ]
        )

    def generate_process_model_to_integrate(
        self,
    ) -> pm2i.ProcessModelToIntegrate:
        ha_pm2i = pm2i.ProcessModelToIntegrate(
            nbr_states=7,
            nbr_inputs=3,
            nbr_outputs=3,
            fct_for_x_dot=self.compute_state_derivative,
            fct_for_y=self.compute_output,
            df_dx=self.compute_df_dx,
        )

        return ha_pm2i


def generate_ha_pmg_fixed_brake_n(
    brake_n: int,
    brake_rho: float,
    gear_ratio_n: float,
) -> type[HybridActuatorPMG]:
    """Generates a class derived from HybridActuatorPMG, where the parameter n
    has a predetermined value. Makes it so its not necessary to specify value for
    key "brake_n" in kwargs when calling the derived class' constructor.

    This function should be used to try system id for different values of "brake_n",
    which should be a positive integer value. Since the GA can only do exploration over
    floating point values, and not integers, it is necessary to run the GA on
    experimental data multiple times, once for every possible value of "brake_n".
    Since "brake_n" is an exponent, its value can't be enormous, so this process
    should not be too time consuming, since the possible values should be smaller
    than 100, if not 10.


    Parameters
    ----------
    brake_n : int
        Exponent used in the hysteresis equation of the hybrid actuator model.

    Returns
    -------
    type[HybridActuatorPMG]
        Class derived from the HybridActuatorPMG class. Will be used by the GA
        to simulate an hybrid actuator response, with a predetermined brake_n
        parameter value.
    """

    class FixedNHAPMG(HybridActuatorPMG):
        def __init__(self, dt: float = None, **kwargs: Dict[str, float]):
            kwargs["brake_n"] = brake_n
            kwargs["brake_rho"] = brake_rho
            kwargs["gear_ratio_n"] = gear_ratio_n
            super().__init__(dt, **kwargs)

    return FixedNHAPMG


def _main():
    FixedNHAPMG = generate_ha_pmg_fixed_brake_n(
        brake_n=2,
        brake_rho=10,
        gear_ratio_n=-1,
    )

    integration_method = "Radau"
    # Used to indicate to stop simulating dynamic responmse of chromosomes when
    # the parameters the chromosome define create an unstable system. Naturally the
    # unstable system takes more time to simulate and thus it is not necessary
    # to let scipy.integrate.solve_ivp compute indifinetly
    integration_timeout = 100.0

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
    theta = np.array(df_hybrid["theta"])

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

    fig, ax = plt.subplots(3, 2)

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

    ax[2][1].set_xlabel(r"$t$ (s)")
    ax[2][1].set_ylabel(r"$\theta(t)$ (rad)")
    ax[2][1].plot(t, theta, label=r"Angular Position", color="C0")
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
                theta.reshape(1, len(t)).T,
            ]
        ]
    )

    x0 = [
        i_m[0],
        i_b[0],
        (np.array(df_hybrid["omega"]))[0],
        theta[0],
        0.0,  # could do better approximation of z maybe
        0.0,
        0.0,
    ]

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

    initial_estimate_motor_params = {  # taken from data/2024/02/17 final results
        "brake_alpha_0": 1.2339353154072616,
        "brake_alpha_1": 0.0010693706580100638,
        "brake_c_0": 0.0004190006356950035,
        "brake_c_1": 0.0732595810249624,
        "brake_R": 0.3955930327689322,
        "brake_L": 0.1068959139843728,
        "brake_A": 4.610578608300889,
        "brake_gamma": 1.7227481301816273e-05,
        "brake_beta": 10.305724441215693,
        "brake_J": 2.621242268913648e-06,
        "motor_K": 0.03732217072017984,  # removed minus sign here since K should not in reality be negative
        "motor_R": 3.4438397496107624,
        "motor_L": 0.0009122877775185453,
        "motor_B": 0.00031917518081070364,
        "motor_J": 1.1813857058653292e-05,
        "m_driver_K": 29.59490847582887,
        "m_driver_w0": 1046.5049874823897,
        "b_driver_K": 30.91307395296483,
        "b_driver_w0": 619.6931263293621,
    }

    ha_pmg = FixedNHAPMG(dt=motor_dt, **initial_estimate_motor_params)

    ha_pm2i = ha_pmg.generate_process_model_to_integrate()

    start_time = time()

    sol_t, sol_u, _, sol_y = ha_pm2i.integrate(
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
    theta_sim = sol_y[2, :]

    ax[0][0].set_xlabel(r"$t$ (s)")
    ax[0][0].set_ylabel(r"$v_{m}(t)$ (V)")
    ax[0][0].plot(sol_t, v_cmd_m_sim, label=r"Sim. Motor Voltage Input", color="C2")
    ax[0][0].legend(loc="upper right")

    ax[1][0].set_xlabel(r"$t$ (s)")
    ax[1][0].set_ylabel(r"$v_{b}(t)$ (V)")
    ax[1][0].plot(sol_t, v_cmd_b_sim, label=r"Sim. Brake Voltage Input", color="C2")
    ax[1][0].legend(loc="upper right")

    ax[2][0].set_xlabel(r"$t$ (s)")
    ax[2][0].set_ylabel(r"$\tau_{d}(t)$ (Nm)")
    ax[2][0].plot(sol_t, tau_u_sim, label=r"Sim. Load Torque", color="C2")
    ax[2][0].legend(loc="upper right")

    ax[0][1].set_xlabel(r"$t$ (s)")
    ax[0][1].set_ylabel(r"$i_{m}(t)$ (A)")
    ax[0][1].plot(sol_t, i_m_sim, label=r"Sim. Motor current", color="C2")
    ax[0][1].legend(loc="upper right")

    ax[1][1].set_xlabel(r"$t$ (s)")
    ax[1][1].set_ylabel(r"$i_{b}(t)$ (A)")
    ax[1][1].plot(sol_t, i_b_sim, label=r"Sim. Brake Current", color="C2")
    ax[1][1].legend(loc="upper right")

    ax[2][1].set_xlabel(r"$t$ (s)")
    ax[2][1].set_ylabel(r"$\theta(t)$ (rad)")
    ax[2][1].plot(sol_t, theta_sim, label=r"Sim. Angular Position", color="C2")
    ax[2][1].legend(loc="upper right")

    print(f"Total example sim time: {time() - start_time}")

    # plt.show()

    chromosome_parameter_ranges = {}

    range_var = 0.9
    for key, value in initial_estimate_motor_params.items():
        chromosome_parameter_ranges[key] = (
            (1 - range_var) * value,
            (1 + range_var) * value,
        )

    genetic_algo_regressor = genetic.Genetic(
        process_model=FixedNHAPMG,
        dt=motor_dt,
        compute_u_from_t=input_gen.value_at_t,
        n_chromosomes=100,
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

    genetic_algo_regressor.fit(X, y, n_iter=40, x0=x0)

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
    theta_fit = sol_y_fit[2, :]

    omega_fit = sol_x_fit[2, :]

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

    ax[0][0].set_xlabel(r"$t$ (s)")
    ax[0][0].set_ylabel(r"$v_{m}(t)$ (V)")
    ax[0][0].plot(
        sol_t_fit, v_m_fit, label=r"Motor Voltage Input - Best Fit", color="C1"
    )
    ax[0][0].legend(loc="upper right")

    ax[1][0].set_xlabel(r"$t$ (s)")
    ax[1][0].set_ylabel(r"$v_{b}(t)$ (V)")
    ax[1][0].plot(
        sol_t_fit, v_b_fit, label=r"Brake Voltage Input - Best Fit", color="C1"
    )
    ax[1][0].legend(loc="upper right")

    ax[2][0].set_xlabel(r"$t$ (s)")
    ax[2][0].set_ylabel(r"$\tau_{d}(t)$ (Nm)")
    ax[2][0].plot(sol_t_fit, tau_u_fit, label=r"Load Torque - Best Fit", color="C1")
    ax[2][0].legend(loc="upper right")

    ax[0][1].set_xlabel(r"$t$ (s)")
    ax[0][1].set_ylabel(r"$i_{m}(t)$ (A)")
    ax[0][1].plot(sol_t_fit, i_m_fit, label=r"Motor current - Best Fit", color="C1")
    ax[0][1].legend(loc="upper right")

    ax[1][1].set_xlabel(r"$t$ (s)")
    ax[1][1].set_ylabel(r"$i_{b}(t)$ (A)")
    ax[1][1].plot(sol_t_fit, i_b_fit, label=r"Brake Current - Best Fit", color="C1")
    ax[1][1].legend(loc="upper right")

    ax[2][1].set_xlabel(r"$t$ (s)")
    ax[2][1].set_ylabel(r"$\theta(t)$ (rad)")
    ax[2][1].plot(
        sol_t_fit, theta_fit, label=r"Angular Position - Best Fit", color="C1"
    )
    ax[2][1].legend(loc="upper right")

    plt.show()

    original_params = np.array(list(initial_estimate_motor_params.values()))


if __name__ == "__main__":
    _main()

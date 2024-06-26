from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

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
    brake_n: int, brake_rho: float
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
            super().__init__(dt, **kwargs)

    return FixedNHAPMG


def _main():
    motor_params = {
        "brake_alpha_0": 3.1428,
        "brake_alpha_1": 0.000576362,
        "brake_c_0": 0.3347,
        "brake_c_1": 0.5919,
        "brake_R": 1.0,
        "brake_L": 0.01573338829,
        "brake_A": 12.1487,
        "brake_gamma": 2.4246e-5,
        "brake_beta": 15.6237,
        "brake_J": 12e-7,
        "motor_K": 2.38e-2,
        "motor_R": 1.0,
        "motor_L": 6e-4,
        "motor_B": 1.33e-5,
        "motor_J": 8e-7,
        "gear_ratio_n": 20,
        "m_driver_K": 1.0,
        "m_driver_w0": 400.0,
        "b_driver_K": 1.0,
        "b_driver_w0": 400.0,
    }

    integration_method = "Radau"
    # Used to indicate to stop simulating dynamic responmse of chromosomes when
    # the parameters the chromosome define create an unstable system. Naturally the
    # unstable system takes more time to simulate and thus it is not necessary
    # to let scipy.integrate.solve_ivp compute indifinetly
    integration_timeout = 150.0

    f_samp = 800

    dt_data = 1 / f_samp
    t_end = 4

    # None if continous, float if discrete
    motor_dt = None
    x0 = None

    # v_m_sg = sg.SquareGenerator(period=1, pulse_width=0.5, amplitude=1)

    # v_b_sg = sg.SquareGenerator(period=1.5, pulse_width=0.5, amplitude=12, offset=12)

    v_m_sg = sg.PrbsGenerator(amplitude=2, offset=0.0, min_period=0.01, seed=4)
    v_b_sg = sg.PrbsGenerator(amplitude=24, offset=24, min_period=0.05, seed=5)

    tau_d_sg = sg.SineGenerator(frequency=0.3, amplitude=0.01, phase=0)

    input_gen = sg.InputGenerator([v_m_sg, v_b_sg, tau_d_sg])

    FixedNHAPMG = generate_ha_pmg_fixed_brake_n(brake_n=2, brake_rho=10)

    ha_pmg = FixedNHAPMG(dt=motor_dt, **motor_params)

    ha_pm2i = ha_pmg.generate_process_model_to_integrate()

    sol_t, sol_u, _, sol_y = ha_pm2i.integrate(
        compute_u_from_t=input_gen.value_at_t,
        dt_data=dt_data,
        t_end=t_end,
        x0=x0,
        method=integration_method,
        timeout=integration_timeout,
    )

    fig, ax = plt.subplots(3, 2)

    v_m = sol_u[0, :]
    v_b = sol_u[1, :]
    tau_d = sol_u[2, :]

    i_m = sol_y[0, :]
    i_b = sol_y[1, :]
    theta = sol_y[2, :]

    ax[0][0].set_xlabel(r"$t$ (s)")
    ax[0][0].set_ylabel(r"$v_{m}(t)$ (V)")
    ax[0][0].plot(sol_t, v_m, label=r"Motor Voltage Input", color="C0")
    ax[0][0].legend(loc="upper right")

    ax[1][0].set_xlabel(r"$t$ (s)")
    ax[1][0].set_ylabel(r"$v_{b}(t)$ (V)")
    ax[1][0].plot(sol_t, v_b, label=r"Brake Voltage Input", color="C0")
    ax[1][0].legend(loc="upper right")

    ax[2][0].set_xlabel(r"$t$ (s)")
    ax[2][0].set_ylabel(r"$\tau_{d}(t)$ (Nm)")
    ax[2][0].plot(sol_t, tau_d, label=r"Torque Disturbance", color="C0")
    ax[2][0].legend(loc="upper right")

    ax[0][1].set_xlabel(r"$t$ (s)")
    ax[0][1].set_ylabel(r"$i_{m}(t)$ (A)")
    ax[0][1].plot(sol_t, i_m, label=r"Motor current", color="C0")
    ax[0][1].legend(loc="upper right")

    ax[1][1].set_xlabel(r"$t$ (s)")
    ax[1][1].set_ylabel(r"$i_{b}(t)$ (A)")
    ax[1][1].plot(sol_t, i_b, label=r"Brake Current", color="C0")
    ax[1][1].legend(loc="upper right")

    ax[2][1].set_xlabel(r"$t$ (s)")
    ax[2][1].set_ylabel(r"$\theta(t)$ (rad)")
    ax[2][1].plot(sol_t, theta, label=r"Angular Position", color="C0")
    ax[2][1].legend(loc="upper right")

    X = np.block([[sol_t.reshape(1, len(sol_t)).T, sol_u.T]])

    y = sol_y.T

    chromosome_parameter_ranges = {}

    range_var = 0.5
    for key, value in motor_params.items():
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
    )

    genetic_algo_regressor.fit(X, y, n_iter=40, x0=x0)

    best_fit_params = genetic_algo_regressor._elite_chromosome

    best_chromosome_motor_params = genetic_algo_regressor._gen_chromosome_dict(
        best_fit_params
    )

    print(f"{best_chromosome_motor_params=}")

    motor_fit = FixedNHAPMG(dt=motor_dt, **best_chromosome_motor_params)
    motor_pm2i_fit = motor_fit.generate_process_model_to_integrate()

    sol_t_fit, sol_u_fit, _, sol_y_fit = motor_pm2i_fit.integrate(
        compute_u_from_t=input_gen.value_at_t, dt_data=dt_data, t_end=t_end, x0=x0
    )

    v_m_fit = sol_u_fit[0, :]
    v_b_fit = sol_u_fit[1, :]
    tau_d_fit = sol_u_fit[2, :]

    i_m_fit = sol_y_fit[0, :]
    i_b_fit = sol_y_fit[1, :]
    theta_fit = sol_y_fit[2, :]

    ax[0][0].set_xlabel(r"$t$ (s)")
    ax[0][0].set_ylabel(r"$v_{m}(t)$ (V)")
    ax[0][0].plot(sol_t, v_m_fit, label=r"Motor Voltage Input - Best Fit", color="C1")
    ax[0][0].legend(loc="upper right")

    ax[1][0].set_xlabel(r"$t$ (s)")
    ax[1][0].set_ylabel(r"$v_{b}(t)$ (V)")
    ax[1][0].plot(sol_t, v_b_fit, label=r"Brake Voltage Input - Best Fit", color="C1")
    ax[1][0].legend(loc="upper right")

    ax[2][0].set_xlabel(r"$t$ (s)")
    ax[2][0].set_ylabel(r"$\tau_{d}(t)$ (Nm)")
    ax[2][0].plot(sol_t, tau_d_fit, label=r"Torque Disturbance - Best Fit", color="C1")
    ax[2][0].legend(loc="upper right")

    ax[0][1].set_xlabel(r"$t$ (s)")
    ax[0][1].set_ylabel(r"$i_{m}(t)$ (A)")
    ax[0][1].plot(sol_t, i_m_fit, label=r"Motor current - Best Fit", color="C1")
    ax[0][1].legend(loc="upper right")

    ax[1][1].set_xlabel(r"$t$ (s)")
    ax[1][1].set_ylabel(r"$i_{b}(t)$ (A)")
    ax[1][1].plot(sol_t, i_b_fit, label=r"Brake Current - Best Fit", color="C1")
    ax[1][1].legend(loc="upper right")

    ax[2][1].set_xlabel(r"$t$ (s)")
    ax[2][1].set_ylabel(r"$\theta(t)$ (rad)")
    ax[2][1].plot(sol_t, theta_fit, label=r"Angular Position - Best Fit", color="C1")
    ax[2][1].legend(loc="upper right")

    plt.show()

    original_params = np.array(list(motor_params.values()))


if __name__ == "__main__":
    _main()

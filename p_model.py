from typing import Dict

import numpy as np

import pysysid.pm2i as pm2i


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
        omega = total_state[2, 0]

        return np.array(
            [
                [i_m],
                [i_b],
                [omega],
            ]
        )

    def compute_output_detailled(
        self, t: float, total_state: np.ndarray, total_input: np.ndarray
    ) -> np.ndarray:
        # hybrid actuator process model
        i_m = total_state[0, 0]
        i_b = total_state[1, 0]
        omega = total_state[2, 0]
        z = total_state[3, 0]

        return self.J * np.array(
            [
                [self.l31 * i_m],
                [self.l32 * omega + self.l37 * omega],
                [(self.l33 + self.l34 * i_b) * z],
                [(self.l38 * i_b) * omega],
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

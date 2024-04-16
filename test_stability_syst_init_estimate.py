import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import control


def _main():

    p = {
        "brake_alpha_0": 0.3,  # multiplied by hysteresis variable z that directly affects the torque, so lets give it a reasonable value
        "brake_alpha_1": 0.0005,  # same as above, multiplied by the current, so let's keep it lower
        "brake_c_0": 3e-4,  # lets set the brake's viscous damping to the twice the motor's value
        "brake_c_1": 3e-2,  # let's assume this is quite high, since the brake should be able to produce high torque with little energy
        "brake_R": 111.0,  # datasheet b6
        "brake_L": 0.1693,  # assuming the brake's electrical time constant is 3 times as big as the motor's electrical time constant
        "brake_A": 1,  # read that this parameter is superfluous and can be set to any values, other values in the hysteresis model wil adapt
        "brake_gamma": 1e-5,  # TODO: no clue
        "brake_beta": 1e1,  # TODO : no clue
        "brake_J": 2.6e-6,  # datasheet b6
        "motor_K": 0.14132959412,  # datasheet s23
        "motor_R": 1.18,  # datasheet s23
        "motor_L": 6e-4,  # taking a guess
        "motor_B": 1.534e-4,  # since mechanical time constant should be 100 times bigger than electrical time constant, after computation, this is the result
        "motor_J": 7.8e-6,  # lets guess 3 times the value of the brake
        "m_driver_K": 10.0,  # total guess
        "m_driver_w0": 100.0,  # total guess
        "b_driver_K": 80.0,  # total guess
        "b_driver_w0": 500.0,  # total guess
    }

    s = control.tf("s")

    # brake_x = [v_b, i_b], brake_u = [v_cmd_b], brake_y = [i_b]

    brake_A = np.array(
        [
            [-p["b_driver_w0"], -p["b_driver_K"] * p["b_driver_w0"]],
            [1.0 / p["brake_L"], -p["brake_R"] / p["brake_L"]],
        ]
    )

    brake_B = np.array(
        [
            [0.5 * p["b_driver_K"] * p["b_driver_w0"]],
            [0.0],
        ]
    )

    brake_C = np.array([[0.0, 1.0]])

    brake_D = np.array([[0.0]])

    brake_lti = control.ss(brake_A, brake_B, brake_C, brake_D)

    print(control.poles(brake_lti))

    motor_A = np.array(
        [
            [-p["m_driver_w0"], -p["m_driver_K"] * p["m_driver_w0"]],
            [1.0 / p["motor_L"], -p["motor_R"] / p["motor_L"]],
        ]
    )

    motor_B = np.array(
        [
            [0.5 * p["m_driver_K"] * p["m_driver_w0"], 0.0],
            [0.0, p["motor_K"] / p["motor_L"]],
        ]
    )

    motor_C = np.eye(2)

    motor_D = np.zeros((2, 2))

    motor_lti = control.ss(motor_A, motor_B, motor_C, motor_D)

    print(control.poles(motor_lti))


if __name__ == "__main__":
    _main()

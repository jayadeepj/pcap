import math
import numpy as np


def beam_deflection_param(radius, length, noise_std=1., E=3e9):
    """
    Calculate the beam deflection parameter (Ks).

    Parameters:
    - E: Elastic modulus
    - radius: Radius of the beam
    - length: Length of the beam
    - noise_std: # Adjust this value to control the amount of noise

    Returns:
    - Kp, Kd: Beam deflection parameters
    """

    kp = (E * math.pi * (radius ** 4)) / (2 * length)
    # Generate & add Gaussian noise
    if noise_std > 0.:
        gauss_noise = np.random.normal(0, noise_std)
        kp += gauss_noise

    kd = kp / 10
    return round(max(kp, 2.), 2), round(max(kd, 1.), 2)


def rud_deflection_param(branch_level, base_kp=100, noise_std=1., ):
    """
    Use a rudimentary method for calculating stiffness parameters
    For L1 use base_kp, every subsequent level, /=2


    Returns:
    - Kp, Kd: Beam deflection parameters
    """

    if branch_level > 5:
        base_kp = base_kp * (2 ** (branch_level - 5))

    if base_kp <= 100:
        assert 0 < branch_level < 6, "Parameters can get too low"

    kp = base_kp / (2 ** (branch_level - 1))

    if noise_std > 0.:
        gauss_noise = np.random.normal(0, noise_std)
        kp += gauss_noise

    kd = kp / 5
    return round(max(kp, 2.), 2), round(max(kd, 2.), 2)


if __name__ == '__main__':
    # some test values
    elastic_modulus = 3e9  # Example elastic modulus in Pa (Pascals)
    beam_radius_l5 = 0.00433  # Example beam radius in meters L5
    beam_length_l5 = 0.2250000000000001  # Example beam length in meters L5

    beam_radius_l4 = 0.00749956  # Example beam radius in meters L4
    beam_length_l4 = 0.24952500000000002  # Example beam length in meters L4

    beam_radius_l3 = 0.012989237919999999  # Example beam radius in meters L3
    beam_length_l3 = 0.27672322499999996  # Example beam length in meters L3

    _kp, _kd = beam_deflection_param(E=elastic_modulus, radius=beam_radius_l4, length=beam_length_l4)
    print("Beam deflection parameters:", _kp, _kd)

    _kp, _kd = rud_deflection_param(branch_level=4, noise_std=2.0)
    print("Rud parameters:", _kp, _kd)

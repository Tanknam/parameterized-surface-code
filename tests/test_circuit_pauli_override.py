import math
import pathlib
import sys

import stim

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.circuit_pauli_override import CircuitWithQubitNoise


def test_scale_to_target_error_rate_uses_smallest_safe_factor():
    circuit = stim.Circuit()
    circuit.append_operation("X_ERROR", [0], [0.1])
    circuit.append_operation("X_ERROR", [1], [0.1])

    noise_map = {
        0: (0.5, 0.3, 0.15),  # total = 0.95
        1: (0.4, 0.2, 0.3),   # total = 0.9
    }

    circuit_with_noise = CircuitWithQubitNoise(circuit, noise_map)

    scaled = circuit_with_noise.scale_to_target_error_rate(1.0)

    # Expected scaling factor is limited by the qubit with the highest noise
    # total, i.e. min(1/0.95, 1/0.9) * 0.995.
    expected_scaling_factor = min(1 / 0.95, 1 / 0.9) * 0.995

    qubit0_total = sum(scaled.qubit_noise_map[0])
    qubit1_total = sum(scaled.qubit_noise_map[1])

    assert math.isclose(qubit0_total, 0.95 * expected_scaling_factor, rel_tol=1e-9)
    assert math.isclose(qubit1_total, 0.9 * expected_scaling_factor, rel_tol=1e-9)

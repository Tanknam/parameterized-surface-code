import math
import pathlib
import sys

import pytest
import stim

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.circuit_pauli_override import (
    CircuitWithQubitNoise,
    pauli_from_t1_t2,
    t1_t2_from_pauli,
)


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


def test_t1_t2_from_pauli_round_trip():
    t1 = 30.0
    t2 = 45.0
    px, py, pz = pauli_from_t1_t2(1.0, t1, t2)
    recovered = t1_t2_from_pauli(1.0, px, py, pz)

    assert math.isclose(recovered.t1_us, t1, rel_tol=1e-9)
    assert math.isclose(recovered.t2_us, t2, rel_tol=1e-9)


def test_get_mean_t1_t2_from_noise_map():
    circuit = stim.Circuit()
    noise_map = {}
    durations = 1.0
    for q, (t1, t2) in enumerate([(20.0, 30.0), (40.0, 60.0), (25.0, 50.0)]):
        noise_map[q] = pauli_from_t1_t2(durations, t1, t2)

    circuit_with_noise = CircuitWithQubitNoise(circuit, noise_map)

    info = circuit_with_noise.get_mean_t1_t2(idle_duration_us=durations)

    expected = t1_t2_from_pauli(
        durations,
        sum(px for px, _, _ in noise_map.values()) / len(noise_map),
        sum(py for _, py, _ in noise_map.values()) / len(noise_map),
        sum(pz for _, _, pz in noise_map.values()) / len(noise_map),
    )

    assert math.isclose(info.t1_us, expected.t1_us, rel_tol=1e-9)
    assert math.isclose(info.t2_us, expected.t2_us, rel_tol=1e-9)


def test_get_mean_t1_t2_requires_noise_map():
    circuit_with_noise = CircuitWithQubitNoise(stim.Circuit(), {})

    with pytest.raises(ValueError):
        circuit_with_noise.get_mean_t1_t2()

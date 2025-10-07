"""Per-qubit noise override for Stim circuits.

This script provides helpers to replace uniform noise instructions emitted by
Stim's surface_code generators with per-qubit probabilities computed from
measured T1/T2 values. It operates directly on a `stim.Circuit`, splitting
aggregate instructions such as `X_ERROR` / `DEPOLARIZE1` into separate
per-qubit `X_ERROR` or `PAULI_CHANNEL_1` entries, and drops any `DEPOLARIZE2`
operations entirely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import stim


@dataclass(frozen=True)
class T1T2Info:
    t1_us: float
    t2_us: float


@dataclass(frozen=False)
class CircuitWithQubitNoise:
    """A circuit with explicit per-qubit noise metadata for accurate scaling."""
    
    circuit: stim.Circuit
    qubit_noise_map: Dict[int, Tuple[float, float, float]]  # qubit -> (px, py, pz)
    
    def get_mean_error_rate(self) -> float:
        """Calculate the true mean error rate across all qubits.
        
        This is based only on the per-qubit noise map (single-qubit errors).
        Two-qubit gate errors are handled separately during transformation.
        """
        if not self.qubit_noise_map:
            return 0.0
        
        total_error_rates = [px + py + pz for px, py, pz in self.qubit_noise_map.values()]
        return float(sum(total_error_rates) / len(total_error_rates))
    
    def get_qubit_error_rates(self) -> Dict[int, float]:
        """Get individual qubit error rates (px + py + pz)."""
        return {q: px + py + pz for q, (px, py, pz) in self.qubit_noise_map.items()}
    
    def scale_to_target_error_rate(self, target_error_rate: float) -> 'CircuitWithQubitNoise':
        """Scale all noise to achieve target mean error rate, preserving relative differences.
        
        Handles physical constraints: ensures px + py + pz <= 1 for all qubits.
        """
        current_mean = self.get_mean_error_rate()
        
        if current_mean == 0.0:
            raise ValueError("Circuit has no error noise to scale")
        
        scaling_factor = target_error_rate / current_mean
        
        # Check if scaling would violate physical constraints
        max_scaled_total = 0.0
        for qubit, (px, py, pz) in self.qubit_noise_map.items():
            scaled_total = (px + py + pz) * scaling_factor
            max_scaled_total = max(max_scaled_total, scaled_total)
        
        if max_scaled_total > 1.0:
            # Find the maximum safe scaling factor. All qubits must respect the
            # px + py + pz <= 1 constraint simultaneously, so we must select the
            # *smallest* safe factor across the ensemble.
            max_safe_factor = float("inf")
            for qubit, (px, py, pz) in self.qubit_noise_map.items():
                total = px + py + pz
                if total > 0:
                    safe_factor = 1.0 / total
                    max_safe_factor = min(max_safe_factor, safe_factor)

            if math.isinf(max_safe_factor):
                raise ValueError("Unable to determine safe scaling factor for circuit noise")

            # Use the minimum of requested and maximum safe scaling
            actual_scaling_factor = min(scaling_factor, max_safe_factor * 0.995)  # Larger margin for safety
            
            # Warn user about the constraint
            achieved_rate = current_mean * actual_scaling_factor
            import warnings
            warnings.warn(
                f"Target error rate {target_error_rate:.6f} would violate physical constraints "
                f"(px+py+pz > 1). Using maximum safe scaling factor {actual_scaling_factor:.4f} "
                f"to achieve error rate {achieved_rate:.6f} instead.",
                UserWarning
            )
            scaling_factor = actual_scaling_factor
        
        # Scale the circuit instructions
        scaled_circuit = _scale_noise_block(self.circuit, scaling_factor)
        
        # Scale the metadata with proper bounds checking
        scaled_noise_map = {}
        for qubit, (px, py, pz) in self.qubit_noise_map.items():
            new_px = px * scaling_factor
            new_py = py * scaling_factor  
            new_pz = pz * scaling_factor
            
            # Ensure physical constraint: px + py + pz <= 1
            total = new_px + new_py + new_pz
            if total > 1.0:
                # Scale down proportionally to respect the constraint with safety margin
                factor = 0.995 / total  # More conservative margin
                new_px *= factor
                new_py *= factor
                new_pz *= factor
            
            # Final bounds check
            new_px = min(1.0, max(0.0, new_px))
            new_py = min(1.0, max(0.0, new_py))
            new_pz = min(1.0, max(0.0, new_pz))
            
            scaled_noise_map[qubit] = (new_px, new_py, new_pz)
        
        return CircuitWithQubitNoise(scaled_circuit, scaled_noise_map)
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistics about the noise distribution."""
        if not self.qubit_noise_map:
            return {}
        
        error_rates = list(self.get_qubit_error_rates().values())
        return {
            'mean_error_rate': float(sum(error_rates) / len(error_rates)),
            'min_error_rate': float(min(error_rates)),
            'max_error_rate': float(max(error_rates)),
            'std_error_rate': float((sum((x - sum(error_rates)/len(error_rates))**2 for x in error_rates) / len(error_rates))**0.5),
            'num_qubits': len(error_rates)
        }
    
    def check_physical_constraints(self) -> bool:
        """Check if all qubit error probabilities satisfy px + py + pz <= 1."""
        for qubit, (px, py, pz) in self.qubit_noise_map.items():
            if px + py + pz > 1.0 + 1e-10:  # Small tolerance for numerical errors
                return False
        return True
    
    def get_max_safe_scaling_factor(self) -> float:
        """Get the maximum scaling factor that preserves physical constraints."""
        if not self.qubit_noise_map:
            return float('inf')
        
        max_safe_factor = float('inf')
        for qubit, (px, py, pz) in self.qubit_noise_map.items():
            total = px + py + pz
            if total > 0:
                safe_factor = 1.0 / total
                max_safe_factor = min(max_safe_factor, safe_factor)
        
        return max_safe_factor if max_safe_factor != float('inf') else 1.0

    def get_mean_t1_t2(self, idle_duration_us: float = 1.0) -> T1T2Info:
        """Return effective mean T1/T2 inferred from the per-qubit noise map."""

        if not self.qubit_noise_map:
            raise ValueError("No noise data available to estimate T1/T2 values")

        num_qubits = len(self.qubit_noise_map)
        mean_px = sum(px for px, _, _ in self.qubit_noise_map.values()) / num_qubits
        mean_py = sum(py for _, py, _ in self.qubit_noise_map.values()) / num_qubits
        mean_pz = sum(pz for _, _, pz in self.qubit_noise_map.values()) / num_qubits

        return t1_t2_from_pauli(idle_duration_us, mean_px, mean_py, mean_pz)


def pauli_from_t1_t2(duration_us: float, t1_us: float, t2_us: float) -> Tuple[float, float, float]:
    """Return (px, py, pz) using the standard Pauli-twirled T1/T2 approximation."""

    duration = float(duration_us)
    t1 = max(float(t1_us), 1e-9)
    t2 = max(float(t2_us), 1e-9)
    if duration <= 0:
        return (0.0, 0.0, 0.0)
    px = py = (1.0 - math.exp(-duration / t1)) / 4.0
    pz = (1.0 + math.exp(-duration / t1) - 2.0 * math.exp(-duration / t2)) / 4.0
    px = max(0.0, min(1.0, px))
    py = max(0.0, min(1.0, py))
    pz = max(0.0, min(1.0, pz))
    return (px, py, pz)


def t1_t2_from_pauli(duration_us: float, px: float, py: float, pz: float) -> T1T2Info:
    """Infer effective T1/T2 values from a Pauli channel description."""

    duration = float(duration_us)
    if duration <= 0:
        raise ValueError("Duration must be positive to infer T1/T2 values")

    px = float(px)
    py = float(py)
    pz = float(pz)

    if px < 0 or py < 0 or pz < 0:
        raise ValueError("Pauli channel probabilities must be non-negative")
    if px > 1 or py > 1 or pz > 1:
        raise ValueError("Pauli channel probabilities must not exceed 1")

    mean_px = 0.5 * (px + py)
    mean_px = max(0.0, mean_px)

    if mean_px == 0:
        exp_t1 = 1.0
        t1 = float("inf")
    else:
        exp_t1 = 1.0 - 4.0 * mean_px
        exp_t1 = min(max(exp_t1, 1e-12), 1.0 - 1e-12)
        t1 = -duration / math.log(exp_t1)

    exp_t2 = (1.0 + exp_t1 - 4.0 * pz) / 2.0
    if exp_t2 >= 1.0:
        t2 = float("inf")
    else:
        exp_t2 = min(max(exp_t2, 1e-12), 1.0 - 1e-12)
        t2 = -duration / math.log(exp_t2)

    return T1T2Info(t1_us=float(t1), t2_us=float(t2))


def fix_physical_t1_t1(distance_x_t1t2: List[Tuple[float, float, float]]):
    """Check that T1 and T2 times are physically valid. If not, force to limit"""
    fixed = []
    for d, t1, t2 in distance_x_t1t2:
        if t2 > 2 * t1:
            print(f"Warning: For qubit {d}, T2={t2} > 2*T1={2*t1}. Forcing T2=2*T1.")
            fixed.append((d, t1, 2 * t1))
        else:
            fixed.append((d, t1, t2))
    return fixed
    

def _split_instruction_per_qubit(
    circuit: stim.Circuit,
    qubit_probs: Mapping[int, Tuple[float, float, float]],
    name: str,
    qubits: Iterable[int],
    kind: str,
) -> None:
    """Append per-qubit noise operations to `circuit`.

    Args:
        qubit_probs: mapping qubit -> (px, py, pz)
        name: either "X_ERROR", "PAULI_CHANNEL_1", or "PAULI_CHANNEL_2"
        kind: "x" to use px values only, "pauli" to use full triples, or "pauli2" for two-qubit.
    """

    qubits_list = list(qubits)
    
    if kind == "pauli2":
        # Handle two-qubit Pauli channels
        if len(qubits_list) % 2 != 0:
            raise ValueError(f"Two-qubit operations require even number of qubits, got {len(qubits_list)}")
        
        for i in range(0, len(qubits_list), 2):
            q1, q2 = qubits_list[i], qubits_list[i + 1]
            px1, py1, pz1 = qubit_probs.get(q1, (0.0, 0.0, 0.0))
            px2, py2, pz2 = qubit_probs.get(q2, (0.0, 0.0, 0.0))
            
            # Compute the 15 two-qubit Pauli probabilities as cross products
            # Order: IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ
            p_two_qubit = [
                (1 - px1 - py1 - pz1) * px2,  # IX
                (1 - px1 - py1 - pz1) * py2,  # IY
                (1 - px1 - py1 - pz1) * pz2,  # IZ
                px1 * (1 - px2 - py2 - pz2),  # XI
                px1 * px2,                     # XX
                px1 * py2,                     # XY
                px1 * pz2,                     # XZ
                py1 * (1 - px2 - py2 - pz2),  # YI
                py1 * px2,                     # YX
                py1 * py2,                     # YY
                py1 * pz2,                     # YZ
                pz1 * (1 - px2 - py2 - pz2),  # ZI
                pz1 * px2,                     # ZX
                pz1 * py2,                     # ZY
                pz1 * pz2,                     # ZZ
            ]
            
            # Check if any error probabilities are non-zero
            if any(p > 0 for p in p_two_qubit):
                circuit.append_operation(name, [q1, q2], p_two_qubit)
        return
    
    # Handle single-qubit operations as before
    for q in qubits_list:
        px, py, pz = qubit_probs.get(q, (0.0, 0.0, 0.0))
        if kind == "x":
            prob = px
            if prob <= 0:
                continue
            circuit.append_operation(name, [q], [prob])
        else:
            if px == 0 and py == 0 and pz == 0:
                continue
            circuit.append_operation(name, [q], [px, py, pz])


def _transform_block(
    block: stim.Circuit,
    qubit_probs: Mapping[int, Tuple[float, float, float]],
) -> stim.Circuit:
    transformed = stim.Circuit()
    for op in block:
        name = op.name
        if name in {"X_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"}:
            targets = [t.value for t in op.targets_copy()]
            if name == "X_ERROR":
                _split_instruction_per_qubit(transformed, qubit_probs, "X_ERROR", targets, kind="x")
            elif name == "DEPOLARIZE1":
                _split_instruction_per_qubit(transformed, qubit_probs, "PAULI_CHANNEL_1", targets, kind="pauli")
            elif name == "DEPOLARIZE2":
                _split_instruction_per_qubit(transformed, qubit_probs, "PAULI_CHANNEL_2", targets, kind="pauli2")
            continue
        if name == "REPEAT":
            body = _transform_block(op.body_copy(), qubit_probs)
            transformed.append(stim.CircuitRepeatBlock(op.repeat_count, body, tag=op.tag))
            continue
        transformed.append(op)
    return transformed


def override_noise_with_pauli_channels(
    circuit: stim.Circuit,
    qubit_t1_t2: Mapping[int, Union[Tuple[float, float], T1T2Info]],
    *,
    idle_duration_us: float = 1.0,
    # oneq_duration_us: Optional[float] = None,
) -> CircuitWithQubitNoise:
    """Return a CircuitWithQubitNoise with per-qubit Pauli noise and metadata."""

    # oneq_duration = idle_duration_us if oneq_duration_us is None else float(oneq_duration_us)

    pauli_lookup: Dict[int, Tuple[float, float, float]] = {}
    for q, info in qubit_t1_t2.items():
        if isinstance(info, T1T2Info):
            t1_us, t2_us = info.t1_us, info.t2_us
        else:
            t1_us, t2_us = info
        pauli_lookup[int(q)] = pauli_from_t1_t2(idle_duration_us, t1_us, t2_us)

    transformed_circuit = _transform_block(circuit, pauli_lookup)
    return CircuitWithQubitNoise(transformed_circuit, pauli_lookup)


def _scale_noise_block(
    block: stim.Circuit,
    scaling_factor: float,
) -> stim.Circuit:
    """Scale all noise instructions in a circuit block by a given factor.
    
    Respects physical constraints: ensures px + py + pz <= 1 for PAULI_CHANNEL_1.
    """
    scaled = stim.Circuit()
    
    for op in block:
        name = op.name
        
        if name == "X_ERROR":
            # Scale X_ERROR probability
            targets = [t.value for t in op.targets_copy()]
            old_prob = op.gate_args_copy()[0]
            new_prob = min(1.0, max(0.0, old_prob * scaling_factor))
            if new_prob > 0:
                scaled.append_operation("X_ERROR", targets, [new_prob])
            
        elif name == "PAULI_CHANNEL_1":
            # Scale PAULI_CHANNEL_1 probabilities with physical constraint
            targets = [t.value for t in op.targets_copy()]
            args = op.gate_args_copy()
            if len(args) == 3:
                px, py, pz = args
                new_px = px * scaling_factor
                new_py = py * scaling_factor
                new_pz = pz * scaling_factor
                
                # Check physical constraint: px + py + pz <= 1
                total = new_px + new_py + new_pz
                if total > 1.0:
                    # Scale down proportionally to respect constraint with safety margin
                    factor = 0.995 / total  # More conservative margin
                    new_px *= factor
                    new_py *= factor
                    new_pz *= factor
                
                # Final bounds check
                new_px = min(1.0, max(0.0, new_px))
                new_py = min(1.0, max(0.0, new_py))
                new_pz = min(1.0, max(0.0, new_pz))
                
                if new_px > 0 or new_py > 0 or new_pz > 0:
                    scaled.append_operation("PAULI_CHANNEL_1", targets, [new_px, new_py, new_pz])
                    
        elif name == "PAULI_CHANNEL_2":
            # Scale PAULI_CHANNEL_2 probabilities with physical constraint
            targets = [t.value for t in op.targets_copy()]
            args = op.gate_args_copy()
            if len(args) == 15:
                # Scale all 15 probabilities
                scaled_probs = [p * scaling_factor for p in args]
                
                # Check physical constraint: sum of all 15 probabilities <= 1
                total = sum(scaled_probs)
                if total > 1.0:
                    # Scale down proportionally to respect constraint with safety margin
                    factor = 0.995 / total  # More conservative margin
                    scaled_probs = [p * factor for p in scaled_probs]
                
                # Final bounds check
                scaled_probs = [min(1.0, max(0.0, p)) for p in scaled_probs]
                
                if any(p > 0 for p in scaled_probs):
                    scaled.append_operation("PAULI_CHANNEL_2", targets, scaled_probs)
                    
        elif name == "Y_ERROR":
            # Scale Y_ERROR probability
            targets = [t.value for t in op.targets_copy()]
            old_prob = op.gate_args_copy()[0]
            new_prob = min(1.0, max(0.0, old_prob * scaling_factor))
            if new_prob > 0:
                scaled.append_operation("Y_ERROR", targets, [new_prob])
                
        elif name == "Z_ERROR":
            # Scale Z_ERROR probability
            targets = [t.value for t in op.targets_copy()]
            old_prob = op.gate_args_copy()[0]
            new_prob = min(1.0, max(0.0, old_prob * scaling_factor))
            if new_prob > 0:
                scaled.append_operation("Z_ERROR", targets, [new_prob])
                
        elif name == "REPEAT":
            # Recursively handle REPEAT blocks
            body = _scale_noise_block(op.body_copy(), scaling_factor)
            scaled.append(stim.CircuitRepeatBlock(op.repeat_count, body, tag=op.tag))
            
        else:
            # Copy all other operations unchanged
            scaled.append(op)
    
    return scaled


def create_circuit_from_t1_t2(
    rounds: int,
    distance: int,
    t1t2_mapping: Mapping[int, Tuple[float, float]],
    *,
    idle_duration_us: float = 1.0,
) -> CircuitWithQubitNoise:
    """Create a circuit with per-qubit T1/T2 noise at natural error rates.
    
    Args:
        rounds: Number of QEC rounds
        distance: Code distance
        t1t2_mapping: Dictionary mapping qubit -> (T1, T2) values
        idle_duration_us: Duration for idle operations
        include_two_qubit_errors: Whether to add synthetic two-qubit gate errors
        
    Returns:
        CircuitWithQubitNoise with natural per-qubit noise levels
    """
    base_circuit = stim.Circuit.generated(
        "surface_code:unrotated_memory_z",
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=0.1,  # This automatically generates DEPOLARIZE2 operations
        after_reset_flip_probability=0.2,
        before_measure_flip_probability=0.3,
        before_round_data_depolarization=0.4,
    )
    
    return override_noise_with_pauli_channels(
        base_circuit, 
        t1t2_mapping, 
        idle_duration_us=idle_duration_us
    )


def create_scaled_circuit_from_t1_t2(
    rounds: int,
    distance: int,
    t1t2_mapping: Mapping[int, Tuple[float, float]],
    target_error_rate: float,
    *,
    idle_duration_us: float = 1.0,
) -> stim.Circuit:
    """Create a circuit with per-qubit T1/T2 noise scaled to a target mean error rate.
    
    Args:
        rounds: Number of QEC rounds
        distance: Code distance
        t1t2_mapping: Dictionary mapping qubit -> (T1, T2) values
        target_error_rate: Desired mean total error probability
        idle_duration_us: Duration for idle operations
        include_two_qubit_errors: Whether to include synthetic two-qubit gate errors
        
    Returns:
        stim.Circuit with scaled per-qubit noise levels
    """  
    circuit_with_noise = create_circuit_from_t1_t2(
        rounds=rounds,
        distance=distance,
        t1t2_mapping=t1t2_mapping,
        idle_duration_us=idle_duration_us,
    )

    return circuit_with_noise.scale_to_target_error_rate(target_error_rate).circuit


def build_rmwpm_decoder(
    circuit_with_noise: CircuitWithQubitNoise,
    *,
    approximate_disjoint_errors: bool = True,
    decompose_errors: bool = True,
    min_probability: float = 1e-15,
):
    """Create an rMWPM decoder tailored to a noise-tagged Stim circuit.

    This is a light-weight convenience wrapper around
    :func:`rmwpm_decoder.RMWPMDecoder.from_circuit`. It allows notebooks and
    scripts to stay within the ``circuit_pauli_override`` namespace when
    instantiating the decoder.
    """

    from .rmwpm_decoder import RMWPMDecoder, SinterRMWPMDecoder

    return RMWPMDecoder.from_circuit(
        circuit_with_noise.circuit,
        approximate_disjoint_errors=approximate_disjoint_errors,
        decompose_errors=decompose_errors,
        min_probability=min_probability,
    )


def create_sinter_rmwpm_decoder(*, min_probability: float = 1e-15):
    """Return a :class:`sinter.Decoder` instance wrapping rMWPM weights."""

    from .rmwpm_decoder import SinterRMWPMDecoder

    return SinterRMWPMDecoder(min_probability=min_probability)

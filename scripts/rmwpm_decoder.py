"""Reweighted MWPM decoder implementation.

This module builds a minimum-weight perfect matching decoder that applies the
per-edge reweighting described in the rMWPM paper. Given a Stim circuit with
per-qubit noise (for example generated with ``CircuitWithQubitNoise`` from
``circuit_pauli_override``), we extract the detector error model, convert each
(graphlike) error mechanism into an edge whose weight is the negative
log-likelihood of the underlying error probability, and feed the resulting graph
into PyMatching. The decoder exposes a convenience wrapper that can be plugged
into Sinter or used directly from notebooks.
"""
from __future__ import annotations

import dataclasses
import logging
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pymatching
import sinter
import stim


LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class RMWPMBuildStats:
    """Summary statistics reported while building the reweighted matcher."""

    num_detectors: int
    num_observables: int
    num_edges: int
    num_boundary_edges: int
    skipped_non_graphlike: int


class RMWPMDecoder:
    """Decoder that performs reweighted MWPM by customising PyMatching edges.

    The decoder is constructed from a :class:`stim.DetectorErrorModel`. Each
    graphlike error term (i.e., one that flips at most two detectors after
    decomposition) is mapped to a PyMatching edge whose weight is

    .. math:: w = - \log(p)

    where ``p`` is the probability attached to that error term in the detector
    error model. This follows Eq. (7) of the referenced paper and ensures that
    long chains passing through high-error-probability qubits are
    preferentially selected by the matching solver.
    """

    def __init__(self, matching: pymatching.Matching, stats: RMWPMBuildStats):
        self._matching = matching
        self._stats = stats

    @property
    def stats(self) -> RMWPMBuildStats:
        return self._stats

    @classmethod
    def from_circuit(
        cls,
        circuit: stim.Circuit,
        *,
        approximate_disjoint_errors: bool = True,
        decompose_errors: bool = True,
        min_probability: float = 1e-15,
    ) -> "RMWPMDecoder":
        """Create a decoder by extracting and reweighting the circuit's DEM.

        Args:
            circuit: Stim circuit containing detector/basis coordinates.
            approximate_disjoint_errors: Forwarded to
                :meth:`stim.Circuit.detector_error_model` so that
                ``PAULI_CHANNEL`` instructions are expanded into independent
                single-qubit events.
            decompose_errors: Whether to ask Stim to suggest graphlike
                decompositions of any hyperedge errors.
            min_probability: Numerical lower bound used when computing
                ``-log(p)`` so that vanishingly small probabilities do not
                produce infinities or NaNs.
        """

        dem = circuit.detector_error_model(
            approximate_disjoint_errors=approximate_disjoint_errors,
            decompose_errors=decompose_errors,
        )
        return cls.from_detector_error_model(
            dem,
            min_probability=min_probability,
        )

    @classmethod
    def from_detector_error_model(
        cls,
        dem: stim.DetectorErrorModel,
        *,
        min_probability: float = 1e-15,
    ) -> "RMWPMDecoder":
        """Create a decoder directly from a detector error model."""

        matching = pymatching.Matching()
        num_edges = 0
        num_boundary_edges = 0
        skipped = 0
        max_fault_id = -1
        detectors_used: Set[int] = set()

        dem_iterable = dem.flattened()

        edge_buckets: Dict[
            Tuple[Tuple[int, ...], frozenset[int]], _EdgeBucket
        ] = {}
        # When multiple error mechanisms toggle the same detector pair (and
        # logical payload), we combine them using the odd-parity probability
        # formula so that the resulting edge reflects the total likelihood of an
        # odd number of such events.

        for inst in dem_iterable:
            if inst.type != "error":
                continue

            args = inst.args_copy()
            if not args:
                continue
            probability = float(args[0])
            if probability <= 0.0:
                continue

            groups = _extract_groups(inst.targets_copy())
            if not groups:
                continue

            for detectors, fault_ids in groups:
                if len(detectors) == 0:
                    continue
                detectors_used.update(detectors)
                if fault_ids:
                    max_fault_id = max(max_fault_id, max(fault_ids))

                if len(detectors) > 2:
                    skipped += 1
                    LOGGER.debug(
                        "Skipping non-graphlike error with detectors=%s",
                        detectors,
                    )
                    continue

                detectors_key: Tuple[int, ...]
                if len(detectors) == 2:
                    detectors_key = tuple(sorted(detectors))
                else:
                    detectors_key = (detectors[0],)

                key = (detectors_key, frozenset(fault_ids))
                bucket = edge_buckets.get(key)
                if bucket is None:
                    bucket = _EdgeBucket(
                        detectors=detectors_key,
                        fault_ids=set(fault_ids),
                    )
                    edge_buckets[key] = bucket
                bucket.update(probability)

        if max_fault_id >= 0:
            matching.ensure_num_fault_ids(max_fault_id + 1)
        else:
            matching.ensure_num_fault_ids(dem.num_observables)

        for bucket in edge_buckets.values():
            probability = bucket.final_probability()
            probability = _clamp_probability(probability, min_probability)
            if probability <= 0.0:
                continue
            weight = _probability_to_weight(probability)
            fault_id_payload = _fault_ids_payload(bucket.fault_ids)

            if len(bucket.detectors) == 1:
                matching.add_boundary_edge(
                    bucket.detectors[0],
                    fault_ids=fault_id_payload,
                    weight=weight,
                    error_probability=probability,
                    merge_strategy="smallest-weight",
                )
                num_boundary_edges += 1
            elif len(bucket.detectors) == 2:
                matching.add_edge(
                    bucket.detectors[0],
                    bucket.detectors[1],
                    fault_ids=fault_id_payload,
                    weight=weight,
                    error_probability=probability,
                    merge_strategy="smallest-weight",
                )
                num_edges += 1

        # Ensure the matching graph accounts for detectors that never appear in
        # any error term (e.g., idle rounds in the DEM). PyMatching assumes the
        # detector indices are contiguous starting at zero.
        if detectors_used:
            num_detectors_in_graph = max(detectors_used) + 1
            if num_detectors_in_graph < dem.num_detectors:
                placeholder_weight = -math.log(min_probability)
                for detector in range(num_detectors_in_graph, dem.num_detectors):
                    matching.add_boundary_edge(
                        detector,
                        weight=placeholder_weight,
                        error_probability=min_probability,
                    )

        stats = RMWPMBuildStats(
            num_detectors=dem.num_detectors,
            num_observables=dem.num_observables,
            num_edges=num_edges,
            num_boundary_edges=num_boundary_edges,
            skipped_non_graphlike=skipped,
        )
        return cls(matching, stats)

    def decode(self, syndrome: np.ndarray, *, return_weight: bool = False):
        """Decode a single syndrome vector."""

        return self._matching.decode(
            syndrome,
            return_weight=return_weight,
        )

    def decode_batch(
        self,
        shots: np.ndarray,
        *,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
        return_weights: bool = False,
    ):
        """Decode a batch of syndrome shots (wrapper around PyMatching)."""

        return self._matching.decode_batch(
            shots,
            bit_packed_shots=bit_packed_shots,
            bit_packed_predictions=bit_packed_predictions,
            return_weights=return_weights,
        )

    def to_matching(self) -> pymatching.Matching:
        """Expose the underlying :class:`pymatching.Matching` object."""

        return self._matching


def _extract_groups(targets: Sequence[stim.DemTarget]):
    """Split a DEM target list on separators into detector/logical groups."""

    groups: List[tuple[List[int], Set[int]]] = []
    detectors: List[int] = []
    logicals: Set[int] = set()

    for target in targets:
        if target.is_separator():
            if detectors or logicals:
                groups.append((detectors, logicals))
                detectors = []
                logicals = set()
            continue
        if target.is_relative_detector_id():
            detectors.append(int(target.val))
        elif target.is_logical_observable_id():
            logicals.add(int(target.val))

    if detectors or logicals:
        groups.append((detectors, logicals))

    return groups


def _fault_ids_payload(fault_ids: Set[int]) -> Optional[Set[int]]:
    if not fault_ids:
        return None
    if len(fault_ids) == 1:
        # PyMatching accepts an int or a set[int]. Returning the set keeps the
        # object hashable for future merges.
        return set(fault_ids)
    return set(fault_ids)


@dataclasses.dataclass
class _EdgeBucket:
    detectors: Tuple[int, ...]
    fault_ids: Set[int]
    _parity_product: float = 1.0

    def update(self, probability: float) -> None:
        self._parity_product *= 1.0 - 2.0 * probability

    def final_probability(self) -> float:
        clamped_product = max(-1.0, min(1.0, self._parity_product))
        return 0.5 - 0.5 * clamped_product


def _clamp_probability(probability: float, min_probability: float) -> float:
    return min(max(probability, min_probability), 1.0 - min_probability)


def _probability_to_weight(probability: float) -> float:
    odds_ratio = (1.0 - probability) / probability
    return math.log(odds_ratio)


class _CompiledRMWPMDecoder(sinter.CompiledDecoder):
    def __init__(self, decoder: RMWPMDecoder):
        self._decoder = decoder

    def decode_shots_bit_packed(
        self,
        *,
        bit_packed_detection_event_data: np.ndarray,
    ) -> np.ndarray:
        return self._decoder.decode_batch(
            bit_packed_detection_event_data,
            bit_packed_shots=True,
            bit_packed_predictions=True,
        )


class SinterRMWPMDecoder(sinter.Decoder):
    """Sinter decoder wrapper that provides rMWPM weighting."""

    def __init__(self, *, min_probability: float = 1e-15):
        self._min_probability = min_probability
        self._last_stats: Optional[RMWPMBuildStats] = None

    def compile_decoder_for_dem(
        self,
        *,
        dem: stim.DetectorErrorModel,
    ) -> sinter.CompiledDecoder:
        decoder = RMWPMDecoder.from_detector_error_model(
            dem,
            min_probability=self._min_probability,
        )
        self._last_stats = decoder.stats
        return _CompiledRMWPMDecoder(decoder)

    @property
    def last_stats(self) -> Optional[RMWPMBuildStats]:
        return self._last_stats
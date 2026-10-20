"""Replace only ML-KEM's NTT/INTT with binary RRAM transform evaluation.

All K-PKE, base multiplication, noise sampling, compression and FO methods are
inherited unchanged from MLKEM. Forward weights remain the existing unsigned
0/1 planes. INTT uses a dedicated twelve-plane bank K=2^32*V modq and REDC32,
where V already includes inverse normalization. Pulses are always binary 0/1:
signed CBD encoding uses three slots
whose digital reconstruction weights are [1, 2, -4], not negative pulse levels.
Every transform returns canonical residues modulo 3329 at its boundary.
This exact logical model makes no timing or analog-device performance claim.
"""

from __future__ import annotations

import re
import numpy as np

from binary_rram import (
    BITS, TILE_INPUT_OFFSETS, TILE_OUTPUT_OFFSETS, WEIGHTS_NAME,
    _pack_rows, popcount_u64,
)
from mlkem import MLKEM, Q, _poly
from rram_intt_montgomery import MontgomeryINTT


SIGNED_SLOT_WEIGHTS = np.array([1, 2, -4], dtype=np.int64)
UNSIGNED_SLOT_WEIGHTS = 1 << np.arange(BITS, dtype=np.int64)
BASELINE_TILE_READS = 2 * 12 * 12 * 4
SIGNED_TILE_READS = 2 * 3 * 12 * 4
CBD_FORWARD_STAGE = re.compile(
    r"(?:keygen\.pke\.(?:s_hat|e_hat)|(?:encaps\.pke|decaps\.reencrypt)\.y_hat)\[([0-9]+)\]"
)


def _signed_forward(source, packed_tiles, eta):
    """Actual 0/1 intersections with the existing unsigned weight-bit tiles."""
    canonical = np.asarray(source, dtype=np.int64)
    centered = np.where(canonical > Q // 2, canonical - Q, canonical)
    if np.any(centered < -eta) or np.any(centered > eta):
        raise ValueError("known CBD transform stage violates its declared eta1 support")
    lanes = centered.reshape(128, 2).T
    codes = lanes & 7  # Three-bit two's complement; no negative voltage/pulse.
    input_bits = ((codes[:, None, :] >> np.arange(3)[None, :, None]) & 1).astype(np.uint8)
    input_words = _pack_rows(input_bits.reshape(2, 3, 2, 64))
    tile_words = input_words[:, :, [0, 0, 1, 1]]
    intersections = tile_words[:, :, None, :, None] & packed_tiles[None, None]
    tile_counts = popcount_u64(intersections)
    # Signed reconstruction occurs only after nonnegative binary counts.
    counts = tile_counts.astype(np.int64)
    merged = np.concatenate((counts[..., 0, :] + counts[..., 2, :],
                             counts[..., 1, :] + counts[..., 3, :]), axis=-1)
    weight_terms = merged << np.arange(BITS, dtype=np.int64)[None, None, :, None]
    per_slot = weight_terms.sum(axis=2, dtype=np.int64)
    slot_terms = per_slot << np.arange(3, dtype=np.int64)[None, :, None]
    slot_terms[:, 2] = -slot_terms[:, 2]
    accumulators = slot_terms.sum(axis=1, dtype=np.int64)
    if np.any(np.abs(accumulators) > eta * 128 * (Q - 1)):
        raise ArithmeticError("signed transform exceeds its proven integer bound")
    output_lanes = accumulators % Q
    result = output_lanes.T.reshape(256).tolist()
    values = dict(
        operation="rram_binary_vmm", transform="ntt",
        weight_artifact="rram_binary/" + WEIGHTS_NAME,
        physical_orientation="input_rows_output_columns", lane_order=["even", "odd"],
        input_representation="signed_cbd", pulse_levels=[0, 1], weight_encoding="unsigned12",
        input_bit_slots=3, weight_bit_planes=BITS, slot_weights=SIGNED_SLOT_WEIGHTS.copy(),
        input_canonical=canonical, input_lanes=lanes, input_bitplanes=input_bits,
        tile_input_offsets=TILE_INPUT_OFFSETS.copy(), tile_output_offsets=TILE_OUTPUT_OFFSETS.copy(),
        tile_counts_axes=["lane", "input_slot", "weight_bit", "tile", "output_column_in_tile"],
        tile_counts=tile_counts, merged_counts=merged, weight_bit_terms=weight_terms,
        per_slot_weight_sums=per_slot, signed_slot_terms=slot_terms,
        weighted_accumulators=accumulators, output_lanes=output_lanes, output=result,
        inverse_normalization_in_weights=False, raw_tile_count_max=int(tile_counts.max()),
        merged_count_max=int(merged.max()), binary_tile_read_vectors=SIGNED_TILE_READS,
        output_representation="canonical", output_is_raw=False,
        count_model="ideal binary counts; sign applied digitally; no analog/cycle model",
    )
    return result, values


class TransformOnlyMLKEM(MLKEM):
    """The only changed cryptographic method is _transform.

    signed_cbd_inputs selects a fixed public-stage schedule, never a schedule
    inferred from secret coefficient magnitudes. Unknown forward stages use
    the original twelve-slot BinaryRRAM implementation. INTT uses the separate
    twelve-slot compensated bank and REDC32, returning ordinary residues.
    reset_counts()
    clears transform and logical input/read counters together.
    """

    def __init__(self, name="ML-KEM-768", trace=None, signed_cbd_inputs=True):
        if not isinstance(signed_cbd_inputs, bool):
            raise ValueError("signed_cbd_inputs must be a bool")
        super().__init__(name, backend="rram_binary", trace=trace)
        self._montgomery_intt = MontgomeryINTT()
        self.backend = "rram_transform_only"
        self.signed_cbd_inputs = signed_cbd_inputs
        self.profile = "signed_cbd3" if signed_cbd_inputs else "unsigned12"
        self.reset_counts()

    def reset_counts(self):
        super().reset_counts()
        self.input_slot_counts = dict(signed3_ntt_calls=0, unsigned12_ntt_calls=0,
                                     unsigned12_intt_calls=0, input_bit_slots=0,
                                     lane_input_bit_slots=0)
        self.logical_read_counts = dict(binary_tile_read_vectors=0,
                                       baseline_tile_read_vectors=0,
                                       saved_tile_read_vectors=0)

    def _transform(self, poly, inverse, stage):
        source = _poly(poly)
        match = CBD_FORWARD_STAGE.fullmatch(stage) if self.signed_cbd_inputs and not inverse else None
        selected = match is not None and int(match.group(1)) < self.p.k
        direction = "intt" if inverse else "ntt"
        slots = 3 if selected else 12
        reads = SIGNED_TILE_READS if selected else BASELINE_TILE_READS
        if inverse:
            def inverse_sink(name, values):
                values.update(backend=self.backend, profile=self.profile)
                self._emit(name, **values)
            result = self._montgomery_intt.transform(source,
                trace=inverse_sink if self.trace is not None else None, stage=stage + ".rram_binary")
        elif selected:
            result, values = _signed_forward(source, self._binary_rram.packed_tiles["ntt"], self.p.eta1)
            values.update(backend=self.backend, profile=self.profile)
            self._emit(stage + ".rram_binary", **values)
        else:
            # Preserve the original full twelve-slot computation and enrich its
            # trace with the same explicit pulse/slot interpretation as above.
            def sink(name, values):
                values.update(backend=self.backend, profile=self.profile,
                              pulse_levels=[0, 1], weight_encoding="unsigned12",
                              input_representation="canonical_unsigned12",
                              slot_weights=UNSIGNED_SLOT_WEIGHTS.copy(),
                              binary_tile_read_vectors=BASELINE_TILE_READS,
                              output_representation="canonical", output_is_raw=False)
                self._emit(name, **values)
            result = self._binary_rram.transform(source, inverse=inverse,
                                                 trace=sink if self.trace is not None else None,
                                                 stage=stage + ".rram_binary")
        self.counts[direction] += 1
        kind = "signed3_ntt_calls" if selected else "unsigned12_" + direction + "_calls"
        self.input_slot_counts[kind] += 1
        self.input_slot_counts["input_bit_slots"] += slots
        self.input_slot_counts["lane_input_bit_slots"] += 2 * slots
        self.logical_read_counts["binary_tile_read_vectors"] += reads
        self.logical_read_counts["baseline_tile_read_vectors"] += BASELINE_TILE_READS
        self.logical_read_counts["saved_tile_read_vectors"] += BASELINE_TILE_READS - reads
        self._emit(stage, operation=direction, backend=self.backend, profile=self.profile,
                   call=self.counts[direction], input=source, output=result,
                   input_bit_slots=slots, output_representation="canonical", output_is_raw=False,
                   output_domain="ordinary", reduction="REDC32" if inverse else "mod_q")
        return result

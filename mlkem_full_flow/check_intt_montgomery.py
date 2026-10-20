#!/usr/bin/env python3
"""Independent audit of INTT-only RV weights and REDC32.

Does not run or generate official ML-KEM vectors. Writes its report only to
transform_only_results/intt_montgomery_review.json. Dense arithmetic appears
only in independent reference checks, never in the implementation under test.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np

import rram_intt_montgomery as mont
from mlkem import MLKEM, PARAMS, Q, GAMMAS
from rram_transform_only import TransformOnlyMLKEM


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_weights():
    folder = ROOT/'rram_intt_montgomery_weights'
    require((folder/'provenance.json').is_file() and (folder/mont.WEIGHTS_NAME).is_file(),
            'Generate the INTT artifacts before running this read-only audit')
    manifest = json.loads((folder/'provenance.json').read_text())
    for name, record in manifest['artifacts'].items():
        path = folder/name
        require(sha(path) == record['sha256'] and path.stat().st_size == record['bytes'],
                'INTT artifact integrity failure: '+name)
    with np.load(ROOT/'matrices/mlkem_shared_matrices.npz', allow_pickle=False) as source:
        forward, inverse = source['W128'].copy(), source['W128_inv'].copy()
    inverse_formula = np.array([[3303*pow(int(gamma), -j, Q) % Q for gamma in GAMMAS]
                                for j in range(128)], dtype=np.int64)
    require(np.array_equal(inverse, inverse_formula), 'Source inverse normalization/order mismatch')
    require(np.array_equal((inverse @ forward) % Q, np.eye(128, dtype=np.int64)),
            'The source V must already be the normalized inverse')
    with np.load(folder/mont.WEIGHTS_NAME, allow_pickle=False) as archive:
        planes = archive['intt_physical_planes'].copy()
        tiles = archive['intt_tiles'].copy()
        bit_weights = archive['bit_weights'].copy()
    require(planes.dtype == np.uint8 and planes.shape == (12, 128, 128)
            and np.all((planes == 0) | (planes == 1)), 'Physical planes must contain unsigned bits')
    require(np.array_equal(bit_weights, 1 << np.arange(12, dtype=np.int64)), 'Unsigned bit significance')
    physical = sum(planes[b].astype(np.int64)*(1 << b) for b in range(12))
    weight = (inverse * (mont.R % Q)) % Q
    require(np.array_equal(physical, weight.T), 'Weight planes must encode (R*V mod q).T')
    require(np.array_equal((weight @ forward) % Q, (mont.R % Q)*np.eye(128, dtype=np.int64)),
            'R compensation or inverse normalization was applied incorrectly')
    for tile, (row, col) in enumerate(((0, 0), (0, 64), (64, 0), (64, 64))):
        require(np.array_equal(tiles[:, tile], planes[:, row:row+64, col:col+64]), 'Physical tile order')
    for bit in range(12):
        csv_plane = np.loadtxt(folder/f'intt_bit{bit:02d}.csv', delimiter=',', dtype=np.uint8)
        require(np.array_equal(csv_plane, planes[bit]), 'CSV plane differs from NPZ: '+str(bit))
    return weight, inverse, {
        'artifact_hashes_checked': len(manifest['artifacts']),
        'matrix_entries_reconstructed': 128*128,
        'csv_binary_cells_checked': 12*128*128,
        'orientation': 'physical row=input; physical column=output',
        'weight_formula': '(R*V) mod q; V already contains 3303',
        'weight_bits': 12, 'highest_weight_bit': 2048,
        'weight_npz_sha256': sha(folder/mont.WEIGHTS_NAME),
        'normalization_count': 1,
    }


def verify_redc():
    # Exhaust all q residue classes in six bands, including the last valid band.
    bands = (0, 1, 2, (1 << 16)-1, 1 << 16, mont.R-1)
    values = [int(r + Q*band) for band in bands for r in range(Q)]
    boundaries = (0, 1, Q-1, Q, mont.R-2, mont.R-1, mont.R, mont.R+1,
                  mont.DOT_BOUND, Q*mont.R-2, Q*mont.R-1)
    values.extend(boundaries)
    rng = np.random.default_rng(2033303)
    values.extend(int(value) for value in rng.integers(0, Q*mont.R, size=1024, dtype=np.int64))
    t = np.asarray(values, dtype=np.int64)
    events = []
    actual = mont.redc32(t, trace=lambda stage, event: events.append((stage, event)))
    r_inverse = pow(mont.R, -1, Q)
    expected = np.asarray([(value*r_inverse) % Q for value in values], dtype=np.int64)
    require(actual.dtype == np.int64 and np.array_equal(actual, expected), 'REDC32 bigint-oracle mismatch')
    require(len(events) == 1, 'REDC32 must emit one reduction event')
    event = events[0][1]
    for name in ('input', 'low32', 'correction_m', 'numerator'):
        require(event[name].dtype == np.uint64, name+' must use uint64')
    for index in range(len(values)):
        value = values[index]
        low = value & (mont.R-1)
        correction = (low*mont.QPRIME) & (mont.R-1)
        numerator = value+correction*Q
        quotient = numerator >> 32
        require(int(event['low32'][index]) == low and int(event['correction_m'][index]) == correction,
                'REDC low-word multiplication mismatch')
        require(int(event['numerator'][index]) == numerator and int(event['quotient'][index]) == quotient,
                'REDC numerator/quotient mismatch')
        require(numerator % mont.R == 0 and 0 <= quotient < 2*Q, 'REDC divisibility/range invariant')
    require((mont.R-1)*mont.QPRIME < 1 << 64, 'Lowword multiplication overflows uint64')
    require((mont.R-1)*mont.QPRIME > (1 << 63)-1, 'Boundary test must exercise beyond signed int64')
    require(2*Q*mont.R < 1 << 45, 'REDC numerator proof changed')
    small_matrix = np.asarray(boundaries[:9], dtype=np.uint64).reshape(3, 3)
    require(np.array_equal(mont.redc32(small_matrix), np.asarray(
        [(int(value)*r_inverse) % Q for value in small_matrix.flat]).reshape(3, 3)),
        'REDC uint64 matrix input compatibility')
    return {'scalar_bigint_comparisons': len(values), 'uint64_shape_comparisons': 9,
            'exhausted_residue_bands': list(bands), 'input_domain': '0 <= T < q*R',
            'maximum_tested_T': max(values), 'q_times_R': Q*mont.R,
            'lowword_times_qprime_max': (mont.R-1)*mont.QPRIME,
            'numerator_strict_bound': 2*Q*mont.R,
            'mask_before_multiply_verified': True, 'trace_intermediates_verified': True}


def verify_transforms(engine, inverse):
    reference = MLKEM('ML-KEM-512', backend='butterfly')
    basis_count = 0
    for position in range(256):
        lane, column = position % 2, position // 2
        for amplitude in (1, Q-1):
            vector = [0]*256
            vector[position] = amplitude
            target = [0]*256
            target[lane::2] = ((inverse[:, column]*amplitude) % Q).tolist()
            actual = engine.transform(vector)
            require(actual == target, f'INTT basis orientation mismatch position={position}, amplitude={amplitude}')
            require(actual == reference.intt(vector), 'INTT basis disagrees with FIPS butterfly reference')
            basis_count += 1
    rng = np.random.default_rng(203512)
    vectors = [[0]*256, [Q-1]*256, [0, Q-1]*128,
               [Q-1 if i in (0, 1, 126, 127, 128, 129, 254, 255) else 0 for i in range(256)]]
    vectors += [rng.integers(0, Q, size=256).tolist() for _ in range(8)]
    for vector in vectors:
        require(engine.transform(vector) == reference.intt(vector), 'INTT dense/edge/random mismatch')
    # A local packed bit perturbation proves the runtime output depends on the
    # binary cells, rather than being computed by an unrelated dense shortcut.
    original = engine.packed_tiles[0, 0, 0].copy()
    probe = [1]+[0]*255
    before = engine.transform(probe)
    direction = -1 if int(original) & 1 else 1
    try:
        engine.packed_tiles[0, 0, 0] ^= np.uint64(1)
        after = engine.transform(probe)
        wanted = before.copy()
        wanted[0] = (wanted[0]+direction*pow(mont.R, -1, Q)) % Q
        require(after == wanted and after != before, 'Runtime binary-cell causality check')
    finally:
        engine.packed_tiles[0, 0, 0] = original
    return {'basis_vectors': basis_count, 'basis_definition': '128 columns x 2 lanes x amplitudes {1,3328}',
            'additional_dense_edge_random_vectors': len(vectors),
            'canonical_outputs': True, 'local_binary_cell_perturbation_verified': True,
            'weight_files_mutated': False}


def verify_trace(engine, weight):
    vector = [(i*7919+17) % Q for i in range(256)]
    events = []
    result = engine.transform(vector, trace=lambda stage, event: events.append((stage, event)))
    require(len(events) == 2, 'Expected binary VMM and separate REDC trace events')
    binary, redc = events[0][1], events[1][1]
    require(binary['operation'] == 'rram_binary_vmm' and binary['reduction'] == 'REDC32', 'Binary trace reduction label')
    require(redc['operation'] == 'montgomery_reduce', 'Missing REDC32 trace')
    require(binary['input_domain'] == binary['output_domain'] == 'ordinary', 'Transform interface domain changed')
    require(binary['normalization_count'] == 1 and binary['inverse_normalization_in_weights'], 'Double or missing normalization')
    require(binary['pulse_levels'] == [0, 1] and binary['weight_encoding'] == 'unsigned12', 'Weight/pulse encoding')
    require(binary['binary_tile_read_vectors'] == 1152, 'Logical read vector count')
    bits, raw = binary['input_bitplanes'], binary['tile_counts']
    require(bits.shape == (2, 12, 128) and raw.shape == (2, 12, 12, 4, 64), 'Binary trace dimensions')
    require(bits.dtype == np.uint8 and raw.dtype == np.uint8 and raw.max() <= 64, 'Binary count dtype/bound')
    require(binary['weighted_terms'].dtype == np.int64 and binary['weighted_accumulators'].dtype == np.int64,
            'Shifts/accumulation must use int64')
    for lane in range(2):
        for slot in range(12):
            for bit in range(12):
                for tile, (row, col) in enumerate(((0, 0), (0, 64), (64, 0), (64, 64))):
                    target = (bits[lane, slot, row:row+64, None]
                              & engine.physical_planes[bit, row:row+64, col:col+64]).sum(axis=0, dtype=np.int64)
                    require(np.array_equal(raw[lane, slot, bit, tile], target), 'Raw AND/popcount mismatch')
    lanes = np.asarray(vector, dtype=np.int64).reshape(128, 2)
    integer_oracle = (weight @ lanes).T
    require(np.array_equal(binary['weighted_accumulators'], integer_oracle), 'Exact integer sum reconstruction')
    require(np.array_equal(redc['input'], integer_oracle), 'REDC must consume the complete unreduced dot')
    require(np.array_equal(binary['montgomery_correction'], redc['correction_m']), 'Correction trace disagreement')
    require(np.array_equal(binary['redc_quotient'], redc['quotient']), 'Quotient trace disagreement')
    require(result == redc['output'].T.reshape(256).tolist(), 'Output lane interleave')
    return {'raw_column_counts_independently_recomputed': raw.size,
            'binary_input_bits_verified': bits.size, 'logical_tile_read_vectors': 1152,
            'separate_redc_event_verified': True, 'unreduced_integer_dot_verified': True}


def verify_interfaces_and_scope(engine):
    unchanged = []
    for name, function in vars(MLKEM).items():
        if callable(function) and name not in ('__init__', 'reset_counts', '_transform'):
            require(getattr(TransformOnlyMLKEM, name) is function, 'Out-of-scope method replacement: '+name)
            unchanged.append(name)
    compare_count = 0
    for parameter in PARAMS:
        baseline = MLKEM(parameter)
        for signed in (False, True):
            events = []
            model = TransformOnlyMLKEM(parameter, signed_cbd_inputs=signed,
                                       trace=lambda stage, event: events.append((stage, event)))
            vector = [(-1 if i % 2 else 1) for i in range(256)]
            require(model.intt(vector) == baseline.intt(vector), 'Public INTT canonicalization changed')
            require(any(event.get('operation') == 'montgomery_reduce' for _, event in events),
                    'The active wrapper did not use Montgomery INTT')
            events.clear()
            require(model.ntt(vector) == baseline.ntt(vector), 'Generic forward NTT changed')
            require(not any(event.get('operation') == 'montgomery_reduce' for _, event in events),
                    'Montgomery must not run on forward NTT')
            events.clear()
            require(model.ntt(vector, stage='keygen.pke.s_hat[0]') == baseline.ntt(vector), 'CBD NTT changed')
            vmm = next(event for _, event in events if event.get('operation') == 'rram_binary_vmm')
            require(vmm['input_bit_slots'] == (3 if signed else 12), 'Forward CBD schedule changed')
            require(not any(event.get('operation') == 'montgomery_reduce' for _, event in events),
                    'Montgomery must not run on CBD NTT')
            compare_count += 3
            seed = bytes(range(32))+b'\x00\x01'
            require(model._sample_ntt(seed, 'audit.sample') == baseline._sample_ntt(seed, 'audit.sample'),
                    'SampleNTT changed representation or deterministic output')
    failures = [lambda: mont.redc32([-1]), lambda: mont.redc32([Q*mont.R]),
                lambda: mont.redc32([0.0]), lambda: mont.redc32([True]),
                lambda: mont.redc32(np.asarray([2**64-1], dtype=np.uint64)),
                lambda: engine.transform([-1]*256), lambda: engine.transform([Q]*256),
                lambda: engine.transform([0]*255), lambda: engine.transform([0.5]*256)]
    for action in failures:
        try:
            action()
        except (ValueError, TypeError):
            continue
        raise ValueError('Input range/type guard accepted an invalid value')
    tree = ast.parse(textwrap.dedent(inspect.getsource(mont.MontgomeryINTT.transform)))
    require(not any(isinstance(node, ast.MatMult) for node in ast.walk(tree)), 'Dense @ in runtime transform')
    forbidden = {'dot', 'matmul', 'einsum', 'tensordot'}
    require(not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in forbidden for node in ast.walk(tree)), 'Dense shortcut in runtime transform')
    return {'inherited_method_identity_checks': len(unchanged), 'unchanged_methods': sorted(unchanged),
            'wrapper_forward_inverse_comparisons': compare_count, 'sample_ntt_comparisons': 6,
            'invalid_inputs_rejected': len(failures), 'runtime_dense_shortcut_absent': True}


def run_verification():
    output = ROOT/'transform_only_results/intt_montgomery_review.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), ROOT/'rram_intt_montgomery.py', ROOT/'rram_transform_only.py',
             ROOT/'mlkem.py', ROOT/'binary_rram.py', ROOT/'matrices/mlkem_shared_matrices.npz',
             ROOT/'rram_binary/mlkem_binary_weights.npz',
             ROOT/'rram_intt_montgomery_weights/intt_montgomery_weights.npz',
             ROOT/'rram_intt_montgomery_weights/provenance.json']
    before = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    report = {'status': 'running', 'scope': 'INTT-only RV+REDC32 boundary audit; no full official vector rerun',
              'source_sha256': before}
    try:
        weight, inverse, report['weights'] = verify_weights()
        report['redc32'] = verify_redc()
        engine = mont.MontgomeryINTT()
        report['transforms'] = verify_transforms(engine, inverse)
        report['trace'] = verify_trace(engine, weight)
        report['scope_and_guards'] = verify_interfaces_and_scope(engine)
        require(all(sha(ROOT/path) == digest for path, digest in before.items()),
                'A source/weight artifact changed while the independent audit was running')
        report['source_and_weight_hashes_unchanged'] = True
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=type(exc).__name__+': '+str(exc))
        output.write_text(json.dumps(report, indent=2)+'\n')
        raise
    output.write_text(json.dumps(report, indent=2)+'\n')
    print('Independent INTT Montgomery audit passed: '+str(output), flush=True)
    return report


def main():
    return run_verification()


if __name__ == '__main__':
    main()

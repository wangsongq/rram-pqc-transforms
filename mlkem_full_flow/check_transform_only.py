#!/usr/bin/env python3
"""Check the INTT-only Montgomery fusion against official ML-KEM vectors.

Forward NTT and SampleNTT keep ordinary representations. INTT alone stores
R*V and returns ordinary residues through REDC32. Digital PKE/KEM methods are
inherited unchanged.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from mlkem import MLKEM
from rram_transform_only import TransformOnlyMLKEM
from validation import execute_case, read_json, require, test_lookup, verify_sources

BASE = Path(__file__).resolve().parent
OUT = BASE / 'transform_only_results'


def linked_boundary_checks():
    """Compare complete linked flows and every NTT/SampleNTT/INTT interface."""
    results = []
    watched = {'ntt', 'intt', 'sample_ntt'}
    for name in ('ML-KEM-512', 'ML-KEM-768', 'ML-KEM-1024'):
        d, z, m = (hashlib.sha256((name + ':' + label).encode()).digest()
                   for label in ('d', 'z', 'm'))
        expected_events = []
        def reference_sink(stage, values):
            if values.get('operation') in watched:
                expected_events.append((stage, values['operation'], list(values['output'])))
        reference = MLKEM(name, trace=reference_sink)
        ek, dk = reference.keygen(d, z)
        shared, ciphertext = reference.encaps(ek, m)
        require(reference.decaps(dk, ciphertext) == shared, 'Reference linked flow')
        tampered = bytes([ciphertext[0] ^ 1]) + ciphertext[1:]
        rejected = reference.decaps(dk, tampered)
        for signed in (False, True):
            events, artifacts = [], {'ntt': set(), 'intt': set()}
            def candidate_sink(stage, values):
                if values.get('operation') in watched:
                    output = list(values['output'])
                    require(all(0 <= value < 3329 for value in output),
                            'All transform/SampleNTT boundaries remain ordinary canonical residues')
                    events.append((stage, values['operation'], output))
                if values.get('operation') == 'rram_binary_vmm':
                    artifacts[values['transform']].add(values['weight_artifact'])
            kem = TransformOnlyMLKEM(name, trace=candidate_sink, signed_cbd_inputs=signed)
            actual_ek, actual_dk = kem.keygen(d, z)
            require((actual_ek, actual_dk) == (ek, dk), 'Linked key generation bytes')
            actual_shared, actual_ciphertext = kem.encaps(actual_ek, m)
            require((actual_shared, actual_ciphertext) == (shared, ciphertext),
                    'Linked encapsulation bytes')
            require(kem.decaps(actual_dk, actual_ciphertext) == shared, 'Linked decapsulation')
            require(kem.decaps(actual_dk, tampered) == rejected, 'Linked implicit rejection')
            require(events == expected_events, 'Every ordinary NTT/SampleNTT/INTT boundary')
            require(artifacts['ntt'] == {'rram_binary/mlkem_binary_weights.npz'},
                    'Forward NTT must retain its original binary weights')
            require(len(artifacts['intt']) == 1 and
                    next(iter(artifacts['intt'])).startswith('rram_intt_montgomery_weights/'),
                    'Inverse transform must actually use its fused binary artifact')
            results.append({'parameter_set': name, 'profile': kem.profile, 'passed': True,
                            'ordinary_boundary_checks': len(events),
                            'weight_artifacts': {key: sorted(value) for key, value in artifacts.items()}})
    return results


def run():
    OUT.mkdir(exist_ok=True)
    summary_path = OUT / 'summary.json'
    summary = {'status': 'running', 'scope': 'RRAM NTT unchanged; only INTT fuses R into already normalized V, with REDC32 and ordinary interfaces. Exact ideal binary counts, not hardware performance.'}
    def save():
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    save()
    try:
        methods = ('_pke_keygen', '_pke_encrypt', '_pke_decrypt', '_dot', '_noise',
                   '_sample_ntt', '_matrix', '_hash', '_check_ek', 'keygen', 'encaps', 'decaps')
        for name in methods:
            require(getattr(TransformOnlyMLKEM, name) is getattr(MLKEM, name),
                    'Digital method must be inherited unchanged: ' + name)
        provenance = verify_sources()
        rows = []
        for dataset in provenance['vector_sets']:
            folder = BASE / 'golden_vectors' / dataset['name']
            prompt = read_json(folder / 'prompt.json')
            expected = test_lookup(read_json(folder / 'expectedResults.json'))
            for group in prompt['testGroups']:
                function = group.get('function', prompt['mode'])
                for signed in (False, True):
                    kem = TransformOnlyMLKEM(group['parameterSet'], signed_cbd_inputs=signed)
                    for case in group['tests']:
                        target = expected[group['tgId'], case['tcId']]
                        actual = execute_case(kem, function, case)
                        matches = []
                        for key, value in target.items():
                            if key == 'tcId':
                                continue
                            golden = bytes.fromhex(value) if isinstance(value, str) else value
                            matches.append(type(actual.get(key)) is type(golden) and actual[key] == golden)
                        rows.append({'corpus': dataset['name'], 'parameter_set': group['parameterSet'],
                                     'profile': kem.profile, 'intt_fusion': 'R*V + REDC32', 'function': function,
                                     'tgId': group['tgId'], 'tcId': case['tcId'],
                                     'passed': bool(matches) and all(matches)})
        with (OUT / 'golden_checks.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        require(all(row['passed'] for row in rows), 'Official byte/boolean checks')
        reference = MLKEM('ML-KEM-512')
        rng = np.random.default_rng(203)
        vectors = [np.zeros(256, dtype=np.int64), np.full(256, 3328, dtype=np.int64)]
        vectors += [rng.integers(-6658, 6659, 256) for _ in range(8)]
        checks = 0
        for signed in (False, True):
            kem = TransformOnlyMLKEM('ML-KEM-512', signed_cbd_inputs=signed)
            for vector in vectors:
                require(kem.ntt(vector) == reference.ntt(vector), 'General forward transform')
                require(kem.intt(vector) == reference.intt(vector), 'General inverse transform')
                checks += 2
            for value in range(-3, 4):
                source = [value] * 256
                require(kem.ntt(source, stage='keygen.pke.s_hat[0]') == reference.ntt(source),
                        'Known CBD stage signed encoding')
                checks += 1
        linked = linked_boundary_checks()
        (OUT / 'linked_flow_checks.json').write_text(json.dumps(linked, indent=2) + '\n')
        summary.update(status='passed', official_unique_cases=provenance['total_test_cases'],
                       executions=len(rows), passed=sum(row['passed'] for row in rows),
                       digital_methods_identical=list(methods), transform_comparisons=checks,
                       profiles=['unsigned12', 'signed_cbd3'],
                       intt_fusion={'R': 2**32, 'stored_weight': '(R * V) mod 3329',
                                    'V_includes_inverse_128': True, 'reduction': 'REDC32',
                                    'input_output': 'ordinary canonical residues'},
                       linked_flow_checks=len(linked),
                       linked_ordinary_boundary_checks=sum(row['ordinary_boundary_checks'] for row in linked),
                       source_sha256={str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in (Path(__file__), BASE/'validation.py', BASE/'rram_transform_only.py', BASE/'mlkem.py',
                                                BASE/'rram_intt_montgomery.py',
                                                BASE/'binary_rram.py', BASE/'rram_binary/mlkem_binary_weights.npz',
                                                BASE/'golden_vectors/provenance.json',
                                                *sorted((BASE/'rram_intt_montgomery_weights').glob('*.npz')),
                                                BASE/'rram_intt_montgomery_weights/provenance.json')})
        save()
        print(json.dumps({key: summary[key] for key in ('status', 'executions', 'passed',
                                                       'transform_comparisons', 'linked_flow_checks',
                                                       'linked_ordinary_boundary_checks')}, indent=2))
    except Exception as error:
        summary.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        save()
        raise
    return summary


if __name__ == '__main__':
    run()

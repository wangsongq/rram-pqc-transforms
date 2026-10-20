"""Complete Round-3 LightSaber/Saber/FireSaber KEM, with inspectable arithmetic.

Targets the original SHAKE/SHA3, centered-binomial Saber family, not the 90s
or uniform variants. Source algorithms and byte layout follow the pinned
official KULeuven-COSIC/SABER reference recorded in core_provenance.json.
Only negacyclic polynomial dot products vary across schoolbook, matrix and
rram_binary backends; sampling, packing, rounding and the FO wrapper are shared.

This is a readable research reference, not constant-time production crypto or
a validated cryptographic module. Traces contain keys and secret randomness.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import hmac
from numbers import Integral
import secrets
from typing import Iterable

import numpy as np


N, Q, P, EQ, EP = 256, 8192, 1024, 13, 10
OFFICIAL_COMMIT = "f7f39e4db2f3e22a21e1dd635e0601caae2b4510"


@dataclass(frozen=True)
class Parameters:
    name: str
    k: int
    mu: int
    et: int
    security_category: int
    n: int = N
    q: int = Q
    p: int = P
    eq: int = EQ
    ep: int = EP

    @property
    def eta(self):
        return self.mu // 2

    @property
    def h1(self):
        return 1 << (EQ - EP - 1)

    @property
    def h2(self):
        return (1 << (EP - 2)) - (1 << (EP - self.et - 1)) + self.h1

    @property
    def ek_len(self):
        return 320 * self.k + 32

    @property
    def dk_pke_len(self):
        return 416 * self.k

    @property
    def dk_len(self):
        return self.dk_pke_len + self.ek_len + 64

    @property
    def ct_len(self):
        return 320 * self.k + 32 * self.et


PARAMS = {
    "LightSaber": Parameters("LightSaber", 2, 10, 3, 1),
    "Saber": Parameters("Saber", 3, 8, 4, 3),
    "FireSaber": Parameters("FireSaber", 4, 6, 6, 5),
}


def _bytes(value, length, name):
    if not isinstance(value, (bytes, bytearray)) or len(value) != length:
        raise ValueError(f"{name} must be a byte array of length {length}")
    return bytes(value)


def pack_poly(values: Iterable[int], bits: int) -> bytes:
    """Official POLq/POLp/POLT/POLmsg layout: consecutive little-endian fields."""
    values = list(values)
    if not 1 <= bits <= 13 or len(values) != N or any(not isinstance(x, Integral) for x in values):
        raise ValueError("packing requires 256 integers and a width in 1..13")
    mask = (1 << bits) - 1
    packed = sum((int(x) & mask) << (bits * i) for i, x in enumerate(values))
    return packed.to_bytes(32 * bits, "little")


def unpack_poly(data: bytes, bits: int) -> list[int]:
    """Inverse of the official consecutive, low-bit-first field packing."""
    if not 1 <= bits <= 13:
        raise ValueError("unpacking width must be in 1..13")
    data = _bytes(data, 32 * bits, "encoded polynomial")
    packed, mask = int.from_bytes(data, "little"), (1 << bits) - 1
    return [(packed >> (bits * i)) & mask for i in range(N)]


def schoolbook_dot(left, right, modulus):
    """Independent direct convolution in Z_modulus[X]/(X^256+1).

    The outer loop explicitly accumulates every shifted row of the ordinary
    convolution. NumPy vectorizes those exact integer multiply/add operations;
    no NTT, FFT, transform matrix or shared auxiliary-engine routine is used.
    Right operands are independently centered modulo q=8192, including when
    the target modulus is p=1024. Canonical secret-key fields thus recover CBD.
    Returns (canonical output, per-product intermediate dictionaries).
    """
    if modulus not in (Q, P):
        raise ValueError("Saber polynomial modulus must be 8192 or 1024")
    if len(left) != len(right) or not 1 <= len(left) <= 4:
        raise ValueError("a Saber dot requires 1..4 polynomial pairs")
    total = np.zeros(N, dtype=np.int64)
    records = []
    for a_input, b_input in zip(left, right):
        if len(a_input) != N or len(b_input) != N:
            raise ValueError("polynomials must contain 256 coefficients")
        if any(not isinstance(x, Integral) for x in list(a_input) + list(b_input)):
            raise ValueError("polynomial coefficients must be integers")
        # Separate centering moduli are intentional. Right secret coefficients
        # are represented modulo 8192 even in the inner product modulo 1024.
        a = np.array([(int(x) + modulus // 2) % modulus - modulus // 2 for x in a_input], dtype=np.int64)
        b = np.array([(int(x) + Q // 2) % Q - Q // 2 for x in b_input], dtype=np.int64)
        if np.any(np.abs(b) > 5):
            raise ValueError("right polynomial is outside the supported Saber CBD range [-5,5]")
        convolution = np.zeros(2 * N, dtype=np.int64)
        for i in range(N):
            convolution[i:i + N] += a[i] * b
        folded = convolution[:N] - convolution[N:]
        total += folded
        records.append(dict(left_centered=a, right_centered=b,
                            ordinary_convolution=convolution, negacyclic_product=folded,
                            accumulated_integer_sum=total.copy()))
    return (total % modulus).tolist(), records


class Saber:
    """Complete deterministic KEM API, optionally drawing omitted input seeds.

    keygen(seed_a,seed_s,z) consumes three raw 32-byte draws in official order.
    encaps(ek,m) consumes the raw 32-byte draw before the official message hash.
    decaps(dk,c) performs reencryption and the official implicit-rejection KDF.
    A decapsulation key is expected to encode a valid small Saber CBD secret.
    """

    def __init__(self, name="Saber", backend="schoolbook", trace=None):
        if name not in PARAMS:
            raise ValueError(f"unknown Saber parameter set {name!r}")
        if backend not in ("schoolbook", "matrix", "rram_binary"):
            raise ValueError("backend must be schoolbook, matrix or rram_binary")
        if trace is not None and not callable(trace):
            raise ValueError("trace must be callable or None")
        self.name, self.backend, self.trace = name, backend, trace
        self.params = self.p = PARAMS[name]
        self.engine = None
        if backend != "schoolbook":
            from auxiliary_ntt import SaberPolynomialEngine
            sink = (lambda stage, values: self._emit(stage, **values)) if trace is not None else None
            self.engine = SaberPolynomialEngine(backend=backend, trace=sink)
        self.reset_counts()

    def reset_counts(self):
        self.counts = dict(poly_dot=0, polynomial_products=0)
        self.phase_counts = {phase: dict(poly_dot=0, polynomial_products=0)
                             for phase in ("keygen", "encaps", "decaps", "other")}
        if self.engine is not None:
            self.engine.reset_counts()

    def _emit(self, stage, **values):
        if self.trace is not None:
            self.trace(stage, deepcopy(values))

    def _hash(self, kind, data, stage, output_bytes=None):
        if kind == "SHA3-256":
            output = hashlib.sha3_256(data).digest()
        elif kind == "SHA3-512":
            output = hashlib.sha3_512(data).digest()
        elif kind == "SHAKE128":
            output = hashlib.shake_128(data).digest(output_bytes)
        else:
            raise ValueError("unknown hash operation")
        self._emit(stage, operation=kind, input=data, output=output)
        return output

    def _matrix(self, seed_a, stage):
        stream = self._hash("SHAKE128", seed_a, stage + ".shake128", 416 * self.p.k**2)
        matrix = []
        for i in range(self.p.k):
            row = []
            for j in range(self.p.k):
                offset = 416 * (i * self.p.k + j)
                encoded = stream[offset:offset + 416]
                poly = unpack_poly(encoded, EQ)
                self._emit(f"{stage}.A[{i},{j}]", operation="matrix_sample", seed=seed_a,
                           row=i, column=j, byte_offset=offset, input=encoded, output=poly)
                row.append(poly)
            matrix.append(row)
        self._emit(stage, operation="gen_matrix", seed=seed_a, xof_bytes=stream, A=matrix)
        return matrix

    def _secret(self, seed, stage):
        width = 32 * self.p.mu
        stream = self._hash("SHAKE128", seed, stage + ".shake128", self.p.k * width)
        eta = self.p.eta
        result = []
        for i in range(self.p.k):
            data = stream[i * width:(i + 1) * width]
            bits = [(byte >> j) & 1 for byte in data for j in range(8)]
            positive, negative, signed = [], [], []
            for coefficient in range(N):
                offset = self.p.mu * coefficient
                positive.append(sum(bits[offset:offset + eta]))
                negative.append(sum(bits[offset + eta:offset + 2 * eta]))
                signed.append(positive[-1] - negative[-1])
            self._emit(f"{stage}[{i}]", operation="sample_cbd", seed=seed, mu=self.p.mu, eta=eta,
                       input=data, bits=bits, positive=positive, negative=negative, output=signed,
                       signed=signed, canonical_q=[x % Q for x in signed])
            result.append(signed)
        self._emit(stage, operation="gen_secret", seed=seed, xof_bytes=stream, output=result,
                   representation="signed CBD; secret serialization keeps the low thirteen bits")
        return result

    def _dot(self, left, right, modulus, stage):
        terms = len(left)
        phase = stage.split(".", 1)[0]
        if phase not in self.phase_counts:
            phase = "other"
        self.counts["poly_dot"] += 1
        self.counts["polynomial_products"] += terms
        self.phase_counts[phase]["poly_dot"] += 1
        self.phase_counts[phase]["polynomial_products"] += terms
        if self.backend == "schoolbook":
            output, records = schoolbook_dot(left, right, modulus)
            for i, record in enumerate(records):
                self._emit(f"{stage}.product[{i}]", operation="schoolbook_negacyclic_product",
                           modulus=modulus, left=left[i], right=right[i], **record)
        else:
            output = self.engine.dot(left, right, modulus, stage=stage + ".engine")
            output = [int(x) for x in output]
        self._emit(stage, operation="polynomial_dot", backend=self.backend, modulus=modulus,
                   left=left, right=right, output=output, terms=terms, phase=phase)
        return output

    def _pke_keygen(self, seed_a_raw, seed_s, stage):
        # Official keypair hashes the first RNG draw through SHAKE128 to 32 bytes.
        seed_a = self._hash("SHAKE128", seed_a_raw, stage + ".seed_a_hash", 32)
        matrix = self._matrix(seed_a, stage + ".matrix")
        secret = self._secret(seed_s, stage + ".s")
        # KeyGen uses A^T; encryption below uses A without a transpose.
        b_q = [self._dot([matrix[j][i] for j in range(self.p.k)], secret, Q, f"{stage}.b_q[{i}]")
               for i in range(self.p.k)]
        b_p = [[((x + self.p.h1) >> (EQ - EP)) & (P - 1) for x in poly] for poly in b_q]
        dk_pke = b"".join(pack_poly(poly, EQ) for poly in secret)
        ek = b"".join(pack_poly(poly, EP) for poly in b_p) + seed_a
        self._emit(stage + ".rounding", operation="round_q_to_p", input=b_q,
                   h1=self.p.h1, right_shift=EQ - EP, output_mask=P - 1, output=b_p)
        self._emit(stage + ".output", operation="pke_keygen", seed_a_raw=seed_a_raw, seed_a=seed_a,
                   seed_s=seed_s, A=matrix, s_signed=secret, b_q=b_q, b_p=b_p,
                   dk_pke=dk_pke, ek=ek)
        return ek, dk_pke

    def _pke_encrypt(self, ek, message, coins, stage):
        k = self.p.k
        seed_a = ek[320 * k:]
        b = [unpack_poly(ek[320 * i:320 * (i + 1)], EP) for i in range(k)]
        self._emit(stage + ".input", ek=ek, message=message, coins=coins, seed_a=seed_a, b_p=b)
        matrix = self._matrix(seed_a, stage + ".matrix")
        secret = self._secret(coins, stage + ".sp")
        bp_q = [self._dot(matrix[i], secret, Q, f"{stage}.bp_q[{i}]") for i in range(k)]
        bp_p = [[((x + self.p.h1) >> (EQ - EP)) & (P - 1) for x in poly] for poly in bp_q]
        v_p = self._dot(b, secret, P, stage + ".v_p")
        message_bits = unpack_poly(message, 1)
        embedded_message = [bit << (EP - 1) for bit in message_bits]
        before_rounding = [x - mu + self.p.h1 for x, mu in zip(v_p, embedded_message)]
        c_m = [(x >> (EP - self.p.et)) & ((1 << self.p.et) - 1) for x in before_rounding]
        c1 = b"".join(pack_poly(poly, EP) for poly in bp_p)
        c2 = pack_poly(c_m, self.p.et)
        c = c1 + c2
        self._emit(stage + ".round_q_to_p", operation="round_q_to_p", input=bp_q,
                   h1=self.p.h1, right_shift=EQ - EP, output_mask=P - 1, output=bp_p)
        self._emit(stage + ".message_rounding", operation="message_subtract_and_round_p_to_t",
                   v_p=v_p, message_bits=message_bits, embedded_message=embedded_message,
                   h1=self.p.h1, input=before_rounding, right_shift=EP - self.p.et,
                   output_mask=(1 << self.p.et) - 1, output=c_m)
        self._emit(stage + ".output", operation="pke_encrypt", A=matrix, sp_signed=secret,
                   bp_q=bp_q, bp_p=bp_p, v_p=v_p, message_bits=message_bits,
                   c_m=c_m, c1=c1, c2=c2, c=c)
        return c

    def _pke_decrypt(self, dk_pke, c, stage):
        k = self.p.k
        secret_q = [unpack_poly(dk_pke[416 * i:416 * (i + 1)], EQ) for i in range(k)]
        bp_p = [unpack_poly(c[320 * i:320 * (i + 1)], EP) for i in range(k)]
        c_m = unpack_poly(c[320 * k:], self.p.et)
        self._emit(stage + ".input", dk_pke=dk_pke, c=c, s_canonical_q=secret_q, bp_p=bp_p, c_m=c_m)
        v_p = self._dot(bp_p, secret_q, P, stage + ".v_p")
        expanded_c_m = [x << (EP - self.p.et) for x in c_m]
        before_rounding = [x + self.p.h2 - cm for x, cm in zip(v_p, expanded_c_m)]
        message_bits = [(x >> (EP - 1)) & 1 for x in before_rounding]
        message = pack_poly(message_bits, 1)
        self._emit(stage + ".rounding", operation="recover_message", v_p=v_p,
                   c_m=c_m, expanded_c_m=expanded_c_m, h2=self.p.h2,
                   input=before_rounding, right_shift=EP - 1, output_mask=1, output=message_bits)
        self._emit(stage + ".output", operation="pke_decrypt", v_p=v_p,
                   message_bits=message_bits, message=message, m_prime=message)
        return message

    def pke_decrypt(self, dk_pke, c):
        """Recover the 32-byte PKE message without KEM reencryption checks."""
        dk_pke = _bytes(dk_pke, self.p.dk_pke_len, "PKE secret key")
        c = _bytes(c, self.p.ct_len, "ciphertext")
        return self._pke_decrypt(dk_pke, c, "pke_decrypt")

    def keygen(self, seed_a=None, seed_s=None, z=None):
        """Return (public_key, secret_key) using three raw 32-byte draws.

        Supplying the draws reproduces official deterministic KAT inputs;
        omitted draws use secrets.token_bytes. The public matrix seed is
        processed internally exactly as in the Round-3 reference.
        """
        seed_a = _bytes(secrets.token_bytes(32) if seed_a is None else seed_a, 32, "raw seed_a")
        seed_s = _bytes(secrets.token_bytes(32) if seed_s is None else seed_s, 32, "seed_s")
        z = _bytes(secrets.token_bytes(32) if z is None else z, 32, "z")
        self._emit("keygen.input", parameter_set=self.name, seed_a_raw=seed_a, seed_s=seed_s, z=z)
        ek, dk_pke = self._pke_keygen(seed_a, seed_s, "keygen.pke")
        h = self._hash("SHA3-256", ek, "keygen.H_ek")
        dk = dk_pke + ek + h + z
        self._emit("keygen.output", ek=ek, dk_pke=dk_pke, h=h, z=z, dk=dk)
        return ek, dk

    def encaps(self, ek, m=None):
        """Return (shared_key, ciphertext) for a serialized public key.

        ``m`` is raw 32-byte randomness, not a prehashed plaintext. The official
        SHA3-256 preprocessing and reencryption randomness derivation occur here.
        """
        ek = _bytes(ek, self.p.ek_len, "encapsulation key")
        m_raw = _bytes(secrets.token_bytes(32) if m is None else m, 32, "raw message randomness")
        self._emit("encaps.input", parameter_set=self.name, ek=ek, m_raw=m_raw)
        message = self._hash("SHA3-256", m_raw, "encaps.H_random_m")
        h = self._hash("SHA3-256", ek, "encaps.H_ek")
        kr = self._hash("SHA3-512", message + h, "encaps.G")
        prekey, coins = kr[:32], kr[32:]
        self._emit("encaps.derive", message=message, h=h, prekey=prekey, coins=coins)
        c = self._pke_encrypt(ek, message, coins, "encaps.pke")
        hc = self._hash("SHA3-256", c, "encaps.H_c")
        K = self._hash("SHA3-256", prekey + hc, "encaps.KDF")
        self._emit("encaps.output", K=K, c=c, prekey=prekey, h_c=hc)
        return K, c

    def decaps(self, dk, c):
        """Return a 32-byte shared key, using implicit rejection on mismatch.

        Reencryption uses the recovered message. An altered ciphertext selects
        the stored fallback z before the final SHA3-256 KDF; it does not return
        an explicit rejection flag. Malformed byte lengths raise ValueError.
        """
        dk = _bytes(dk, self.p.dk_len, "decapsulation key")
        c = _bytes(c, self.p.ct_len, "ciphertext")
        boundary = self.p.dk_pke_len
        dk_pke, ek = dk[:boundary], dk[boundary:boundary + self.p.ek_len]
        h, z = dk[-64:-32], dk[-32:]
        self._emit("decaps.input", parameter_set=self.name, dk=dk, c=c, dk_pke=dk_pke, ek=ek, h=h, z=z)
        # Official Saber uses the stored public-key hash directly. Unlike the
        # ML-KEM public wrapper, it does not recompute and validate H(ek) here.
        message = self._pke_decrypt(dk_pke, c, "decaps.decrypt")
        kr = self._hash("SHA3-512", message + h, "decaps.G")
        prekey, coins = kr[:32], kr[32:]
        self._emit("decaps.derive", m_prime=message, h=h, prekey=prekey, coins=coins)
        c_prime = self._pke_encrypt(ek, message, coins, "decaps.reencrypt")
        accepted = hmac.compare_digest(c, c_prime)
        hc = self._hash("SHA3-256", c, "decaps.H_c")
        mask = -int(accepted) & 255
        selected_prekey = bytes((a & mask) | (b & (mask ^ 255)) for a, b in zip(prekey, z))
        K = self._hash("SHA3-256", selected_prekey + hc, "decaps.KDF")
        self._emit("decaps.output", operation="implicit_rejection", c=c, c_prime=c_prime,
                   accepted=accepted, m_prime=message, prekey=prekey, fallback_prekey=z,
                   selected_prekey=selected_prekey, h_c=hc, K=K)
        return K

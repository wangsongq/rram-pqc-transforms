"""Readable FIPS 203 ML-KEM reference with complete intermediate-value tracing.

Algorithms 3--21, final FIPS 203 (2024-08-13), for all three parameter sets.
The ``butterfly`` backend implements Algorithms 9 and 10. The ``matrix``
backend applies the same integer transform to the even and odd coefficient
lanes using the supplied W128 / W128_inv matrices. The ``rram_binary`` backend
uses 0/1 weight/input planes, tiled binary AND/popcount, then digital shift/add.
All backends use the same K-PKE and Fujisaki--Okamoto wrapper, and model exact
logical arithmetic rather than analog device behavior.

This research reference is not constant-time or a validated cryptographic
module. An enabled trace contains private keys, randomness, and shared secrets.
Provenance and original source hashes are recorded in core_provenance.json.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import hmac
from numbers import Integral
from pathlib import Path
import secrets
from typing import Callable, Iterable


N = 256
Q = 3329
ZETA = 17
INV128 = 3303
MATRIX_PATH = Path(__file__).resolve().parent / "matrices/mlkem_shared_matrices.npz"
MATRIX_SHA256 = "b96b6c339da9c9599319094c8171603cba2fb6a338a823790be762f40c8de233"


@dataclass(frozen=True)
class Parameters:
    """FIPS 203 parameter set; lengths are serialized byte counts."""
    name: str
    k: int
    eta1: int
    eta2: int
    du: int
    dv: int
    security_category: int
    n: int = N
    q: int = Q
    zeta: int = ZETA

    @property
    def ek_len(self) -> int:
        return 384 * self.k + 32

    @property
    def dk_len(self) -> int:
        return 768 * self.k + 96

    @property
    def ct_len(self) -> int:
        return 32 * (self.du * self.k + self.dv)


PARAMS = {
    "ML-KEM-512": Parameters("ML-KEM-512", 2, 3, 2, 10, 4, 1),
    "ML-KEM-768": Parameters("ML-KEM-768", 3, 2, 2, 10, 4, 3),
    "ML-KEM-1024": Parameters("ML-KEM-1024", 4, 2, 2, 11, 5, 5),
}


def bitrev7(value: int) -> int:
    return int(f"{value:07b}"[::-1], 2)


ZETAS = tuple(pow(ZETA, bitrev7(i), Q) for i in range(128))
GAMMAS = tuple(pow(ZETA, 2 * bitrev7(i) + 1, Q) for i in range(128))


def _bytes(value: bytes, length: int, name: str) -> bytes:
    if not isinstance(value, (bytes, bytearray)) or len(value) != length:
        raise ValueError(f"{name} must be a byte array of length {length}")
    return bytes(value)


def _poly(values: Iterable[int]) -> list[int]:
    values = list(values)
    if len(values) != N or any(not isinstance(x, Integral) for x in values):
        raise ValueError("a polynomial must contain exactly 256 integer coefficients")
    return [int(x) % Q for x in values]


def byte_encode(values: Iterable[int], d: int) -> bytes:
    """Algorithm 5: little-endian d-bit encoding of one 256-coefficient array."""
    if not 1 <= d <= 12:
        raise ValueError("encoding width must be in 1..12")
    values = list(values)
    if len(values) != N:
        raise ValueError("byte_encode expects 256 coefficients")
    modulus = Q if d == 12 else 1 << d
    packed = sum((int(x) % modulus) << (d * i) for i, x in enumerate(values))
    return packed.to_bytes(32 * d, "little")


def byte_decode(data: bytes, d: int) -> list[int]:
    """Algorithm 6; d=12 reduces modulo q as specified in FIPS 203."""
    if not 1 <= d <= 12:
        raise ValueError("decoding width must be in 1..12")
    data = _bytes(data, 32 * d, "encoded polynomial")
    packed = int.from_bytes(data, "little")
    mask = (1 << d) - 1
    modulus = Q if d == 12 else 1 << d
    return [((packed >> (d * i)) & mask) % modulus for i in range(N)]


def compress(values: Iterable[int], d: int) -> list[int]:
    """Equation 4.7: exact nearest-integer rounding followed by mod 2**d."""
    if not 1 <= d < 12:
        raise ValueError("compression width must be in 1..11")
    return [(((int(x) % Q) << d) + Q // 2) // Q % (1 << d) for x in values]


def decompress(values: Iterable[int], d: int) -> list[int]:
    """Equation 4.8: exact nearest-integer rounding, with half rounded up."""
    if not 1 <= d < 12:
        raise ValueError("decompression width must be in 1..11")
    return [(Q * (int(y) % (1 << d)) + (1 << (d - 1))) >> d for y in values]


def multiply_ntts(f: Iterable[int], g: Iterable[int]) -> list[int]:
    """Algorithms 11--12: 128 quadratic base multiplications, not pointwise."""
    f, g = _poly(f), _poly(g)
    product = [0] * N
    for i, gamma in enumerate(GAMMAS):
        a0, a1 = f[2 * i:2 * i + 2]
        b0, b1 = g[2 * i:2 * i + 2]
        product[2 * i] = (a0 * b0 + gamma * a1 * b1) % Q
        product[2 * i + 1] = (a0 * b1 + a1 * b0) % Q
    return product


def _add(a: list[int], b: list[int]) -> list[int]:
    return [(x + y) % Q for x, y in zip(a, b)]


@lru_cache(maxsize=1)
def _load_matrices():
    """Load the byte-identical project matrices only for the matrix backend."""
    import numpy as np

    if hashlib.sha256(MATRIX_PATH.read_bytes()).hexdigest() != MATRIX_SHA256:
        raise ValueError("ML-KEM matrix artifact SHA-256 does not match provenance")
    with np.load(MATRIX_PATH, allow_pickle=False) as data:
        forward = data["W128"].copy()
        inverse = data["W128_inv"].copy()
    for matrix in (forward, inverse):
        if matrix.shape != (128, 128) or matrix.dtype != np.dtype("int64"):
            raise ValueError("matrix must be 128 by 128 with int64 coefficients")
        if np.any(matrix < 0) or np.any(matrix >= Q):
            raise ValueError("matrix coefficients must be canonical residues")
        matrix.flags.writeable = False
    # Every dot product is at most 128*(q-1)**2 = 1,417,674,752.
    # Hence int64 gives exact integer arithmetic, with ample overflow margin.
    return forward, inverse


class MLKEM:
    """Full ML-KEM with reproducible inputs and an optional (stage, values) sink.

    ``keygen(d, z)`` and ``encaps(ek, m)`` accept explicit 32-byte randomness.
    Omitted randomness uses secrets.token_bytes, without any validation claim
    about the platform RNG. ``trace`` receives immutable snapshots in the sense
    that subsequent arithmetic cannot change previously emitted list values.
    ``counts`` counts complete 256-coefficient NTT and INTT calls cumulatively.
    """

    def __init__(self, name: str = "ML-KEM-768", backend: str = "butterfly",
                 trace: Callable[[str, dict], None] | None = None):
        if name not in PARAMS:
            raise ValueError(f"unknown ML-KEM parameter set: {name!r}")
        if backend not in ("butterfly", "matrix", "rram_binary"):
            raise ValueError("backend must be 'butterfly', 'matrix' or 'rram_binary'")
        if trace is not None and not callable(trace):
            raise ValueError("trace must be callable or None")
        self.name = name
        self.params = self.p = PARAMS[name]
        self.backend = backend
        self.trace = trace
        self.counts = {"ntt": 0, "intt": 0}
        self._matrices = _load_matrices() if backend == "matrix" else None
        self._binary_rram = None
        if backend == "rram_binary":
            from binary_rram import BinaryRRAM
            self._binary_rram = BinaryRRAM()

    def reset_counts(self) -> None:
        self.counts.update(ntt=0, intt=0)

    def _emit(self, stage: str, **values) -> None:
        if self.trace is not None:
            self.trace(stage, deepcopy(values))

    def _hash(self, operation: str, data: bytes, stage: str) -> bytes:
        if operation == "H":
            output = hashlib.sha3_256(data).digest()
        elif operation == "G":
            output = hashlib.sha3_512(data).digest()
        elif operation == "J":
            output = hashlib.shake_256(data).digest(32)
        else:
            raise ValueError("unknown hash operation")
        self._emit(stage, operation=operation, input=data, output=output)
        return output

    def _transform(self, poly: Iterable[int], inverse: bool, stage: str) -> list[int]:
        source = _poly(poly)
        result = source.copy()
        operation = "intt" if inverse else "ntt"
        self.counts[operation] += 1
        call = self.counts[operation]
        if self.backend == "rram_binary":
            sink = (lambda name, values: self._emit(name, **values)) if self.trace is not None else None
            result = self._binary_rram.transform(source, inverse=inverse, trace=sink,
                                                 stage=stage + ".rram_binary")
        elif self.backend == "matrix":
            import numpy as np

            matrix = self._matrices[1 if inverse else 0]
            lanes = np.asarray(source, dtype=np.int64).reshape(128, 2)
            # Columns are the even/odd lanes; no floating point or analog noise.
            accumulators = matrix @ lanes
            result = (accumulators % Q).reshape(N).tolist()
            self._emit(stage + ".matrix", operation="matrix_vmm", transform=operation,
                       matrix="W128_inv" if inverse else "W128", input_even=source[0::2],
                       input_odd=source[1::2], accumulator_even=accumulators[:, 0].tolist(),
                       accumulator_odd=accumulators[:, 1].tolist(),
                       output_even=result[0::2], output_odd=result[1::2],
                       inverse_normalization_in_matrix=inverse)
        else:
            index = 127 if inverse else 1
            length = 2 if inverse else 128
            while (length <= 128 if inverse else length >= 2):
                for start in range(0, N, 2 * length):
                    zeta = ZETAS[index]
                    index += -1 if inverse else 1
                    for j in range(start, start + length):
                        a, b = result[j], result[j + length]
                        if inverse:
                            result[j] = (a + b) % Q
                            result[j + length] = zeta * (b - a) % Q
                        else:
                            t = zeta * b % Q
                            result[j] = (a + t) % Q
                            result[j + length] = (a - t) % Q
                self._emit(stage + f".layer{length}", operation="butterfly_layer",
                           transform=operation, length=length, output=result)
                length = length * 2 if inverse else length // 2
            if inverse:
                result = [x * INV128 % Q for x in result]
        self._emit(stage, operation=operation, backend=self.backend, call=call,
                   input=source, output=result)
        return result

    def ntt(self, poly: Iterable[int], stage: str = "ntt") -> list[int]:
        """Transform 256 integer coefficients; return 256 canonical residues.

        Both 128-coefficient parity lanes count as one complete transform.
        ``stage`` labels trace records; callers normally leave its default.
        """
        return self._transform(poly, False, stage)

    def intt(self, poly: Iterable[int], stage: str = "intt") -> list[int]:
        """Invert one complete transform, including the 128**-1 normalization."""
        return self._transform(poly, True, stage)

    def _sample_ntt(self, seed: bytes, stage: str) -> list[int]:
        """Algorithm 7, with an extensible SHAKE128 prefix (no fixed byte cap)."""
        xof = hashlib.shake_128(seed)
        stream = xof.digest(504)
        offset = 0
        result, candidates, accepted = [], [], []
        while len(result) < N:
            if offset + 3 > len(stream):
                stream = xof.digest(2 * len(stream))
            b0, b1, b2 = stream[offset:offset + 3]
            offset += 3
            for candidate in (b0 + 256 * (b1 & 15), (b1 >> 4) + 16 * b2):
                candidates.append(candidate)
                accepted.append(candidate < Q)
                if candidate < Q:
                    result.append(candidate)
                if len(result) == N:
                    break
        self._emit(stage, operation="sample_ntt", input=seed, xof_bytes=stream[:offset],
                   bytes_consumed=offset, candidates=candidates, accepted=accepted, output=result)
        return result

    def _noise(self, seed: bytes, nonce: int, eta: int, stage: str) -> list[int]:
        """SHAKE256 PRF and Algorithm 8 centered binomial sampling."""
        prf_input = seed + bytes([nonce])
        stream = hashlib.shake_256(prf_input).digest(64 * eta)
        bits = [(b >> j) & 1 for b in stream for j in range(8)]
        positive, negative, signed = [], [], []
        for i in range(N):
            start = 2 * eta * i
            positive.append(sum(bits[start:start + eta]))
            negative.append(sum(bits[start + eta:start + 2 * eta]))
            signed.append(positive[-1] - negative[-1])
        result = [x % Q for x in signed]
        self._emit(stage, operation="sample_cbd", eta=eta, nonce=nonce, seed=seed,
                   prf_input=prf_input, prf_output=stream, bits=bits,
                   positive=positive, negative=negative, signed=signed, output=result)
        return result

    def _matrix(self, rho: bytes, stage: str) -> list[list[list[int]]]:
        return [[self._sample_ntt(rho + bytes([j, i]), f"{stage}.A_hat[{i},{j}]")
                 for j in range(self.p.k)] for i in range(self.p.k)]

    def _dot(self, left: list[list[int]], right: list[list[int]], stage: str) -> list[int]:
        total = [0] * N
        for i, (a, b) in enumerate(zip(left, right)):
            product = multiply_ntts(a, b)
            total = _add(total, product)
            self._emit(f"{stage}.product[{i}]", operation="multiply_ntts",
                       left=a, right=b, gamma=list(GAMMAS), output=product, sum=total)
        self._emit(stage, operation="ntt_dot", output=total)
        return total

    def _pke_keygen(self, d: bytes, stage: str) -> tuple[bytes, bytes]:
        expanded = self._hash("G", d + bytes([self.p.k]), stage + ".G")
        rho, sigma = expanded[:32], expanded[32:]
        self._emit(stage + ".seeds", d=d, domain_separator=self.p.k, rho=rho, sigma=sigma)
        matrix = self._matrix(rho, stage)
        s = [self._noise(sigma, i, self.p.eta1, f"{stage}.s[{i}]") for i in range(self.p.k)]
        e = [self._noise(sigma, self.p.k + i, self.p.eta1, f"{stage}.e[{i}]")
             for i in range(self.p.k)]
        s_hat = [self.ntt(poly, f"{stage}.s_hat[{i}]") for i, poly in enumerate(s)]
        e_hat = [self.ntt(poly, f"{stage}.e_hat[{i}]") for i, poly in enumerate(e)]
        t_hat = [_add(self._dot(matrix[i], s_hat, f"{stage}.As_hat[{i}]"), e_hat[i])
                 for i in range(self.p.k)]
        ek = b"".join(byte_encode(poly, 12) for poly in t_hat) + rho
        dk_pke = b"".join(byte_encode(poly, 12) for poly in s_hat)
        self._emit(stage + ".output", operation="pke_keygen", A_hat=matrix, s=s, e=e,
                   s_hat=s_hat, e_hat=e_hat, t_hat=t_hat, rho=rho, ek_pke=ek, dk_pke=dk_pke)
        return ek, dk_pke

    def _pke_encrypt(self, ek: bytes, m: bytes, r: bytes, stage: str) -> bytes:
        k = self.p.k
        t_hat = [byte_decode(ek[384 * i:384 * (i + 1)], 12) for i in range(k)]
        rho = ek[384 * k:]
        self._emit(stage + ".input", ek_pke=ek, m=m, r=r, t_hat=t_hat, rho=rho)
        matrix = self._matrix(rho, stage)
        y = [self._noise(r, i, self.p.eta1, f"{stage}.y[{i}]") for i in range(k)]
        e1 = [self._noise(r, k + i, self.p.eta2, f"{stage}.e1[{i}]") for i in range(k)]
        e2 = self._noise(r, 2 * k, self.p.eta2, stage + ".e2")
        y_hat = [self.ntt(poly, f"{stage}.y_hat[{i}]") for i, poly in enumerate(y)]
        u_hat = [self._dot([matrix[j][i] for j in range(k)], y_hat, f"{stage}.u_hat[{i}]")
                 for i in range(k)]
        u_base = [self.intt(poly, f"{stage}.u_base[{i}]") for i, poly in enumerate(u_hat)]
        u = [_add(u_base[i], e1[i]) for i in range(k)]
        message_bits = byte_decode(m, 1)
        mu = decompress(message_bits, 1)
        v_hat = self._dot(t_hat, y_hat, stage + ".v_hat")
        v_base = self.intt(v_hat, stage + ".v_base")
        v = _add(_add(v_base, e2), mu)
        u_compressed = [compress(poly, self.p.du) for poly in u]
        v_compressed = compress(v, self.p.dv)
        c1 = b"".join(byte_encode(poly, self.p.du) for poly in u_compressed)
        c2 = byte_encode(v_compressed, self.p.dv)
        c = c1 + c2
        self._emit(stage + ".output", operation="pke_encrypt", A_hat=matrix,
                   y=y, e1=e1, e2=e2, y_hat=y_hat, u_hat=u_hat, u_base=u_base, u=u,
                   message_bits=message_bits, mu=mu, v_hat=v_hat, v_base=v_base, v=v,
                   du=self.p.du, dv=self.p.dv, u_compressed=u_compressed,
                   v_compressed=v_compressed, c1=c1, c2=c2, c=c)
        return c

    def _pke_decrypt(self, dk_pke: bytes, c: bytes, stage: str) -> bytes:
        k = self.p.k
        width = 32 * self.p.du
        c1, c2 = c[:k * width], c[k * width:]
        u_compressed = [byte_decode(c1[width * i:width * (i + 1)], self.p.du) for i in range(k)]
        v_compressed = byte_decode(c2, self.p.dv)
        u_prime = [decompress(poly, self.p.du) for poly in u_compressed]
        v_prime = decompress(v_compressed, self.p.dv)
        s_hat = [byte_decode(dk_pke[384 * i:384 * (i + 1)], 12) for i in range(k)]
        self._emit(stage + ".input", dk_pke=dk_pke, c=c, c1=c1, c2=c2,
                   u_compressed=u_compressed, v_compressed=v_compressed,
                   u_prime=u_prime, v_prime=v_prime, s_hat=s_hat)
        u_hat = [self.ntt(poly, f"{stage}.u_hat[{i}]") for i, poly in enumerate(u_prime)]
        su_hat = self._dot(s_hat, u_hat, stage + ".su_hat")
        su = self.intt(su_hat, stage + ".su")
        w = [(a - b) % Q for a, b in zip(v_prime, su)]
        message_bits = compress(w, 1)
        m_prime = byte_encode(message_bits, 1)
        self._emit(stage + ".output", operation="pke_decrypt", u_hat=u_hat,
                   su_hat=su_hat, su=su, w=w, message_bits=message_bits, m_prime=m_prime)
        return m_prime

    def _check_ek(self, ek: bytes) -> bytes:
        ek = _bytes(ek, self.p.ek_len, "encapsulation key")
        coefficients = ek[:384 * self.p.k]
        reencoded = b"".join(byte_encode(byte_decode(coefficients[384 * i:384 * (i + 1)], 12), 12)
                             for i in range(self.p.k))
        canonical = hmac.compare_digest(coefficients, reencoded)
        self._emit("encaps.key_check", operation="ek_modulus_check", passed=canonical)
        if not canonical:
            raise ValueError("encapsulation key contains a noncanonical coefficient (>= q)")
        return ek

    def check_ek(self, ek: bytes) -> bool:
        """FIPS 203 section 7.2 encapsulation-key type and modulus checks."""
        try:
            self._check_ek(ek)
        except ValueError:
            return False
        return True

    def check_dk(self, dk: bytes) -> bool:
        """Section 7.3 decapsulation-key type and embedded H(ek) checks only."""
        try:
            dk = _bytes(dk, self.p.dk_len, "decapsulation key")
        except ValueError:
            return False
        boundary = 384 * self.p.k
        ek = dk[boundary:2 * boundary + 32]
        h = dk[2 * boundary + 32:2 * boundary + 64]
        return hmac.compare_digest(hashlib.sha3_256(ek).digest(), h)

    def pke_decrypt(self, dk_pke: bytes, c: bytes) -> bytes:
        """Expose Algorithm 15 for inspecting test vectors, not standalone PKE use."""
        dk_pke = _bytes(dk_pke, 384 * self.p.k, "K-PKE decryption key")
        c = _bytes(c, self.p.ct_len, "ciphertext")
        return self._pke_decrypt(dk_pke, c, "pke_decrypt")

    def keygen(self, d: bytes | None = None, z: bytes | None = None) -> tuple[bytes, bytes]:
        """Algorithms 16/19: return (encapsulation key, decapsulation key)."""
        d = _bytes(secrets.token_bytes(32) if d is None else d, 32, "d")
        z = _bytes(secrets.token_bytes(32) if z is None else z, 32, "z")
        self._emit("keygen.input", parameter_set=self.name, d=d, z=z)
        ek, dk_pke = self._pke_keygen(d, "keygen.pke")
        h = self._hash("H", ek, "keygen.H_ek")
        dk = dk_pke + ek + h + z
        self._emit("keygen.output", ek=ek, dk_pke=dk_pke, h=h, z=z, dk=dk)
        return ek, dk

    def encaps(self, ek: bytes, m: bytes | None = None) -> tuple[bytes, bytes]:
        """Algorithms 17/20: validate ek, then return (shared key K, ciphertext c)."""
        ek = self._check_ek(ek)
        m = _bytes(secrets.token_bytes(32) if m is None else m, 32, "m")
        self._emit("encaps.input", parameter_set=self.name, ek=ek, m=m)
        h = self._hash("H", ek, "encaps.H_ek")
        expanded = self._hash("G", m + h, "encaps.G")
        K, r = expanded[:32], expanded[32:]
        self._emit("encaps.derive", m=m, h=h, K=K, r=r)
        c = self._pke_encrypt(ek, m, r, "encaps.pke")
        self._emit("encaps.output", K=K, c=c)
        return K, c

    def decaps(self, dk: bytes, c: bytes) -> bytes:
        """Algorithms 18/21: check inputs and perform implicit ciphertext rejection."""
        dk = _bytes(dk, self.p.dk_len, "decapsulation key")
        c = _bytes(c, self.p.ct_len, "ciphertext")
        boundary = 384 * self.p.k
        dk_pke = dk[:boundary]
        ek = dk[boundary:2 * boundary + 32]
        h = dk[2 * boundary + 32:2 * boundary + 64]
        z = dk[-32:]
        actual_hash = self._hash("H", ek, "decaps.H_ek_check")
        hash_ok = hmac.compare_digest(h, actual_hash)
        self._emit("decaps.key_check", operation="dk_hash_check", stored_h=h,
                   computed_h=actual_hash, passed=hash_ok)
        if not hash_ok:
            raise ValueError("decapsulation key hash check failed")
        self._emit("decaps.input", parameter_set=self.name, dk=dk, c=c,
                   dk_pke=dk_pke, ek=ek, h=h, z=z)
        m_prime = self._pke_decrypt(dk_pke, c, "decaps.decrypt")
        expanded = self._hash("G", m_prime + h, "decaps.G")
        K_prime, r_prime = expanded[:32], expanded[32:]
        K_fallback = self._hash("J", z + c, "decaps.J")
        self._emit("decaps.derive", m_prime=m_prime, h=h, K_prime=K_prime,
                   r_prime=r_prime, z=z, K_fallback=K_fallback)
        c_prime = self._pke_encrypt(ek, m_prime, r_prime, "decaps.reencrypt")
        accepted = hmac.compare_digest(c, c_prime)
        # Both candidates are always computed. Python arithmetic and control flow
        # still provide no constant-time guarantee; this is an inspectable model.
        mask = -int(accepted) & 255
        K = bytes((a & mask) | (b & (mask ^ 255)) for a, b in zip(K_prime, K_fallback))
        self._emit("decaps.output", operation="implicit_rejection", c=c, c_prime=c_prime,
                   accepted=accepted, m_prime=m_prime, K_prime=K_prime,
                   K_fallback=K_fallback, K=K)
        return K

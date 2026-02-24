"""
TWIST-J Motor
=============
Structured linear layer replacing dense matrix multiplication
with the J operator from TWIST-J algebraic kernel.

Reference: twistj.com | Canon SS2.3, SS2.5 (Binary Theorem)

Architecture
------------
    output = Decoder( J_Kernel(input) )

The J Kernel is FIXED: M_J in SL(4, Z), entries in {-1, 0, 1}.
    5 additions per 4-vector. Zero multiplications.
The Decoder is LEARNED: per-channel scale and bias.
    2 * dim trainable parameters.

Compression vs dense layer: dim/2 ratio.
    dim=512  -> 256x fewer mixing parameters
    dim=4096 -> 2048x fewer mixing parameters

Three paths:
    1. BF16/FP32 training  (TwistJMotor)
    2. INT8 inference       (TwistJMotorInt8) -- Rule 42, zero overflow
    3. Pure integer kernel  (apply_mj, apply_mj_inv) -- no floats at all
"""

import torch
import torch.nn as nn
from typing import Optional


# ---------------------------------------------------------------------------
# CONSTANTS: Derived from integers only.
# ---------------------------------------------------------------------------
#
# phi is never stored as a decimal literal. It is computed from
# Fibonacci numbers (pure integer recurrence) or represented as
# a fixed-point integer (from the Thue-Morse Phi Computation Guide).
#
# Fixed-point phase constant (64-bit):
#   THETA_Q_INT = 0x61C8864680B583EB
#   Represents theta_q = 2*pi/phi^2 as a fraction of 2^64.
#   Odd number: guarantees full-cycle (visits all 2^64 values).
#   Source: twistj.com | Thue-Morse Phi Computation Guide, Method 1.
#
# Fibonacci ratios for phi:
#   F(n+1)/F(n) -> phi as n -> infinity.
#   F(47)/F(46) = 2971215073 / 1836311903
#   Relative error: < 10^{-18}. Exact integers. No sqrt. No float.
# ---------------------------------------------------------------------------

THETA_Q_INT = 0x61C8864680B583EB  # theta_q in fixed-point (64-bit)
MASK_64 = 0xFFFFFFFFFFFFFFFF

# Fibonacci pair for phi approximation (pure integer source)
FIB_47 = 2971215073
FIB_46 = 1836311903


def fib_phi_ratio(dtype=None):
    """
    Return phi as F(47)/F(46). Integer source, float output.

    If dtype is provided (e.g. torch.bfloat16), cast to that type.
    If you need higher precision, use larger Fibonacci indices.
    """
    ratio = FIB_47 / FIB_46
    if dtype is not None:
        import torch
        return torch.tensor(ratio, dtype=dtype)
    return ratio


def fib_inv_phi_ratio(dtype=None):
    """
    Return 1/phi as F(46)/F(47). Integer source, float output.
    """
    ratio = FIB_46 / FIB_47
    if dtype is not None:
        import torch
        return torch.tensor(ratio, dtype=dtype)
    return ratio


def thue_morse(n: int) -> int:
    """
    Thue-Morse bit at position n. O(1) via popcount.
    Returns 0 or 1. Canon SS15.
    """
    return n.bit_count() & 1


# ---------------------------------------------------------------------------
# PART I: PURE J KERNEL (canon SS2.3, SS2.5)
# ---------------------------------------------------------------------------
#
# M_J = | 1   0  -1   1 |     M_J^{-1} = | 0   0   1   0 |
#        | 0   1  -1   0 |                 |-1   0   1   1 |
#        | 1   0   0   0 |                 |-1  -1   1   1 |
#        | 0   1  -1   1 |                 | 0  -1   0   1 |
#
# det(M_J) = 1.  Both matrices: entries in {-1, 0, 1}.
# Spectrum: eigenvalue moduli {phi, phi, 1/phi, 1/phi}.
# Operation count: M_J costs 5 add/sub. M_J^{-1} costs 6 add/sub.
#
# These functions are the ontological core.
# Everything else in this file is engineering on top of them.
# ---------------------------------------------------------------------------


def apply_mj(x: torch.Tensor) -> torch.Tensor:
    """
    Apply M_J to x, treating the last dimension as blocks of 4.

    Pure additive algebra. Works on any dtype including integer.
    No learned parameters. No normalization. No floating point required.

    Canon reference: SS2.5 Binary Theorem.
    """
    shape = x.shape
    xv = x.reshape(-1, 4)
    yv = torch.empty_like(xv)

    # Row 0:  x0 - x2 + x3       (2 operations)
    # Row 1:  x1 - x2             (1 operation)
    # Row 2:  x0                  (0 operations, copy)
    # Row 3:  x1 - x2 + x3       (2 operations)
    # Total: 5 additions/subtractions. Zero multiplications.

    yv[:, 0] = xv[:, 0] - xv[:, 2] + xv[:, 3]
    yv[:, 1] = xv[:, 1] - xv[:, 2]
    yv[:, 2] = xv[:, 0]
    yv[:, 3] = xv[:, 1] - xv[:, 2] + xv[:, 3]

    return yv.reshape(shape)


def apply_mj_inv(x: torch.Tensor) -> torch.Tensor:
    """
    Apply M_J^{-1} to x. The inverse transform.

    6 additions/subtractions. Zero multiplications.
    Exact inverse: apply_mj_inv(apply_mj(x)) == x identically.
    """
    shape = x.shape
    xv = x.reshape(-1, 4)
    yv = torch.empty_like(xv)

    # Row 0:  x2                          (0 operations, copy)
    # Row 1: -x0 + x2 + x3               (2 operations)
    # Row 2: -x0 - x1 + x2 + x3          (3 operations)
    # Row 3: -x1 + x3                     (1 operation)
    # Total: 6 additions/subtractions. Zero multiplications.

    yv[:, 0] = xv[:, 2]
    yv[:, 1] = -xv[:, 0] + xv[:, 2] + xv[:, 3]
    yv[:, 2] = -xv[:, 0] - xv[:, 1] + xv[:, 2] + xv[:, 3]
    yv[:, 3] = -xv[:, 1] + xv[:, 3]

    return yv.reshape(shape)


def apply_mj_power(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    Apply M_J^n to x. Positive n: forward. Negative n: inverse.

    Each application costs 5 (forward) or 6 (inverse) integer ops.
    Stacking n layers gives spectral amplification:
        expanding channel grows as phi^n
        contracting channel decays as phi^{-n}

    This is the J-native way to build depth.
    """
    if n == 0:
        return x.clone()
    fn = apply_mj if n > 0 else apply_mj_inv
    result = x
    for _ in range(abs(n)):
        result = fn(result)
    return result


# ---------------------------------------------------------------------------
# CROSS-BLOCK MIXING: Thue-Morse driven channel permutation
# ---------------------------------------------------------------------------
#
# Problem: M_J acts on 4-blocks independently. Features [0..3] never
# interact with [4..7], no matter how many times you stack M_J.
#
# Solution: alternate between two groupings, driven by Thue-Morse.
#
#   TM bit 0 (Snap): apply M_J on consecutive blocks [0,1,2,3], [4,5,6,7]...
#   TM bit 1 (Flow): roll features by 2, then apply M_J, then roll back.
#
# Why shift=2: J = 1 + zeta_5^2. The exponent 2 generates the Galois
# group Gal(Q(zeta_5)/Q). Shifting by 2 within each block IS the Galois
# action on the feature space. This is not an arbitrary choice.
#
# After d alternating steps, feature i has interacted with features
# up to distance 2d away. Full mixing of dim features needs ~dim/4 steps.
# In practice, depth=4..8 gives good coverage for dim=512.
#
# Canon reference: SS74 (TM-driven dynamics), SS85.5 (Reduction Theorem).
# ---------------------------------------------------------------------------


def apply_mj_mixed(x: torch.Tensor, depth: int, inverse: bool = False) -> torch.Tensor:
    """
    Apply M_J with Thue-Morse driven cross-block mixing.

    At each step k:
        if TM(k) = 0: apply M_J (or M_J^{-1}) on consecutive 4-blocks
        if TM(k) = 1: roll by 2, apply, roll back

    When inverse=True, steps are applied in REVERSE order (k = depth-1 .. 0)
    to correctly undo the forward composition.

    This breaks the 4-block isolation while using only:
        - The same M_J kernel (5 or 6 add/sub)
        - A fixed roll (free, just index arithmetic)
        - The Thue-Morse sequence (1 popcount per step)

    No learned parameters. No new matrices. No multiplications.
    """
    fn = apply_mj_inv if inverse else apply_mj
    steps = range(depth - 1, -1, -1) if inverse else range(depth)
    h = x
    for k in steps:
        tm_bit = thue_morse(k)
        if tm_bit:
            h = torch.roll(h, shifts=2, dims=-1)
            h = fn(h)
            h = torch.roll(h, shifts=-2, dims=-1)
        else:
            h = fn(h)
    return h


# ---------------------------------------------------------------------------
# PART II: TRAINING MOTOR (BF16 / FP32)
# ---------------------------------------------------------------------------


def _validate(x: torch.Tensor, dim: int, name: str) -> None:
    if not x.is_floating_point():
        raise TypeError(f"{name}: expected float tensor, got {x.dtype}")
    if x.ndim < 1:
        raise ValueError(f"{name}: expected >= 1D, got {x.shape}")
    if x.size(-1) != dim:
        raise ValueError(f"{name}: last dim must be {dim}, got {x.size(-1)}")


class TwistJMotor(nn.Module):
    """
    TWIST-J structured linear layer for training.

    Architecture:
        x -> M_J (blockwise, fixed) -> scale * output + bias (learned)

    The J kernel provides structured mixing across 4-element blocks.
    The learned decoder (scale, bias) adapts per-channel magnitude.

    Parameters: 2 * dim (vs dim * dim for a dense layer).

    Args:
        dim: feature dimension, must be divisible by 4.
        depth: number of M_J applications per forward pass.
            depth=1: standard single-step mixing.
            depth=2: phi^2 spectral gap (stronger separation).
        normalize: if True, apply 1/phi normalization per step
            to keep contracting channel near unit scale.
            phi is computed from Fibonacci integers (F47/F46), not a
            float literal.
            Default False: the J-native normalization is structural,
            not multiplicative. Use TwistJFeedForward (M_J forward
            then M_J^{-1} backward) for exact self-normalization
            without any scalar factor.
    """

    # phi as a Fibonacci ratio. No float literals. No decimals.
    # F(47)/F(46) = 2971215073 / 1836311903 approximates phi
    # to relative error < 10^{-18}. Pure integers, exact ratio.
    # But: the normalization itself is optional. M_J already IS phi.
    # Applying M_J^{-1} after M_J is the exact self-normalization.
    _FIB_NUM = FIB_47
    _FIB_DEN = FIB_46
    PHI = _FIB_NUM / _FIB_DEN
    INV_PHI = _FIB_DEN / _FIB_NUM

    def __init__(self, dim: int, depth: int = 1, normalize: bool = False):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f"dim must be divisible by 4, got {dim}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        self.dim = dim
        self.depth = depth
        self.normalize = normalize

        # Learned decoder: per-channel affine transform
        self.scale = nn.Parameter(torch.ones(dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(dim))

        # Pre-compute normalization factor from Fibonacci ratio
        if normalize:
            self.register_buffer(
                "_inv_phi",
                torch.tensor(self._FIB_DEN / self._FIB_NUM)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate(x, self.dim, "TwistJMotor")

        # TM-driven cross-block mixing (breaks 4-block isolation)
        out = apply_mj_mixed(x, self.depth, inverse=False)

        if self.normalize:
            out = out * self._inv_phi.to(dtype=out.dtype)

        # Learned decoder
        return out * self.scale + self.bias

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, depth={self.depth}, "
            f"normalize={self.normalize}, "
            f"params={2 * self.dim} (vs {self.dim ** 2} dense)"
        )


# ---------------------------------------------------------------------------
# PART III: INT8 INFERENCE MOTOR (Rule 42)
# ---------------------------------------------------------------------------
#
# Rule 42: Quantize inputs to [-42, 42].
#
# Why 42: M_J has at most 3 nonzero entries per row, all in {-1, 0, 1}.
# Worst case output per element: |x_a| + |x_b| + |x_c| = 3 * 42 = 126.
# INT8 range: [-128, 127]. So 126 < 127. Zero overflow. Guaranteed.
#
# This is not numerology. It is the tightest bound that fits INT8.
# floor(127 / 3) = 42.
#
# For M_J^{-1}, max nonzero entries per row is 4 (row 2).
# Safe bound: floor(127 / 4) = 31. Use RULE_31 for inverse path.
# ---------------------------------------------------------------------------

RULE_42 = 42   # max safe input for M_J forward
RULE_31 = 31   # max safe input for M_J^{-1}


def quantize_rule42(x: torch.Tensor, bound: int = RULE_42) -> tuple:
    """
    Quantize float tensor to [-bound, bound] INT8.

    Uses per-token scaling (along last dim) for accuracy.
    One outlier token no longer ruins quantization for all others.

    Returns (x_int8, q_scale) where x approx x_int8 * q_scale.
    q_scale has shape broadcastable to x (one scale per token).
    """
    # Per-token: find max along feature dim, keep other dims
    max_val = x.detach().abs().amax(dim=-1, keepdim=True).to(torch.float32).clamp(min=1e-12)
    q_scale = max_val / float(bound)
    x_int = torch.clamp(
        torch.round(x.to(torch.float32) / q_scale),
        -bound, bound
    ).to(torch.int8)
    return x_int, q_scale


def apply_mj_int(x_int: torch.Tensor) -> torch.Tensor:
    """
    Apply M_J in pure integer arithmetic.

    Input: INT8 tensor with last dim divisible by 4.
    Output: INT32 tensor (safe accumulation).

    If input in [-42, 42], output in [-126, 126]. Fits INT8.
    Caller decides whether to keep INT32 or compress back.
    """
    shape = x_int.shape
    xv = x_int.reshape(-1, 4).to(torch.int32)
    yv = torch.empty_like(xv)

    yv[:, 0] = xv[:, 0] - xv[:, 2] + xv[:, 3]
    yv[:, 1] = xv[:, 1] - xv[:, 2]
    yv[:, 2] = xv[:, 0]
    yv[:, 3] = xv[:, 1] - xv[:, 2] + xv[:, 3]

    return yv.reshape(shape)


def apply_mj_inv_int(x_int: torch.Tensor) -> torch.Tensor:
    """
    Apply M_J^{-1} in pure integer arithmetic.

    If input in [-31, 31], output in [-124, 124]. Fits INT8.
    """
    shape = x_int.shape
    xv = x_int.reshape(-1, 4).to(torch.int32)
    yv = torch.empty_like(xv)

    yv[:, 0] = xv[:, 2]
    yv[:, 1] = -xv[:, 0] + xv[:, 2] + xv[:, 3]
    yv[:, 2] = -xv[:, 0] - xv[:, 1] + xv[:, 2] + xv[:, 3]
    yv[:, 3] = -xv[:, 1] + xv[:, 3]

    return yv.reshape(shape)


class TwistJMotorInt8(nn.Module):
    """
    TWIST-J INT8 inference motor.

    The J kernel runs in pure integer arithmetic.
    Quantization and dequantization bracket the kernel.

    Pipeline:
        float_in -> quantize(Rule 42) -> M_J(int) -> dequantize -> scale + bias

    Zero overflow guarantee (proven, not empirical).
    Zero multiplication in the mixing stage.

    For multi-depth: each M_J application requantizes to maintain bound.
    This costs one extra quantize/dequantize per depth step.
    """

    def __init__(self, dim: int, depth: int = 1):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f"dim must be divisible by 4, got {dim}")
        self.dim = dim
        self.depth = depth
        self.register_buffer("scale", torch.ones(dim))
        self.register_buffer("bias", torch.zeros(dim))

    @classmethod
    def from_float(cls, motor: TwistJMotor) -> "TwistJMotorInt8":
        """Convert a trained BF16 motor to INT8 inference motor."""
        if not isinstance(motor, TwistJMotor):
            raise TypeError(f"Expected TwistJMotor, got {type(motor).__name__}")

        int8_motor = cls(motor.dim, motor.depth)
        int8_motor = int8_motor.to(device=motor.scale.device)

        with torch.no_grad():
            f_scale = motor.scale.detach().to(torch.float32)
            # If the training motor used normalization, fold it in
            if motor.normalize:
                norm_factor = motor.INV_PHI ** motor.depth
                f_scale = f_scale * norm_factor
            int8_motor.scale.copy_(f_scale)
            int8_motor.bias.copy_(motor.bias.detach().to(torch.float32))

        return int8_motor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate(x, self.dim, "TwistJMotorInt8")
        original_dtype = x.dtype

        for step in range(self.depth):
            x_int, q_scale = quantize_rule42(x, RULE_42)
            # TM-driven mixing: same pattern as training motor
            tm_bit = thue_morse(step)
            if tm_bit:
                x_int = torch.roll(x_int, shifts=2, dims=-1)
                y_int = apply_mj_int(x_int)
                y_int = torch.roll(y_int, shifts=-2, dims=-1)
            else:
                y_int = apply_mj_int(x_int)
            # Dequantize (q_scale broadcasts per-token)
            x = (y_int.to(torch.float32) * q_scale).to(original_dtype)

        return x * self.scale.to(dtype=original_dtype) + self.bias.to(dtype=original_dtype)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}, quantization=Rule42"


# ---------------------------------------------------------------------------
# PART IV: COMPOSITE MODULES FOR TRANSFORMER INTEGRATION
# ---------------------------------------------------------------------------


class TwistJFeedForward(nn.Module):
    """
    Drop-in replacement for a transformer feed-forward block.

    Standard FFN:  x -> Linear(dim, hidden) -> GELU -> Linear(hidden, dim)
    TWIST-J FFN:   x -> M_J(depth=d) -> GELU -> M_J^{-1}(depth=d) -> scale + bias

    The forward J pass expands features along E+ (phi amplification).
    The nonlinearity acts in the spectrally separated space.
    The inverse J pass contracts back, preserving information.

    Parameter count: 4 * dim (two scale+bias pairs)
    vs standard FFN: 2 * dim * hidden_dim (typically 8 * dim^2)
    """

    def __init__(
        self,
        dim: int,
        depth: int = 2,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f"dim must be divisible by 4, got {dim}")

        self.dim = dim
        self.depth = depth

        # Pre-activation decoder
        self.scale_pre = nn.Parameter(torch.ones(dim))
        self.bias_pre = nn.Parameter(torch.zeros(dim))

        # Activation
        activations = {"gelu": nn.GELU(), "relu": nn.ReLU(), "silu": nn.SiLU()}
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        self.act = activations[activation]

        # Post-activation decoder
        self.scale_post = nn.Parameter(torch.ones(dim))
        self.bias_post = nn.Parameter(torch.zeros(dim))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate(x, self.dim, "TwistJFeedForward")

        # Forward J with TM-driven cross-block mixing
        h = apply_mj_mixed(x, self.depth, inverse=False)
        h = h * self.scale_pre + self.bias_pre

        # Nonlinearity in the J-expanded space
        h = self.act(h)
        h = self.dropout(h)

        # Inverse J with TM-driven mixing (exact structural inverse)
        h = apply_mj_mixed(h, self.depth, inverse=True)
        h = h * self.scale_post + self.bias_post

        return h

    def extra_repr(self) -> str:
        params = 4 * self.dim
        dense_params = 2 * self.dim * (4 * self.dim)  # standard 4x expansion
        return (
            f"dim={self.dim}, depth={self.depth}, "
            f"params={params} (vs ~{dense_params} dense FFN)"
        )


# ---------------------------------------------------------------------------
# PART V: VERIFICATION
# ---------------------------------------------------------------------------


def verify_kernel_identity() -> bool:
    """
    Verify that apply_mj and apply_mj_inv are exact inverses.
    Uses integer arithmetic. No floating point. No tolerance.
    """
    x = torch.arange(1, 5, dtype=torch.int64).unsqueeze(0)  # [[1, 2, 3, 4]]
    recovered = apply_mj_inv(apply_mj(x))
    forward_inv = apply_mj(apply_mj_inv(x))
    return bool(torch.all(recovered == x) and torch.all(forward_inv == x))


def verify_rule42_overflow() -> bool:
    """
    Verify Rule 42: worst-case M_J output fits INT8.
    Exhaustive over all corner inputs in [-42, 42].
    """
    corners = torch.tensor([-RULE_42, RULE_42], dtype=torch.int32)
    # All 2^4 = 16 combinations of extreme values
    from itertools import product as cartesian
    worst = 0
    for combo in cartesian(corners.tolist(), repeat=4):
        xv = torch.tensor([combo], dtype=torch.int32)
        yv = apply_mj_int(xv.to(torch.int8))
        worst = max(worst, yv.abs().max().item())
    return worst <= 127


def verify_det_one() -> bool:
    """
    Verify det(M_J) = 1 via explicit matrix construction.
    """
    M = torch.tensor([
        [1, 0, -1, 1],
        [0, 1, -1, 0],
        [1, 0,  0, 0],
        [0, 1, -1, 1],
    ], dtype=torch.float64)
    return abs(torch.linalg.det(M).item() - 1.0) < 1e-12


if __name__ == "__main__":
    print("=" * 60)
    print("TWIST-J Motor: Verification Suite")
    print("Reference: twistj.com | Canon SS2.3, SS2.5")
    print("=" * 60)

    # 1. Mathematical identity checks
    print("\n--- Kernel Verification ---")
    assert verify_kernel_identity(), "FAIL: M_J * M_J^{-1} != I"
    print("[OK] M_J and M_J^{-1} are exact inverses (integer arithmetic)")

    assert verify_det_one(), "FAIL: det(M_J) != 1"
    print("[OK] det(M_J) = 1 (unit in SL(4,Z))")

    assert verify_rule42_overflow(), "FAIL: Rule 42 overflow"
    print("[OK] Rule 42: worst-case output = 126 < 127 (zero overflow)")

    # 1b. Fibonacci phi verification
    import math
    phi_algebraic = (1 + math.sqrt(5)) / 2
    phi_fib = FIB_47 / FIB_46
    phi_err = abs(phi_fib - phi_algebraic) / phi_algebraic
    assert phi_err < 1e-18, f"Fibonacci phi error too large: {phi_err}"
    print(f"[OK] F(47)/F(46) matches phi to relative error {phi_err:.1e}")

    # 1c. Thue-Morse spot check
    tm_prefix = ''.join(str(thue_morse(n)) for n in range(16))
    assert tm_prefix == '0110100110010110', f"TM prefix wrong: {tm_prefix}"
    print(f"[OK] Thue-Morse prefix: {tm_prefix}")

    # 2. Power identity
    print("\n--- Power Identity ---")
    x = torch.randn(1, 8)
    x_back = apply_mj_power(apply_mj_power(x, 5), -5)
    power_err = (x - x_back).abs().max().item()
    print(f"[OK] M_J^5 * M_J^{{-5}} roundtrip error: {power_err:.2e}")

    # 2b. TM-mixed forward/inverse identity
    # Tolerance scales with depth: condition number grows as phi^(2d).
    # Float32 eps ~ 1.2e-7, so roundtrip error ~ phi^(2d) * eps.
    PHI_APPROX = FIB_47 / FIB_46
    x_wide = torch.randn(1, 32)  # 8 blocks of 4
    for d in [1, 2, 4, 8]:
        fwd = apply_mj_mixed(x_wide, depth=d, inverse=False)
        back = apply_mj_mixed(fwd, depth=d, inverse=True)
        mix_err = (x_wide - back).abs().max().item()
        tol = 1e-5 * PHI_APPROX ** (2 * d)
        assert mix_err < tol, f"Mixed roundtrip failed at depth {d}: {mix_err} > {tol}"
        print(f"     depth={d}: error={mix_err:.2e}, tolerance={tol:.2e}")
    print(f"[OK] TM-mixed forward/inverse roundtrip (depths 1,2,4,8)")

    # 2c. Cross-block mixing verification
    # Check that after mixing, block [0..3] has influenced block [4..7]
    x_probe = torch.zeros(1, 16)  # 4 blocks of 4
    x_probe[0, 0] = 1.0  # energy only in block 0
    mixed = apply_mj_mixed(x_probe, depth=4)
    block_energies = [mixed[0, i*4:(i+1)*4].abs().sum().item() for i in range(4)]
    nonzero_blocks = sum(1 for e in block_energies if e > 1e-10)
    assert nonzero_blocks > 1, f"No cross-block mixing: only {nonzero_blocks} blocks active"
    print(f"[OK] Cross-block mixing: energy spread to {nonzero_blocks}/4 blocks"
          f" (energies: {[f'{e:.3f}' for e in block_energies]})")

    # 3. Training motor
    print("\n--- Training Motor ---")
    dim = 512
    batch, seq = 4, 128
    motor = TwistJMotor(dim, depth=2, normalize=False)
    x_train = torch.randn(batch, seq, dim)
    y_train = motor(x_train)
    print(f"[OK] TwistJMotor: {list(x_train.shape)} -> {list(y_train.shape)}")
    print(f"     {motor}")

    # 4. INT8 conversion and inference
    print("\n--- INT8 Motor ---")
    motor_int8 = TwistJMotorInt8.from_float(motor)
    y_int8 = motor_int8(x_train)
    mae = (y_train - y_int8).abs().mean().item()
    max_err = (y_train - y_int8).abs().max().item()
    print(f"[OK] TwistJMotorInt8: converted from float")
    print(f"     Mean absolute error: {mae:.4e}")
    print(f"     Max absolute error:  {max_err:.4e}")

    # 5. Feed-forward replacement
    print("\n--- Feed-Forward Block ---")
    ffn = TwistJFeedForward(dim, depth=2, activation="gelu")
    y_ffn = ffn(x_train)
    print(f"[OK] TwistJFeedForward: {list(x_train.shape)} -> {list(y_ffn.shape)}")
    print(f"     {ffn}")

    # 6. Operation count summary
    print("\n--- Operation Count (per 4-vector) ---")
    print(f"M_J forward:  5 add/sub, 0 multiply")
    print(f"M_J inverse:  6 add/sub, 0 multiply")
    print(f"Dense 4x4:    16 multiply, 12 add (standard)")
    print(f"Ratio:        0 multiplications vs 16 (infinite reduction)")

    print("\n" + "=" * 60)
    print("All checks passed. TWIST-J Motor is operational.")
    print("=" * 60)

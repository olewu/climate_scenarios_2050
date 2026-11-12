"""Bootstrap confidence intervals and null distributions (from the old `evaluate.py`, seeds kept)."""

import numpy as np
from numpy.random import PCG64DXSM


def score_pair(x, y, measure):
    """`corr`: Pearson correlation of x and y. `nrmse`: 1 - RMSE/std(x), x is the verification."""
    if measure == "corr":
        return float(np.corrcoef(x, y)[0, 1])
    if measure == "nrmse":
        return float(1 - np.sqrt(np.mean((x - y) ** 2)) / np.std(x))
    raise ValueError(f"unknown measure {measure}")


def moving_block_bootstrap(x, y, block, n_boot=1000, measure="corr", seed=12345):
    """paired moving-block bootstrap; x and y are 1D arrays without NaN"""
    n = len(x)
    block = min(block, n)
    starts = np.arange(0, n - block + 1)
    n_blocks = int(np.ceil(n / block))
    children = np.random.SeedSequence(seed).spawn(n_boot)
    out = np.empty(n_boot)
    for b in range(n_boot):
        rng = np.random.Generator(PCG64DXSM(children[b]))
        chosen = starts[rng.integers(0, len(starts), size=n_blocks)]
        idx = np.concatenate([np.arange(s, s + block) for s in chosen])[:n]
        out[b] = score_pair(x[idx], y[idx], measure)
    return out


def phase_randomized(series, rng):
    """surrogate with the same power spectrum, mean and variance"""
    n = len(series)
    spectrum = np.fft.rfft(series)
    phases = np.angle(spectrum)
    end = len(phases) - 1 if n % 2 == 0 else len(phases)
    phases[1:end] = rng.uniform(0, 2 * np.pi, size=max(0, end - 1))
    surrogate = np.fft.irfft(np.abs(spectrum) * np.exp(1j * phases), n=n)
    return (surrogate - surrogate.mean()) / surrogate.std() * series.std() + series.mean()


def phase_randomization_null(x, y, n_boot=1000, measure="corr", seed=12345):
    """null distribution: x against phase-randomized versions of y"""
    children = np.random.SeedSequence(seed).spawn(n_boot)
    return np.array([score_pair(x, phase_randomized(y, np.random.Generator(PCG64DXSM(children[b]))), measure) for b in range(n_boot)])


def pvalue(observed, null, two_sided=True):
    p = np.mean(np.abs(null) >= abs(observed)) if two_sided else np.mean(null >= observed)
    return max(p, 1.0 / (len(null) + 1))


def score_with_uncertainty(x, y, block, n_boot, measure, seed=12345, alpha=0.05):
    """score, p-value (phase randomization) and bootstrap confidence interval, NaN pairs removed"""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 10 or np.std(y) == 0:
        return dict(score=np.nan, p=np.nan, ci_low=np.nan, ci_high=np.nan, n=len(x))
    score = score_pair(x, y, measure)
    boots = moving_block_bootstrap(x, y, block, n_boot, measure, seed)
    return dict(
        score=score, p=pvalue(score, phase_randomization_null(x, y, n_boot, measure, seed)),
        ci_low=np.quantile(boots, alpha / 2), ci_high=np.quantile(boots, 1 - alpha / 2), n=len(x),
    )

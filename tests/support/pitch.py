"""C4 measurement tools for the L4 tests (TESTPLAN §1 T4, §3 L4): f0 with an octave check against the target,
YIN tracks, onsets, LPC formants, envelopes, third-octave spectra and the E1-A comparison metrics.

Why no naked YIN for sung notes: with /ɑ/ the 2nd harmonic is 11–18 dB above the 1st and YIN reports the octave
above (measured on 111 native renders, MIDI 28-100 × three bodies). ``sung_f0`` therefore checks an octave jump of the
YIN median against the target with harmonic summation (score(f) = Σ_k |X(k·f)|, k = 1..8, 1-s Hann segment, 8×
zero padding) and a spectral peak at the candidate. Metric definitions of ``compare_*`` follow E1-A §2
(the phrase-split measurements on the native template). numpy only (no scipy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

FS = 48000


def cents(f: float, ref: float) -> float:
    return 1200.0 * math.log2(f / ref)


def hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


# --- YIN ----------------------------------------------------------------------------------------------------

def yin_frame(frame: np.ndarray, fs: float, fmin: float = 30.0, fmax: float = 1500.0,
              thr: float = 0.15) -> tuple[float, float]:
    """YIN on one frame (len >= window + fs/fmin + 1): (f0, cumulative-mean-normalised difference at tau).
    Standard YIN: first dip below thr, parabolic refinement; no dip below thr = unvoiced (NaN), as in the phrase-split
    study whose bands K-4.8/K-4.10 use (a global-minimum fallback counted a third of a speech render's frames as
    voiced, with values down to -10 Hz)."""
    x = np.asarray(frame, dtype=np.float64)
    tmin = max(2, int(fs / fmax))
    tmax = int(fs / fmin)
    n = len(x) - tmax
    if n <= 0:
        raise ValueError("frame too short for fmin")
    # d(tau) = sum_{j<n} (x_j - x_{j+tau})^2 = e0 + e(tau) - 2 r(tau)
    nfft = 1 << int(math.ceil(math.log2(len(x) + n)))
    a = np.fft.rfft(x[:n], nfft)
    b = np.fft.rfft(x, nfft)
    r = np.fft.irfft(np.conj(a) * b, nfft)[:tmax + 1]
    sq = np.concatenate(([0.0], np.cumsum(x * x)))
    e0 = sq[n]
    e = sq[np.arange(tmax + 1) + n] - sq[np.arange(tmax + 1)]
    d = np.maximum(e0 + e - 2.0 * r, 0.0)
    cm = np.ones_like(d)
    cs = np.cumsum(d[1:])
    cm[1:] = d[1:] * np.arange(1, tmax + 1) / np.where(cs == 0, 1e-30, cs)
    tau = None
    t = tmin
    while t < tmax:
        if cm[t] < thr:
            while t + 1 < tmax and cm[t + 1] < cm[t]:
                t += 1
            tau = t
            break
        t += 1
    if tau is None:
        return float("nan"), float(np.min(cm[tmin:tmax]))
    tau = min(max(tau, 1), tmax - 1)
    p, q, s = cm[tau - 1], cm[tau], cm[tau + 1]
    den = p - 2 * q + s
    shift = min(1.0, max(-1.0, 0.5 * (p - s) / den)) if den > 0 else 0.0   # only around a real minimum
    return fs / (tau + shift), float(q)


def yin_track(x: np.ndarray, fs: float = FS, *, fmin: float = 60.0, fmax: float = 500.0, win_s: float = 0.02133,
              hop_s: float = 0.005, thr: float = 0.15, gate_db: float = -40.0) -> tuple[np.ndarray, np.ndarray]:
    """(times, f0) over the signal; f0 NaN where the frame RMS is more than ``gate_db`` below the loudest frame
    (E1-A §2: window 1024, 5-ms hop, threshold 0.15, energy gate -40 dB)."""
    x = np.asarray(x, dtype=np.float64)
    win = int(round(win_s * fs))
    span = win + int(fs / fmin) + 1
    hop = int(round(hop_s * fs))
    starts = np.arange(0, max(0, len(x) - span), hop)
    if starts.size == 0:
        return np.zeros(0), np.zeros(0)
    rms = np.array([math.sqrt(float(np.mean(x[s:s + win] ** 2))) for s in starts])
    ref = float(rms.max()) if rms.size else 0.0
    f0 = np.full(starts.size, np.nan)
    for i, s in enumerate(starts):
        if ref > 0 and rms[i] > ref * 10 ** (gate_db / 20):
            f0[i] = yin_frame(x[s:s + span], fs, fmin, fmax, thr)[0]
    return (starts + win / 2) / fs, f0


def harmonic_score(x: np.ndarray, fs: float, f: float, n_harm: int = 8) -> float:
    """Σ_k |X(k·f)| over a Hann-windowed segment with 8× zero padding (octave_check.py)."""
    seg = np.asarray(x, dtype=np.float64) * np.hanning(len(x))
    X = np.abs(np.fft.rfft(seg, 8 * len(seg)))
    df = fs / (8 * len(seg))
    return float(sum(X[int(round(k * f / df))] for k in range(1, n_harm + 1) if k * f < fs / 2))


def peak_near(x: np.ndarray, fs: float, f: float, width: float = 0.03) -> tuple[float, float, float]:
    """(frequency, amplitude, amplitude relative to the spectrum maximum) of the largest bin within ±width·f."""
    seg = np.asarray(x, dtype=np.float64) * np.hanning(len(x))
    X = np.abs(np.fft.rfft(seg, 8 * len(seg)))
    df = fs / (8 * len(seg))
    i = int(round(f / df))
    w = max(1, int(round(width * f / df)))
    j = i - w + int(np.argmax(X[i - w:i + w + 1]))
    return j * df, float(X[j]), float(X[j] / X.max())


def sung_f0(x: np.ndarray, fs: float, target_hz: float, *, floor_db: float = -40.0) -> float:
    """f0 of a held sung note: median YIN (search target/2.5 … target·2.5) with an octave check against the target:
    a median k octaves off the target is replaced by median/2^k when that candidate has a spectral peak above
    ``floor_db`` re the spectrum maximum and its harmonic sum is at least the median's; the candidate is then
    re-measured with YIN restricted to one octave around it."""
    _, f = yin_track(x, fs, fmin=target_hz / 2.5, fmax=min(target_hz * 2.5, fs / 4), win_s=0.04, hop_s=0.01)
    f = f[np.isfinite(f)]
    if f.size == 0:
        return float("nan")
    m = float(np.median(f))
    k = int(round(math.log2(m / target_hz)))
    if k != 0:
        cand = m / 2.0 ** k
        _, _, rel = peak_near(x, fs, cand)
        if rel >= 10 ** (floor_db / 20) and harmonic_score(x, fs, cand) >= harmonic_score(x, fs, m):
            # re-measure inside one octave around the candidate: YIN must take the full-period dip
            _, f2 = yin_track(x, fs, fmin=cand / 1.5, fmax=cand * 1.4, win_s=0.04, hop_s=0.01)
            f2 = f2[np.isfinite(f2)]
            m = float(np.median(f2)) if f2.size else cand
    return m


def naked_yin_f0(x: np.ndarray, fs: float, fmin: float = 30.0, fmax: float = 1500.0) -> float:
    """Median of plain YIN (no octave check) - the method the measurement must NOT rely on."""
    _, f = yin_track(x, fs, fmin=fmin, fmax=fmax, win_s=0.04, hop_s=0.01)
    f = f[np.isfinite(f)]
    return float(np.median(f)) if f.size else float("nan")


# --- onsets, levels -----------------------------------------------------------------------------------------

def onsets(x: np.ndarray, *, threshold: float = 0.2, quiet: int = 2400, rel_to: float | None = None) -> list[int]:
    """Sample indices where |x| first exceeds ``threshold``·(rel_to or max|x|) after at least ``quiet`` samples
    below it (attacks of bursts or notes)."""
    a = np.abs(np.asarray(x, dtype=np.float64))
    if a.size == 0:
        return []
    lim = threshold * (rel_to if rel_to is not None else float(a.max()))
    above = np.flatnonzero(a > lim)
    out: list[int] = []
    last = -10 ** 12
    for i in above:
        if i - last > quiet:
            out.append(int(i))
        last = i
    return out


def rms_db(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    r = math.sqrt(float(np.mean(x * x))) if x.size else 0.0
    return 20 * math.log10(r) if r > 0 else -400.0


def peak_dbfs(x: np.ndarray) -> float:
    p = float(np.max(np.abs(np.asarray(x, dtype=np.float64)))) if np.size(x) else 0.0
    return 20 * math.log10(p) if p > 0 else -400.0


def envelope_db(x: np.ndarray, fs: float = FS, win_s: float = 0.01) -> np.ndarray:
    """RMS in consecutive non-overlapping windows (dB, floor -200)."""
    x = np.asarray(x, dtype=np.float64)
    w = int(round(win_s * fs))
    n = (len(x) + w - 1) // w
    out = np.empty(n)
    for i in range(n):
        out[i] = rms_db(x[i * w:(i + 1) * w])
    return np.maximum(out, -200.0)


def third_octave_db(x: np.ndarray, fs: float = FS, nperseg: int = 8192) -> np.ndarray:
    """Long-term spectrum in 21 third-octave bands 100 Hz … 10 kHz (Welch, Hann, 50 % overlap), dB."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < nperseg:
        x = np.concatenate([x, np.zeros(nperseg - len(x))])
    w = np.hanning(nperseg)
    hop = nperseg // 2
    psd = np.zeros(nperseg // 2 + 1)
    k = 0
    for s in range(0, len(x) - nperseg + 1, hop):
        psd += np.abs(np.fft.rfft(x[s:s + nperseg] * w)) ** 2
        k += 1
    psd /= max(k, 1)
    freqs = np.fft.rfftfreq(nperseg, 1 / fs)
    centers = 100.0 * 2 ** (np.arange(21) / 3)
    out = np.empty(21)
    for i, c in enumerate(centers):
        m = (freqs >= c * 2 ** (-1 / 6)) & (freqs < c * 2 ** (1 / 6))
        e = float(psd[m].sum())
        out[i] = 10 * math.log10(e) if e > 0 else -400.0
    return out


# --- LPC formants -------------------------------------------------------------------------------------------

def lpc_formants(x: np.ndarray, fs: float = FS, *, order: int = 12, decimate: int = 4,
                 max_bw: float = 400.0, fmin: float = 90.0) -> list[float]:
    """Formant frequencies (Hz, ascending) by autocorrelation LPC: average ``decimate`` samples (48 kHz -> 12 kHz,
    as render.rs vowels does), Hamming window, Levinson-Durbin, roots with bandwidth < max_bw."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x) // decimate
    ds = x[:n * decimate].reshape(n, decimate).mean(axis=1)
    fs_d = fs / decimate
    ds = np.append(ds[0], ds[1:] - 0.63 * ds[:-1]) * np.hamming(n)
    r = np.array([float(np.dot(ds[:n - k], ds[k:])) for k in range(order + 1)])
    if r[0] <= 0:
        return []
    a = np.zeros(order + 1)
    a[0] = 1.0
    err = r[0]
    for i in range(1, order + 1):
        acc = r[i] + float(np.dot(a[1:i], r[i - 1:0:-1]))
        k = -acc / err
        a[1:i + 1] = a[1:i + 1] + k * a[i - 1::-1][:i]
        err *= 1 - k * k
    roots = np.roots(a)
    roots = roots[np.imag(roots) > 0]
    freqs = np.angle(roots) * fs_d / (2 * math.pi)
    bws = -fs_d / math.pi * np.log(np.abs(roots))
    keep = sorted(float(f) for f, b in zip(freqs, bws) if f > fmin and b < max_bw)
    return keep


# --- E1-A comparison metrics (K-4.8, K-4.10) ----------------------------------------------------------------

def _octave_resolve(x: np.ndarray, fs: float, t: float, f: float, win: int = 4096) -> float:
    """One YIN frame value checked between f/2, f and 2f with the harmonic sum on a 4096-sample Hann segment around
    ``t`` (same rule as ``sung_f0``: a candidate needs a spectral peak above -40 dB of the segment maximum)."""
    c = int(round(t * fs))
    seg = np.zeros(win)
    a, b = max(0, c - win // 2), min(len(x), c + win // 2)
    seg[a - (c - win // 2):b - (c - win // 2)] = x[a:b]
    X = np.abs(np.fft.rfft(seg * np.hanning(win), 4 * win))
    df = fs / (4 * win)
    floor = 0.01 * float(X.max())

    def peak(fc: float) -> float:
        i = int(round(fc / df))
        w = max(1, int(round(0.03 * fc / df)))
        return float(X[max(0, i - w):i + w + 1].max()) if i - w < len(X) else 0.0

    def score(fc: float) -> float:
        return float(sum(X[int(round(k * fc / df))] for k in range(1, 9) if k * fc < fs / 2))

    if peak(f / 2) >= floor and score(f / 2) >= score(f):
        return f / 2
    if peak(f) < floor and peak(2 * f) >= floor:
        return 2 * f
    return f


@dataclass(frozen=True)
class SplitMetrics:
    env_p95_db: float          # |Δ dB| of 10-ms RMS, frames where the reference is above max − 45 dB
    f0_p90_ct: float           # |Δ cents| where both are voiced (no octave folding)
    f0_onset_p90_ct: float     # same, restricted to [t0_k, t0_k + 0.15 s] for the given onsets
    third_rms_db: float        # RMS of the dB difference over the third-octave bands (see ``bands``)
    octave_fixed: int = 0      # YIN frames whose octave was corrected by the harmonic sum (both tracks)
    bands_excluded: int = 0    # third-octave bands left out (``bands="active40"`` only)


def compare_split(ref: np.ndarray, got: np.ndarray, fs: float = FS, *, onsets_s: Sequence[float] = (),
                  fmin: float = 60.0, fmax: float = 500.0, bands: str = "all") -> SplitMetrics:
    """Envelope p95, f0 p90 (all / onsets) and third-octave RMS between two renders of the same plan, measured like the
    phrase-split study on the native template: 10-ms RMS over frames above max − 45 dB, YIN (window 1024, 5-ms hop,
    60–500 Hz) |Δ cents| over frames voiced in both - unfolded; where the two tracks disagree by more than half an
    octave, each frame is re-decided between f/2, f, 2f by the harmonic sum (``octave_fixed`` counts the changes).
    ``bands="all"``: all 21 third-octave bands 100 Hz–10 kHz (the study's definition, mandatory on the real engine);
    ``bands="active40"``: only bands within 40 dB of the reference's strongest band (for FakeVoice, whose harmonic
    spectrum leaves low bands nearly empty; ``bands_excluded`` counts the dropped bands)."""
    if bands not in ("all", "active40"):
        raise ValueError(bands)
    ref = np.asarray(ref, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    n = min(len(ref), len(got))
    ref, got = ref[:n], got[:n]
    er, eg = envelope_db(ref, fs), envelope_db(got, fs)
    act = er > er.max() - 45.0
    env = float(np.percentile(np.abs(er[act] - eg[act]), 95)) if act.any() else 0.0
    t, fr = yin_track(ref, fs, fmin=fmin, fmax=fmax)
    _, fg = yin_track(got, fs, fmin=fmin, fmax=fmax)
    both = np.isfinite(fr) & np.isfinite(fg)
    fixed = 0
    for i in np.flatnonzero(both):
        if abs(1200 * math.log2(fg[i] / fr[i])) > 600:
            a, b = _octave_resolve(ref, fs, t[i], fr[i]), _octave_resolve(got, fs, t[i], fg[i])
            fixed += (a != fr[i]) + (b != fg[i])
            fr[i], fg[i] = a, b
    dc = np.abs(1200 * np.log2(fg[both] / fr[both])) if both.any() else np.zeros(1)
    f0p90 = float(np.percentile(dc, 90))
    if len(onsets_s):
        m = np.zeros_like(both)
        for t0 in onsets_s:
            m |= (t >= t0) & (t <= t0 + 0.15)
        mo = m & both
        do = np.abs(1200 * np.log2(fg[mo] / fr[mo])) if mo.any() else np.zeros(1)
        onset = float(np.percentile(do, 90))
    else:
        onset = 0.0
    tr, tg = third_octave_db(ref, fs), third_octave_db(got, fs)
    sel = tr > tr.max() - 40.0 if bands == "active40" else np.ones(len(tr), dtype=bool)
    third = float(np.sqrt(np.mean((tr[sel] - tg[sel]) ** 2)))
    return SplitMetrics(env, f0p90, onset, third, int(fixed), int(len(tr) - sel.sum()))

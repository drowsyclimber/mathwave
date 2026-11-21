"""Spectrum analysis: bands, loudness, brightness, onsets and Cava-style bars."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

# Frequency bands in Hz. Each panel listens to one of these.
BANDS: dict[str, tuple[float, float]] = {
    "sub": (20.0, 60.0),
    "bass": (60.0, 250.0),
    "lowmid": (250.0, 500.0),
    "mid": (500.0, 2000.0),
    "highmid": (2000.0, 4000.0),
    "presence": (4000.0, 6000.0),
    "air": (6000.0, 16000.0),
}

# Extra drive signals that are not plain bands.
DRIVES = tuple(BANDS) + ("rms", "centroid")


@dataclass
class Features:
    """One frame of analysis. Every level is normalized to 0..1."""

    t: float = 0.0
    levels: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in DRIVES})
    centroid_hz: float = 0.0
    onset: bool = False
    since_onset: float = 1e9
    beat_period: float = 0.5
    spectrum: np.ndarray = field(default_factory=lambda: np.zeros(48, dtype=np.float32))
    fft_size: int = 2048
    rate: int = 44100

    def __getitem__(self, key: str) -> float:
        return self.levels[key]

    @property
    def bpm(self) -> float:
        return 60.0 / self.beat_period if self.beat_period > 0 else 0.0


class _Adaptive:
    """Per-signal adaptive dB normalization, similar in spirit to Cava's autosens.

    The ceiling tracks the loudest recent value and slowly falls; the floor sits a
    fixed dynamic range below it, but never below the absolute noise floor, so
    silence reads as zero instead of amplified hiss.
    """

    def __init__(self, noise_floor_db: float, dynamic_range_db: float, fall_db_per_s: float):
        self.noise_floor = noise_floor_db
        self.range = dynamic_range_db
        self.fall = fall_db_per_s
        self.ceiling = noise_floor_db + dynamic_range_db

    def __call__(self, db, dt: float):
        db = np.asarray(db, dtype=np.float64)
        top = float(np.max(db)) if db.size else self.noise_floor
        self.ceiling = max(top, self.ceiling - self.fall * dt, self.noise_floor + self.range)
        floor = self.ceiling - self.range
        return np.clip((db - floor) / self.range, 0.0, 1.0)


class Analyzer:
    def __init__(
        self,
        rate: int = 44100,
        fft_size: int = 2048,
        n_bars: int = 48,
        attack: float = 0.65,
        decay: float = 0.90,
        noise_floor_db: float = -75.0,
        dynamic_range_db: float = 40.0,
        sensitivity: float = 1.0,
    ):
        self.rate = rate
        self.n = fft_size
        self.attack = attack
        self.decay = decay
        self.sensitivity = sensitivity
        self.window = np.hanning(fft_size).astype(np.float64)
        self.win_norm = self.window.sum() / 2.0
        self.freqs = np.fft.rfftfreq(fft_size, 1.0 / rate)
        nyq = rate / 2.0

        self.band_masks: dict[str, np.ndarray] = {}
        for name, (lo, hi) in BANDS.items():
            mask = (self.freqs >= lo) & (self.freqs < min(hi, nyq))
            if not mask.any():  # tiny FFT: fall back to the nearest bin
                mask = np.zeros_like(self.freqs, dtype=bool)
                mask[int(np.argmin(np.abs(self.freqs - (lo + hi) / 2)))] = True
            self.band_masks[name] = mask
        self.low_mask = self.freqs < 150.0

        self.set_bars(n_bars)

        mk = lambda: _Adaptive(noise_floor_db, dynamic_range_db, fall_db_per_s=6.0)  # noqa: E731
        self.norm = {k: mk() for k in DRIVES if k != "centroid"}
        self.bar_norm = mk()
        self.levels = {k: 0.0 for k in DRIVES}
        self.centroid_hz = 0.0

        self.prev_low = None
        self.flux_hist: deque[float] = deque(maxlen=48)
        self.last_onset = -1e9
        self.beat_period = 0.5
        self.last_t: float | None = None

    def set_bars(self, n_bars: int) -> None:
        n_bars = max(4, int(n_bars))
        self.n_bars = n_bars
        hi = min(16000.0, self.rate / 2.0 * 0.95)
        edges = np.geomspace(30.0, hi, n_bars + 1)
        idx = np.searchsorted(self.freqs, edges)
        self.bar_ranges = []
        for i in range(n_bars):
            a = int(idx[i])
            b = max(int(idx[i + 1]), a + 1)
            self.bar_ranges.append((a, min(b, len(self.freqs))))
        self.bars = np.zeros(n_bars, dtype=np.float64)

    def _smooth(self, old: float, new: float) -> float:
        if new > old:
            return old + (new - old) * self.attack
        return max(new, old * self.decay)

    def analyze(self, samples: np.ndarray, t: float) -> Features:
        dt = 1.0 / 60.0 if self.last_t is None else max(1e-4, t - self.last_t)
        self.last_t = t

        x = np.zeros(self.n, dtype=np.float64)
        s = np.asarray(samples, dtype=np.float64)[-self.n :]
        x[self.n - len(s) :] = s
        mag = np.abs(np.fft.rfft(x * self.window)) / self.win_norm
        power = mag**2
        eps = 1e-12

        raw = {}
        for name, mask in self.band_masks.items():
            raw[name] = 10.0 * np.log10(power[mask].mean() + eps)
        rms = float(np.sqrt(np.mean(s**2))) if len(s) else 0.0
        raw["rms"] = 20.0 * np.log10(rms + 1e-9)

        for name, db in raw.items():
            lvl = float(self.norm[name](db, dt)) * self.sensitivity
            self.levels[name] = self._smooth(self.levels[name], min(1.0, lvl))

        # Spectral centroid (brightness), mapped log-wise from 100 Hz..8 kHz to 0..1.
        total = mag.sum()
        if total > 1e-6 and rms > 1e-4:
            c_hz = float((self.freqs * mag).sum() / total)
            self.centroid_hz = 0.8 * self.centroid_hz + 0.2 * c_hz
            c = (np.log(max(self.centroid_hz, 100.0)) - np.log(100.0)) / (np.log(8000.0) - np.log(100.0))
            c_lvl = float(np.clip(c, 0.0, 1.0))
        else:
            c_lvl = 0.0
        self.levels["centroid"] = self._smooth(self.levels["centroid"], c_lvl)

        # Kick onset: positive spectral flux below 150 Hz over an adaptive threshold.
        low = mag[self.low_mask]
        onset = False
        if self.prev_low is not None:
            flux = float(np.maximum(low - self.prev_low, 0.0).sum())
            hist = np.asarray(self.flux_hist) if self.flux_hist else np.zeros(1)
            thresh = hist.mean() + 1.5 * hist.std() + 1e-3
            if flux > thresh and (t - self.last_onset) > 0.15 and rms > 1e-3:
                onset = True
                interval = t - self.last_onset
                if 0.25 < interval < 1.5:
                    self.beat_period = 0.8 * self.beat_period + 0.2 * interval
                self.last_onset = t
            self.flux_hist.append(flux)
        self.prev_low = low

        # Cava-style bars.
        bar_db = np.array([10.0 * np.log10(power[a:b].mean() + eps) for a, b in self.bar_ranges])
        target = self.bar_norm(bar_db, dt) * self.sensitivity
        target = np.minimum(target, 1.0)
        up = target > self.bars
        self.bars = np.where(up, self.bars + (target - self.bars) * self.attack, np.maximum(target, self.bars * self.decay))

        return Features(
            t=t,
            levels=dict(self.levels),
            centroid_hz=self.centroid_hz,
            onset=onset,
            since_onset=t - self.last_onset,
            beat_period=self.beat_period,
            spectrum=self.bars.astype(np.float32).copy(),
            fft_size=self.n,
            rate=self.rate,
        )

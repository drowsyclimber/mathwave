"""Audio analysis: bands, spectral features, onsets, tempo, pitch, chroma and stereo.

Every value a panel can listen to is a "drive": a name with a 0..1 level per frame.
Drives come from four places:

  bands      the seven built-in bands plus any [bands] you define, or band(lo, hi)
  spectrum   rms, centroid, rolloff, flatness, flux
  musical    pitch, note, tonality, kick, snare, hat, phase
  stereo     left, right, side, width, balance

Levels are normalized in dB against a slowly falling recent peak (similar in spirit
to Cava's autosens), then smoothed with attack, decay and an optional peak hold.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

# Built-in frequency bands in Hz.
BANDS: dict[str, tuple[float, float]] = {
    "sub": (20.0, 60.0),
    "bass": (60.0, 250.0),
    "lowmid": (250.0, 500.0),
    "mid": (500.0, 2000.0),
    "highmid": (2000.0, 4000.0),
    "presence": (4000.0, 6000.0),
    "air": (6000.0, 16000.0),
}

FEATURE_INFO: dict[str, str] = {
    "rms": "loudness (RMS)",
    "centroid": "brightness (spectral centroid, 100 Hz..8 kHz)",
    "rolloff": "where the highs end (85% rolloff, 100 Hz..16 kHz)",
    "flatness": "noisiness: 0 tonal, 1 noise-like",
    "flux": "how fast the spectrum is changing",
    "pitch": "bass note pitch within pitch_range (log scale)",
    "note": "dominant pitch class: C=0 .. B=1",
    "tonality": "how strongly one pitch class dominates",
    "kick": "kick trigger: jumps to 1 on each hit, then decays",
    "snare": "snare trigger: jumps to 1 on each hit, then decays",
    "hat": "hi-hat trigger: jumps to 1 on each hit, then decays",
    "phase": "position in the current beat, 0..1",
    "left": "left channel loudness",
    "right": "right channel loudness",
    "side": "loudness of the stereo difference (L - R)",
    "width": "stereo width: 0 mono, 0.5 one side only, 1 out of phase",
    "balance": "0 hard left, 0.5 center, 1 hard right",
}
FEATURES = tuple(FEATURE_INFO)
# Back-compat: the original drive names.
DRIVES = tuple(BANDS) + ("rms", "centroid")

NOTE_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
VALID_FFT = (1024, 2048, 4096, 8192, 16384)


def drive_names(custom_bands: dict | None = None) -> set[str]:
    return set(BANDS) | set(FEATURES) | set(custom_bands or {})


# ---------------------------------------------------------------- settings


@dataclass
class BandSpec:
    lo: float
    hi: float
    mode: str = "mean"          # mean | peak
    range_db: float | None = None
    floor_db: float | None = None
    fall_db: float | None = None  # dB per second the recent peak falls
    fft: int | None = None
    norm: str = "own"           # own | shared


@dataclass
class Smoothing:
    """Rise/fall behavior. *_ms values are frame-rate independent and win over the
    per-frame factors. attack_ms/decay_ms are the time to cover 90% of a change."""

    attack: float = 0.65
    decay: float = 0.90
    attack_ms: float | None = None
    decay_ms: float | None = None
    hold_ms: float = 0.0

    def factors(self, dt: float) -> tuple[float, float]:
        if self.attack_ms is not None:
            a = 1.0 if self.attack_ms <= 0 else 1.0 - 0.1 ** (dt / (self.attack_ms / 1000.0))
        else:
            a = self.attack
        if self.decay_ms is not None:
            d = 0.0 if self.decay_ms <= 0 else 0.1 ** (dt / (self.decay_ms / 1000.0))
        else:
            d = self.decay
        return a, d


class Follower:
    """Envelope follower with attack, decay and peak hold. Works on floats or arrays."""

    def __init__(self, smoothing: Smoothing):
        self.s = smoothing
        self.v = None
        self.held = None

    def __call__(self, x, dt: float):
        x = np.asarray(x, dtype=np.float64)
        if self.v is None or np.shape(self.v) != x.shape:
            self.v = x.copy()
            self.held = np.zeros_like(x)
            return self.v.copy() if self.v.ndim else float(self.v)
        a, d = self.s.factors(dt)
        up = x >= self.v
        self.held = np.where(up, 0.0, self.held + dt)
        holding = self.held < self.s.hold_ms / 1000.0
        rise = self.v + (x - self.v) * a
        fall = np.where(holding, self.v, np.maximum(x, self.v * d))
        self.v = np.where(up, rise, fall)
        return self.v.copy() if self.v.ndim else float(self.v)


@dataclass
class AnalysisSettings:
    rate: int = 44100
    fft_size: int = 2048
    n_bars: int = 48
    sensitivity: float = 1.0
    noise_floor_db: float = -75.0
    dynamic_range_db: float = 40.0
    fall_db_per_s: float = 6.0
    smoothing: Smoothing = field(default_factory=Smoothing)
    bands: dict[str, BandSpec] = field(default_factory=dict)  # custom and overridden built-ins
    # weighting
    tilt_db_per_oct: float = 0.0
    weighting: str = "none"     # none | a
    # multi-resolution: bands entirely below split_hz use low_fft, entirely above use high_fft
    multires: bool = False
    low_fft: int = 8192
    high_fft: int = 1024
    split_hz: float = 250.0
    # onsets
    kick_bands: tuple = ((30.0, 150.0),)
    snare_bands: tuple = ((180.0, 400.0), (2000.0, 5000.0))
    hat_bands: tuple = ((7000.0, 16000.0),)
    onset_sensitivity: float = 1.5
    refractory_ms: float = 120.0
    envelope_ms: float = 200.0
    # tempo
    bpm: float | None = None    # fixed tempo; None = detect from kicks
    bpm_multiplier: float = 1.0
    bpm_range: tuple = (70.0, 180.0)
    # pitch
    pitch_range: tuple = (30.0, 300.0)


def a_weight_db(f: np.ndarray) -> np.ndarray:
    f2 = np.maximum(f, 1.0) ** 2
    ra = (12194.0**2 * f2**2) / ((f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194.0**2))
    return 20.0 * np.log10(ra) + 2.0


# ---------------------------------------------------------------- features


@dataclass
class Features:
    """One frame of analysis. Every level is normalized to 0..1."""

    t: float = 0.0
    levels: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in drive_names()})
    raw: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in drive_names()})
    centroid_hz: float = 0.0
    rolloff_hz: float = 0.0
    onset: bool = False          # kick onset this frame
    since_onset: float = 1e9     # seconds since the last kick
    onsets: dict[str, bool] = field(default_factory=dict)
    beat_period: float = 0.5
    tempo_source: str = "auto"   # auto | fixed | tap
    pitch_hz: float = 0.0
    pitch_conf: float = 0.0
    chroma_pc: int = 0
    spectrum: np.ndarray = field(default_factory=lambda: np.zeros(48, dtype=np.float32))
    raw_spectrum: np.ndarray = field(default_factory=lambda: np.zeros(48, dtype=np.float32))
    band_ranges: dict[str, tuple[float, float, str]] = field(default_factory=dict)
    band_db: dict[str, float] = field(default_factory=dict)
    fft_size: int = 2048
    rate: int = 44100

    def __getitem__(self, key: str) -> float:
        return self.levels[key]

    @property
    def bpm(self) -> float:
        return 60.0 / self.beat_period if self.beat_period > 0 else 0.0

    @property
    def note_name(self) -> str:
        return NOTE_NAMES[self.chroma_pc % 12]

    @property
    def pitch_name(self) -> str:
        if self.pitch_hz <= 0:
            return "–"
        midi = 69 + 12 * math.log2(self.pitch_hz / 440.0)
        n = int(round(midi))
        return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"


class _Adaptive:
    """Adaptive dB normalization: the ceiling tracks the loudest recent value and slowly
    falls; the floor sits a fixed range below it but never below the noise floor."""

    def __init__(self, noise_floor_db: float, dynamic_range_db: float, fall_db_per_s: float):
        self.noise_floor = noise_floor_db
        self.range = max(1.0, dynamic_range_db)
        self.fall = fall_db_per_s
        self.ceiling = noise_floor_db + self.range

    def __call__(self, db, dt: float):
        db = np.asarray(db, dtype=np.float64)
        top = float(np.max(db)) if db.size else self.noise_floor
        self.ceiling = max(top, self.ceiling - self.fall * dt, self.noise_floor + self.range)
        floor = self.ceiling - self.range
        return np.clip((db - floor) / self.range, 0.0, 1.0)


class _Onset:
    """Spectral-flux onset detector over one or more frequency ranges.

    A hit needs three things:
      1. the flux in its ranges jumps above the recent average by `sensitivity`
         standard deviations (and the refractory time has passed);
      2. contrast: the new energy per bin in its weakest range is at least `ratio`
         times the new energy per bin in a reference region. This is what keeps a
         snare's noise from firing the kick, a kick from firing the hats, and a
         hat from firing the snare, even when one drum plays alone;
      3. a gate: every range's loudest bin is within GATE_DB of the loudest bin in
         the whole spectrum, so a range that is nearly empty cannot fire;
      4. the signal is loud enough to count at all."""

    GATE_DB = 40.0

    def __init__(self, masks: list, sensitivity: float, refractory: float, ref=None, ratio: float = 1.0):
        self.masks = masks  # bool masks over the main spectrum's bins
        self.ref = ref
        self.ratio = ratio
        self.sens = sensitivity
        self.refractory = refractory
        self.prev = None
        self.hist: deque[float] = deque(maxlen=48)
        self.last = -1e9
        self.count = 0

    def __call__(self, mag: np.ndarray, t: float, loud: bool) -> bool:
        hit = False
        if self.prev is not None and len(self.prev) == len(mag) and self.masks:
            d = np.maximum(mag - self.prev, 0.0)
            flux = float(sum(d[m].sum() for m in self.masks))
            hist = np.asarray(self.hist) if self.hist else np.zeros(1)
            thresh = hist.mean() + self.sens * hist.std() + 1e-3
            if flux > thresh and (t - self.last) > self.refractory and loud and self._gated(mag) and self._contrast(d):
                hit = True
                self.last = t
                self.count += 1
            self.hist.append(flux)
        self.prev = mag.copy()
        return hit

    def _gated(self, mag: np.ndarray) -> bool:
        floor = (float(mag.max()) + 1e-12) * 10 ** (-self.GATE_DB / 20)
        return all(m.any() and float(mag[m].max()) >= floor for m in self.masks)

    def _contrast(self, d: np.ndarray) -> bool:
        if self.ref is None or not self.ref.any():
            return True
        own = min(float(d[m].mean()) for m in self.masks if m.any())
        return own >= self.ratio * float(d[self.ref].mean())


class Analyzer:
    def __init__(self, rate: int = 44100, fft_size: int = 2048, n_bars: int = 48, attack: float = 0.65,
                 decay: float = 0.90, noise_floor_db: float = -75.0, dynamic_range_db: float = 40.0,
                 sensitivity: float = 1.0, settings: AnalysisSettings | None = None):
        if settings is None:
            settings = AnalysisSettings(rate=rate, fft_size=fft_size, n_bars=n_bars, sensitivity=sensitivity,
                                        noise_floor_db=noise_floor_db, dynamic_range_db=dynamic_range_db,
                                        smoothing=Smoothing(attack=attack, decay=decay))
        self.s = settings
        self.rate = settings.rate
        self.n = settings.fft_size
        self._spec_cache: dict[int, tuple] = {}
        self._mask_cache: dict[tuple, np.ndarray] = {}

        # band table: built-ins (possibly overridden) + custom
        self.bands: dict[str, BandSpec] = {}
        for name, (lo, hi) in BANDS.items():
            self.bands[name] = settings.bands.get(name, BandSpec(lo, hi))
        for name, spec in settings.bands.items():
            self.bands.setdefault(name, spec)
        self._band_norm: dict[str, _Adaptive] = {}
        self._shared_norm = self._adaptive()
        for name in self.bands:
            self._register_norm(name)

        self.names = drive_names({k: v for k, v in self.bands.items() if k not in BANDS})
        self.followers: dict[str, Follower] = {}
        self.levels = {k: 0.0 for k in self.names}
        self.raw = {k: 0.0 for k in self.names}

        self.norm = {k: self._adaptive() for k in ("rms", "flux", "side")}
        self._lr_norm = self._adaptive()
        self.set_bars(settings.n_bars)
        self.bar_norm = self._adaptive()

        main_freqs = self._freqs(self.n)
        mk = lambda bands: [self._mask(self.n, lo, hi) for lo, hi in bands]  # noqa: E731
        refr = settings.refractory_ms / 1000.0
        union = lambda masks: np.logical_or.reduce(masks) if masks else None  # noqa: E731
        sens = settings.onset_sensitivity
        self.onset_det = {
            # kick: new low end must outweigh new highs per bin (a snare's noise is flat)
            "kick": _Onset(mk(settings.kick_bands), sens, max(refr, 0.15),
                           ref=self._mask(self.n, 1000.0, 16000.0), ratio=4.0),
            # snare: every range must carry real new energy next to the hat band
            "snare": _Onset(mk(settings.snare_bands), sens, refr,
                            ref=union(mk(settings.hat_bands)), ratio=0.25),
            # hat: new highs must outweigh new low mids per bin
            "hat": _Onset(mk(settings.hat_bands), sens, max(0.05, refr * 0.6),
                          ref=self._mask(self.n, 100.0, 1500.0), ratio=4.0),
        }
        self.last_hit = {k: -1e9 for k in self.onset_det}
        self._main_freqs = main_freqs
        self.prev_mag = None

        self.beat_period = 60.0 / settings.bpm if settings.bpm else 0.5
        self.taps: list[float] = []
        self.tap_period: float | None = None
        self.centroid_hz = 0.0
        self.rolloff_hz = 0.0
        self.pitch_hz = 0.0
        self.pitch_conf = 0.0
        self.chroma = np.zeros(12)
        self.last_t: float | None = None
        self.prev_low = None  # kept for compatibility

    # -- setup helpers

    def _adaptive(self, spec: BandSpec | None = None) -> _Adaptive:
        s = self.s
        floor = spec.floor_db if spec and spec.floor_db is not None else s.noise_floor_db
        rng = spec.range_db if spec and spec.range_db is not None else s.dynamic_range_db
        fall = spec.fall_db if spec and spec.fall_db is not None else s.fall_db_per_s
        return _Adaptive(floor, rng, fall)

    def _register_norm(self, name: str) -> None:
        spec = self.bands[name]
        if spec.norm != "shared":
            self._band_norm[name] = self._adaptive(spec)

    def add_band(self, name: str, lo: float, hi: float, mode: str = "mean") -> None:
        """Register an inline band(lo, hi) or peak(lo, hi) used by a drive expression."""
        if name in self.bands:
            return
        self.bands[name] = BandSpec(lo, hi, mode)
        self._register_norm(name)
        self.names.add(name)
        self.levels.setdefault(name, 0.0)
        self.raw.setdefault(name, 0.0)

    def _freqs(self, n: int) -> np.ndarray:
        return np.fft.rfftfreq(n, 1.0 / self.rate)

    def _mask(self, n: int, lo: float, hi: float) -> np.ndarray:
        key = (n, lo, hi)
        m = self._mask_cache.get(key)
        if m is None:
            freqs = self._freqs(n)
            m = (freqs >= lo) & (freqs < min(hi, self.rate / 2.0))
            if not m.any():
                m = np.zeros_like(freqs, dtype=bool)
                m[int(np.argmin(np.abs(freqs - (lo + hi) / 2)))] = True
            self._mask_cache[key] = m
        return m

    def band_fft(self, spec: BandSpec) -> int:
        if spec.fft:
            return spec.fft
        if self.s.multires:
            if spec.hi <= self.s.split_hz:
                return self.s.low_fft
            if spec.lo >= self.s.split_hz:
                return self.s.high_fft
        return self.n

    def set_bars(self, n_bars: int) -> None:
        n_bars = max(4, int(n_bars))
        self.n_bars = n_bars
        freqs = self._freqs(self.n)
        hi = min(16000.0, self.rate / 2.0 * 0.95)
        edges = np.geomspace(30.0, hi, n_bars + 1)
        idx = np.searchsorted(freqs, edges)
        self.bar_ranges = []
        for i in range(n_bars):
            a = int(idx[i])
            b = max(int(idx[i + 1]), a + 1)
            self.bar_ranges.append((a, min(b, len(freqs))))
        self.bar_follow = Follower(self.s.smoothing)

    # -- tempo

    def tap(self, t: float) -> None:
        if self.taps and t - self.taps[-1] > 2.0:
            self.taps = []
        self.taps.append(t)
        self.taps = self.taps[-6:]
        if len(self.taps) >= 2:
            self.tap_period = float(np.mean(np.diff(self.taps)))

    def reset_tempo(self) -> None:
        self.taps = []
        self.tap_period = None

    def _fold(self, period: float) -> float:
        lo, hi = self.s.bpm_range
        bpm = 60.0 / period
        for _ in range(8):
            if bpm < lo:
                bpm *= 2
            elif bpm > hi:
                bpm /= 2
            else:
                break
        return 60.0 / bpm

    # -- analysis

    def _spectrum(self, x: np.ndarray, n: int):
        cached = self._spec_cache.get(n)
        if cached is None:
            window = np.hanning(n)
            freqs = self._freqs(n)
            gain_db = np.zeros_like(freqs)
            if self.s.tilt_db_per_oct:
                gain_db += self.s.tilt_db_per_oct * np.log2(np.maximum(freqs, 20.0) / 1000.0)
            if self.s.weighting == "a":
                gain_db += a_weight_db(freqs)
            weight = 10.0 ** (gain_db / 10.0)
            cached = (window, window.sum() / 2.0, freqs, weight)
            self._spec_cache[n] = cached
        window, norm, freqs, weight = cached
        buf = np.zeros(n)
        s = x[-n:]
        buf[n - len(s):] = s
        mag = np.abs(np.fft.rfft(buf * window)) / norm
        power = mag**2
        return freqs, mag, power, power * weight

    def _smooth(self, name: str, value: float, dt: float) -> float:
        f = self.followers.get(name)
        if f is None:
            f = self.followers[name] = Follower(self.s.smoothing)
        return float(f(value, dt))

    def analyze(self, samples: np.ndarray, t: float) -> Features:
        dt = 1.0 / 60.0 if self.last_t is None else max(1e-4, t - self.last_t)
        self.last_t = t
        s = self.s
        arr = np.asarray(samples, dtype=np.float64)
        if arr.ndim == 2:
            left = arr[:, 0]
            right = arr[:, 1] if arr.shape[1] > 1 else arr[:, 0]
            mono = 0.5 * (left + right)
        else:
            mono = arr
            left = right = arr
        eps = 1e-12

        sizes = {self.n} | {self.band_fft(b) for b in self.bands.values()} | {self.pitch_fft}
        spectra = {n: self._spectrum(mono, n) for n in sizes}
        freqs, mag, power, wpower = spectra[self.n]
        wmag = np.sqrt(wpower)

        raw_db: dict[str, float] = {}
        band_db = {}
        shared_names, shared_db = [], []
        for name, spec in self.bands.items():
            n = self.band_fft(spec)
            _, _, p, wp = spectra[n]
            m = self._mask(n, spec.lo, spec.hi)
            vals = wp[m]
            db = 10.0 * np.log10((vals.max() if spec.mode == "peak" else vals.mean()) + eps)
            band_db[name] = db
            if spec.norm == "shared":
                shared_names.append(name)
                shared_db.append(db)
            else:
                raw_db[name] = db

        seg = mono[-self.n:]
        rms = float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0
        loud = rms > 1e-3
        unsmoothed: dict[str, float] = {}
        sens = s.sensitivity

        for name, db in raw_db.items():
            unsmoothed[name] = min(1.0, float(self._band_norm[name](db, dt)) * sens)
        if shared_names:
            vals = self._shared_norm(np.array(shared_db), dt)
            for name, v in zip(shared_names, vals):
                unsmoothed[name] = min(1.0, float(v) * sens)

        unsmoothed["rms"] = min(1.0, float(self.norm["rms"](20.0 * np.log10(rms + 1e-9), dt)) * sens)

        # brightness, rolloff, flatness
        total = wmag.sum()
        if total > 1e-6 and rms > 1e-4:
            self.centroid_hz = 0.8 * self.centroid_hz + 0.2 * float((freqs * wmag).sum() / total)
            cum = np.cumsum(wmag)  # magnitude, as in the usual definition
            k = int(np.searchsorted(cum, 0.85 * cum[-1]))
            self.rolloff_hz = 0.8 * self.rolloff_hz + 0.2 * float(freqs[min(k, len(freqs) - 1)])
            logmap = lambda hz, lo, hi: float(np.clip((np.log(max(hz, lo)) - np.log(lo)) / (np.log(hi) - np.log(lo)), 0, 1))  # noqa: E731
            unsmoothed["centroid"] = logmap(self.centroid_hz, 100.0, 8000.0)
            unsmoothed["rolloff"] = logmap(self.rolloff_hz, 100.0, 16000.0)
            band = (freqs >= 100) & (freqs <= 10000)
            pb = power[band] + 1e-14
            flat = float(np.exp(np.mean(np.log(pb))) / np.mean(pb))
            unsmoothed["flatness"] = float(np.clip((10 * np.log10(flat + 1e-12) + 30.0) / 30.0, 0, 1))
        else:
            unsmoothed["centroid"] = unsmoothed["rolloff"] = unsmoothed["flatness"] = 0.0

        # flux over the whole spectrum
        if self.prev_mag is not None and len(self.prev_mag) == len(mag):
            flux = float(np.maximum(mag - self.prev_mag, 0).sum())
            unsmoothed["flux"] = float(self.norm["flux"](20 * np.log10(flux + 1e-9), dt)) if loud else 0.0
        else:
            unsmoothed["flux"] = 0.0
        self.prev_mag = mag

        # onsets and trigger envelopes
        hits = {}
        env_k = s.envelope_ms / 1000.0
        for name, det in self.onset_det.items():
            hits[name] = det(mag, t, loud)
        for name in self.onset_det:
            if hits[name]:
                if name == "kick":
                    interval = t - self.last_hit["kick"]
                    if 0.2 < interval < 2.0:
                        self.beat_period = 0.8 * self.beat_period + 0.2 * self._fold(interval)
                self.last_hit[name] = t
            since = t - self.last_hit[name]
            unsmoothed[name] = float(0.1 ** (since / env_k)) if env_k > 0 else float(since < dt)

        # tempo: tap wins, then fixed, then detected; the multiplier applies last
        if self.tap_period:
            period, source = self.tap_period, "tap"
        elif s.bpm:
            period, source = 60.0 / s.bpm, "fixed"
        else:
            period, source = self.beat_period, "auto"
        period = period / max(0.125, s.bpm_multiplier)
        anchor = self.last_hit["kick"] if self.last_hit["kick"] > -1e8 else 0.0
        unsmoothed["phase"] = float(((t - anchor) / period) % 1.0) if period > 0 else 0.0

        # pitch (bass fundamental) and chroma from the pitch-resolution spectrum
        pf, pm, _, _ = spectra[self.pitch_fft]
        self._pitch(pf, pm, loud)
        lo, hi = s.pitch_range
        if self.pitch_hz > 0:
            unsmoothed["pitch"] = float(np.clip(np.log(self.pitch_hz / lo) / np.log(hi / lo), 0, 1))
        else:
            unsmoothed["pitch"] = self.raw.get("pitch", 0.0)
        pc, tonal = self._chroma(pf, pm, loud)
        unsmoothed["note"] = pc / 11.0
        unsmoothed["tonality"] = tonal

        # stereo
        l_rms = float(np.sqrt(np.mean(left[-self.n:] ** 2)))
        r_rms = float(np.sqrt(np.mean(right[-self.n:] ** 2)))
        side = 0.5 * (left[-self.n:] - right[-self.n:])
        s_rms = float(np.sqrt(np.mean(side**2)))
        lr = self._lr_norm(np.array([20 * np.log10(l_rms + 1e-9), 20 * np.log10(r_rms + 1e-9)]), dt)
        unsmoothed["left"], unsmoothed["right"] = float(lr[0]) * sens, float(lr[1]) * sens
        unsmoothed["side"] = float(self.norm["side"](20 * np.log10(s_rms + 1e-9), dt)) if s_rms > 1e-5 else 0.0
        m_rms = float(np.sqrt(np.mean((0.5 * (left[-self.n:] + right[-self.n:])) ** 2)))
        # 0 mono, 0.5 one side only, 1 fully out of phase
        unsmoothed["width"] = float(np.clip(s_rms / (m_rms + s_rms + 1e-9), 0, 1)) if max(l_rms, r_rms) > 1e-3 else 0.0
        unsmoothed["balance"] = float(np.clip(0.5 + 0.5 * (r_rms - l_rms) / (r_rms + l_rms + 1e-9), 0, 1))

        # smoothing: triggers and phase pass through, everything else follows
        passthrough = {"kick", "snare", "hat", "phase", "note"}
        for name, v in unsmoothed.items():
            v = float(min(1.0, max(0.0, v)))
            unsmoothed[name] = v
            self.levels[name] = v if name in passthrough else self._smooth(name, v, dt)
        self.raw = unsmoothed

        # Cava-style bars
        bar_db = np.array([10.0 * np.log10(wpower[a:b].mean() + eps) for a, b in self.bar_ranges])
        target = np.minimum(self.bar_norm(bar_db, dt) * sens, 1.0)
        bars = self.bar_follow(target, dt)

        return Features(
            t=t,
            levels=dict(self.levels),
            raw=dict(unsmoothed),
            centroid_hz=self.centroid_hz,
            rolloff_hz=self.rolloff_hz,
            onset=hits["kick"],
            since_onset=t - self.last_hit["kick"],
            onsets=hits,
            beat_period=period,
            tempo_source=source,
            pitch_hz=self.pitch_hz,
            pitch_conf=self.pitch_conf,
            chroma_pc=int(np.argmax(self.chroma)) if self.chroma.any() else 0,
            spectrum=np.asarray(bars, dtype=np.float32),
            raw_spectrum=target.astype(np.float32),
            band_ranges={k: (b.lo, b.hi, b.mode) for k, b in self.bands.items()},
            band_db=band_db,
            fft_size=self.n,
            rate=self.rate,
        )

    @property
    def pitch_fft(self) -> int:
        return max(8192, self.n)

    @property
    def need(self) -> int:
        """How many samples analyze() wants: the largest FFT in use."""
        return max([self.n, self.pitch_fft] + [self.band_fft(b) for b in self.bands.values()])

    def _pitch(self, freqs: np.ndarray, mag: np.ndarray, loud: bool) -> None:
        lo, hi = self.s.pitch_range
        m = (freqs >= lo) & (freqs <= hi)
        if not loud or not m.any():
            self.pitch_conf *= 0.9
            return
        idx = np.flatnonzero(m)
        seg = mag[idx]
        k = int(np.argmax(seg))
        peak = seg[k]
        conf = float(peak / (np.median(seg) + 1e-12))
        if conf < 4.0:
            self.pitch_conf = 0.9 * self.pitch_conf
            return
        df = freqs[1] - freqs[0]

        def at(hz: float) -> float:
            j = int(round(hz / df))
            return float(mag[max(0, j - 1): j + 2].max()) if j + 1 < len(mag) else 0.0

        # Weighted subharmonic summation: each candidate scores its first five harmonics,
        # with later ones counting less. A weak fundamental under strong harmonics still
        # wins, and a pure tone beats its own subharmonics.
        cands = set()
        for j in idx[np.argsort(seg)[-4:]]:
            if mag[j] < 0.2 * peak:
                continue
            for div in (1, 2, 3):
                hz = freqs[j] / div
                if hz >= lo:
                    cands.add(int(round(hz / df)))
        best, best_score = idx[k], -1.0
        for c in cands:
            hz = c * df
            if at(hz) < 0.15 * peak:
                continue
            score = sum(0.84 ** (h - 1) * at(h * hz) for h in range(1, 6))
            if score > best_score * 1.02:
                best, best_score = c, score
        # snap to the local maximum, then parabolic interpolation on log magnitude
        lo_i, hi_i = max(1, best - 1), min(len(mag) - 2, best + 1)
        i = lo_i + int(np.argmax(mag[lo_i: hi_i + 1]))
        a, b, c = np.log(mag[i - 1] + 1e-12), np.log(mag[i] + 1e-12), np.log(mag[i + 1] + 1e-12)
        denom = a - 2 * b + c
        off = 0.5 * (a - c) / denom if denom != 0 else 0.0
        f0 = (i + off) * df
        if self.pitch_hz > 0 and abs(f0 / self.pitch_hz - 1) < 0.06:
            self.pitch_hz = 0.6 * self.pitch_hz + 0.4 * f0  # same note: settle smoothly
        else:
            self.pitch_hz = f0  # a new note: jump straight to it
        self.pitch_conf = min(1.0, conf / 40.0)

    def _chroma(self, freqs: np.ndarray, mag: np.ndarray, loud: bool) -> tuple[int, float]:
        if not hasattr(self, "_pc_idx") or len(self._pc_idx) != len(freqs):
            band = (freqs >= 50.0) & (freqs <= 2000.0)
            midi = 69 + 12 * np.log2(np.maximum(freqs, 1.0) / 440.0)
            self._pc_band = band
            self._pc_idx = np.mod(np.round(midi).astype(int), 12)
        if loud:
            w = mag * self._pc_band
            vec = np.bincount(self._pc_idx, weights=w, minlength=12)[:12]
            if vec.sum() > 0:
                vec = vec / vec.sum()
                self.chroma = 0.85 * self.chroma + 0.15 * vec
        total = self.chroma.sum()
        if total <= 0:
            return 0, 0.0
        pc = int(np.argmax(self.chroma))
        share = self.chroma[pc] / total
        return pc, float(np.clip((share - 1 / 12) / (0.5 - 1 / 12), 0, 1))


def settings_replace(s: AnalysisSettings, **kw) -> AnalysisSettings:
    return replace(s, **kw)

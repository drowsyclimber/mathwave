"""The demo track: every sound is a formula, no samples.

The arrangement follows the reel idea. A layer is added every 16 beats, and each
one is harsher than the last: 808, dive bomb, Reese, tan screech, FM wobble and a
Fourier-series saw lead. The loop then starts again.
"""

from __future__ import annotations

import wave

import numpy as np

BPM = 140.0
BEAT = 60.0 / BPM
SECTION_BEATS = 16
N_LAYERS = 6
TAU = 2.0 * np.pi


def kick_808(tb: np.ndarray) -> np.ndarray:
    """y = e^(-t/τa) · sin(2π ∫ f dt), f(t) = f1 + (f0 - f1) e^(-t/τp)."""
    f0, f1, tau_p, tau_a = 160.0, 46.0, 0.035, 0.38
    phase = TAU * (f1 * tb + (f0 - f1) * tau_p * (1.0 - np.exp(-tb / tau_p)))
    return np.exp(-tb / tau_a) * np.sin(phase)


def dive_bomb(td: np.ndarray) -> np.ndarray:
    """f(t) = f1 + c / (t + t0): the 1/x pitch dive. Phase = 2π(f1 t + c ln((t+t0)/t0))."""
    f1, c, t0 = 70.0, 55.0, 0.04
    td = np.maximum(td, 0.0)
    phase = TAU * (f1 * td + c * np.log((td + t0) / t0))
    env = np.clip(1.0 - td / (2 * BEAT), 0.0, 1.0)
    return np.tanh(3.0 * np.sin(phase)) * env


def reese(t: np.ndarray, f: np.ndarray, delta: float = 2.7) -> np.ndarray:
    """sin(a) + sin(b) = 2 sin((a+b)/2) cos((a-b)/2): two detuned waves that beat."""
    a = TAU * f * t
    b = TAU * (f + delta) * t
    return np.tanh(1.4 * (np.sin(a) + np.sin(b) + 0.5 * np.sin(2 * a) + 0.5 * np.sin(2 * b)))


def screech(t: np.ndarray, f: np.ndarray, gain: float = 0.35) -> np.ndarray:
    """clip(g · tan(2π f t), -1, 1): the asymptotes turn into a harsh buzz."""
    return np.clip(gain * np.tan(TAU * f * t), -1.0, 1.0)


def fm_wobble(t: np.ndarray, pos: np.ndarray, fc: float = 82.4) -> np.ndarray:
    """sin(ωc t + I(t) · sin(ωm t)), with the index I swept by an eighth-note LFO."""
    index = 1.0 + 7.0 * (0.5 + 0.5 * np.sin(TAU * pos * 2.0))
    return np.tanh(1.5 * np.sin(TAU * fc * t + index * np.sin(TAU * 2.0 * fc * t)))


def fourier_saw(t: np.ndarray, f: np.ndarray, n: int = 14) -> np.ndarray:
    """(2/π) Σ (-1)^(k+1) sin(k ω t) / k for k = 1..N."""
    y = np.zeros_like(t)
    for k in range(1, n + 1):
        y += ((-1) ** (k + 1)) * np.sin(TAU * k * f * t) / k
    return (2.0 / np.pi) * y


def render(t: np.ndarray, layers: int | None = None, stereo: bool = False) -> np.ndarray:
    """Render the demo at absolute times t (seconds). Returns float32 in -1..1.

    With stereo=True returns shape (n, 2). Kick, dive, Reese and wobble sit in the
    center; the screech ping-pongs between the speakers each bar and the saw arpeggio
    pans across the stereo field, so the stereo drives (balance, side, width) move.
    The mid channel (L+R)/2 equals the mono render."""
    t = np.asarray(t, dtype=np.float64)
    pos = t / BEAT
    section = np.floor(pos / SECTION_BEATS).astype(np.int64) % N_LAYERS
    if layers is not None:
        section = np.full_like(section, layers - 1)
    bar_beat = pos % 4.0
    bar = np.floor(pos / 4.0).astype(np.int64)
    tb = (pos % 1.0) * BEAT

    notes = np.array([55.0, 55.0, 65.41, 49.0])[bar % 4]
    out = 0.85 * kick_808(tb)
    panned = []  # (signal, pan in -1..1)

    gate = (section >= 1) & (bar_beat >= 2.0)
    out += np.where(gate, 0.22 * dive_bomb((bar_beat - 2.0) * BEAT), 0.0)

    gate = section >= 2
    duck = 1.0 - 0.8 * np.exp(-tb / 0.08)  # sidechain under the kick
    out += np.where(gate, 0.20 * duck * reese(t, notes), 0.0)

    gate = (section >= 3) & ((pos % 1.0) >= 0.5) & ((pos % 1.0) < 0.75)
    sweep = 880.0 * 2.0 ** ((pos % 4.0) / 4.0)
    scr = np.where(gate, 0.10 * screech(t, sweep), 0.0)
    out += scr
    panned.append((scr, np.where(bar % 2 == 0, -0.8, 0.8)))

    gate = section >= 4
    out += np.where(gate, 0.16 * duck * fm_wobble(t, pos), 0.0)

    gate = (section >= 5) & ((pos * 4.0) % 1.0 < 0.6)
    arp = np.array([440.0, 523.25, 659.25, 880.0])[np.floor(pos * 4.0).astype(np.int64) % 4]
    saw = np.where(gate, 0.07 * fourier_saw(t, arp), 0.0)
    out += saw
    panned.append((saw, np.sin(TAU * pos / 8.0) * 0.9))

    if not stereo:
        return (0.9 * np.tanh(out)).astype(np.float32)
    # Move each panned layer from the center towards one side; mid stays equal to mono.
    side = np.zeros_like(out)
    for sig, pan in panned:
        side += sig * pan
    left, right = out + side, out - side
    mono = 0.9 * np.tanh(out)
    gain = np.where(np.abs(out) > 1e-9, mono / np.where(np.abs(out) > 1e-9, out, 1.0), 0.9)
    return np.stack([left * gain, right * gain], axis=1).astype(np.float32)


def write_wav(path: str, seconds: float = 42.0, rate: int = 44100) -> None:
    n = int(seconds * rate)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        step = rate  # one second at a time keeps memory flat
        for start in range(0, n, step):
            t = np.arange(start, min(n, start + step)) / rate
            pcm = (np.clip(render(t, stereo=True), -1, 1) * 32767.0).astype("<i2")
            w.writeframes(pcm.tobytes())

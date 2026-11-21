"""The function panels. Each one is a formula whose parameters are driven by the music.

A panel exposes:
  formula   the symbolic formula that is plotted (printed in the panel footer)
  identity  an optional second line (an identity or the matching pitch law)
  drive     the expression that produces its 0..1 level (a band, or a mix of bands)
  sound     what the same formula sounds like when you synthesize it
  KNOBS     panel-specific constants that [tune.<key>] sections can override
  layers()  numeric curves to draw for the current frame
  live()    the formula's parameters right now, printed under the formula

Every frame a panel turns the analysis into one level with Tuning.apply(), then maps
that level onto its formula's parameters using its knobs.
"""

from __future__ import annotations

import ast
import difflib
import math
from dataclasses import dataclass

import numpy as np

from .dsp import DRIVES, Features

TAU = 2.0 * np.pi

# ---------------------------------------------------------------- safe expressions

_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log,
    "sqrt": np.sqrt, "abs": np.abs, "tanh": np.tanh, "sinh": np.sinh, "cosh": np.cosh,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan, "sinc": np.sinc,
    "sign": np.sign, "floor": np.floor, "mod": np.mod, "clip": np.clip,
    "min": np.minimum, "max": np.maximum,
}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": TAU}
_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv, ast.USub, ast.UAdd,
    ast.Compare, ast.Lt, ast.Gt, ast.LtE, ast.GtE,
)
DRIVE_NAMES = set(DRIVES)


def compile_expr(src: str, names: set[str]):
    """Compile a math expression after checking it only uses safe syntax and names."""
    tree = ast.parse(src.replace("^", "**"), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _NODES):
            raise ValueError(f"'{src}': {type(node).__name__} is not allowed")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _FUNCS):
            raise ValueError(f"'{src}': only these functions are allowed: {', '.join(sorted(_FUNCS))}")
        if isinstance(node, ast.Name) and node.id not in names and node.id not in _FUNCS and node.id not in _CONSTS:
            raise ValueError(f"'{src}': unknown name '{node.id}'")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError(f"'{src}': only numeric constants are allowed")
    return compile(tree, "<mathwave>", "eval")


def _eval(code, env) -> float:
    return float(eval(code, {"__builtins__": {}}, dict(_FUNCS, **_CONSTS, **env)))


# ---------------------------------------------------------------- tuning

COMMON_KEYS = {
    "drive": "band or expression for the level, e.g. sub or max(sub, 0.7*bass)",
    "gain": "multiply the level before shaping (default 1)",
    "threshold": "ignore everything below this level, 0..1 (default 0)",
    "curve": "exponent: 1 linear, 2 punchier, 0.5 more sensitive (default 1)",
    "min": "level the panel rests at, 0..1 (default 0)",
    "max": "level the panel peaks at, 0..1 (default 1); min > max inverts",
    "attack": "this panel's rise speed, 0..1, higher is snappier (default: global)",
    "decay": "this panel's fall speed, 0..1, higher is slower (default: global)",
}


@dataclass
class Tuning:
    gain: float = 1.0
    threshold: float = 0.0
    curve: float = 1.0
    lo: float = 0.0
    hi: float = 1.0
    attack: float | None = None
    decay: float | None = None

    @property
    def own_smoothing(self) -> bool:
        return self.attack is not None or self.decay is not None

    @property
    def is_default(self) -> bool:
        return self == Tuning()

    def shape(self, x):
        """gain -> threshold -> curve -> range. Works on floats and arrays."""
        x = np.clip(np.asarray(x, dtype=np.float64) * self.gain, 0.0, 1.0)
        if self.threshold > 0:
            x = np.clip((x - self.threshold) / (1.0 - self.threshold), 0.0, 1.0)
        if self.curve != 1.0:
            x = x**self.curve
        return self.lo + (self.hi - self.lo) * x

    def summary(self) -> str:
        parts = []
        if self.gain != 1.0:
            parts.append(f"gain={self.gain:g}")
        if self.threshold:
            parts.append(f"threshold={self.threshold:g}")
        if self.curve != 1.0:
            parts.append(f"curve={self.curve:g}")
        if (self.lo, self.hi) != (0.0, 1.0):
            parts.append(f"range={self.lo:g}..{self.hi:g}")
        if self.attack is not None:
            parts.append(f"attack={self.attack:g}")
        if self.decay is not None:
            parts.append(f"decay={self.decay:g}")
        return " ".join(parts)


def _float(key: str, val: str, lo: float | None = None, hi: float | None = None, lo_open=False, hi_open=False) -> float:
    try:
        v = float(val)
    except ValueError:
        raise ValueError(f"{key} = {val}: not a number")
    if not math.isfinite(v):
        raise ValueError(f"{key} = {val}: must be finite")
    if lo is not None and (v < lo or (lo_open and v == lo)):
        raise ValueError(f"{key} = {val}: must be {'>' if lo_open else '>='} {lo:g}")
    if hi is not None and (v > hi or (hi_open and v == hi)):
        raise ValueError(f"{key} = {val}: must be {'<' if hi_open else '<='} {hi:g}")
    return v


def _breaks(y: np.ndarray, where: np.ndarray) -> np.ndarray:
    """Insert NaNs where a discontinuity happens so the renderer lifts the pen."""
    y = y.copy()
    jump = np.zeros_like(where, dtype=bool)
    jump[1:] = where[1:] != where[:-1]
    y[jump] = np.nan
    return y


# ---------------------------------------------------------------- panels


class Panel:
    key = "base"
    title = "panel"
    formula = ""
    identity = ""
    default_drive = "rms"
    sound = ""
    domain = (0.0, 1.0)
    yrange = (-1.1, 1.1)
    # name -> (default, description). Overridable from [tune.<key>].
    KNOBS: dict[str, tuple[float, str]] = {}
    INT_KNOBS: set[str] = set()

    def __init__(self):
        self.p: dict[str, float] = {}
        self.level = 0.0
        self.k = {name: v for name, (v, _) in self.KNOBS.items()}
        self.tune = Tuning()
        self.global_attack = 0.65
        self.global_decay = 0.90
        self.set_drive(self.default_drive)

    # -- tuning

    def set_drive(self, src: str) -> None:
        self._drive_code = compile_expr(src, DRIVE_NAMES)
        self.drive = src.strip()

    def configure(self, section: dict[str, str], global_attack: float = 0.65, global_decay: float = 0.90) -> None:
        """Apply a [tune.<key>] section. Raises ValueError with a readable message."""
        self.global_attack, self.global_decay = global_attack, global_decay
        t = Tuning()
        for raw_key, val in section.items():
            key = raw_key.strip().lower()
            val = val.strip()
            if key == "drive":
                self.set_drive(val)
            elif key == "gain":
                t.gain = _float(key, val, 0.0, lo_open=True)
            elif key == "threshold":
                t.threshold = _float(key, val, 0.0, 1.0, hi_open=True)
            elif key == "curve":
                t.curve = _float(key, val, 0.05, 20.0)
            elif key == "min":
                t.lo = _float(key, val, 0.0, 1.0)
            elif key == "max":
                t.hi = _float(key, val, 0.0, 1.0)
            elif key == "attack":
                t.attack = _float(key, val, 0.0, 1.0, lo_open=True)
            elif key == "decay":
                t.decay = _float(key, val, 0.0, 1.0, hi_open=True)
            elif key in self.k:
                v = _float(key, val)
                if key in self.INT_KNOBS:
                    if v < 1:
                        raise ValueError(f"{key} = {val}: must be at least 1")
                    v = float(int(round(v)))
                self.k[key] = v
            else:
                valid = sorted(COMMON_KEYS) + sorted(self.k)
                close = difflib.get_close_matches(key, valid, n=1)
                hint = f" (did you mean {close[0]}?)" if close else ""
                raise ValueError(f"unknown key '{key}'{hint}; see mathwave --list-knobs {self.key}")
        self.tune = t
        self._smoothed = None

    def drive_level(self, f: Features) -> float:
        """The panel's tuned 0..1 level for this frame."""
        own = self.tune.own_smoothing
        levels = f.raw if own else f.levels
        x = float(self.tune.shape(_eval(self._drive_code, levels)))
        if not own:
            return x
        attack = self.tune.attack if self.tune.attack is not None else self.global_attack
        decay = self.tune.decay if self.tune.decay is not None else self.global_decay
        prev = getattr(self, "_smoothed", None)
        if prev is None:
            prev = x
        prev = prev + (x - prev) * attack if x > prev else max(x, prev * decay)
        self._smoothed = prev
        return prev

    def tuning_summary(self) -> str:
        parts = []
        if self.drive != self.default_drive:
            parts.append(f"drive={self.drive}")
        s = self.tune.summary()
        if s:
            parts.append(s)
        changed = [f"{n}={v:g}" for n, v in self.k.items() if v != self.KNOBS[n][0]]
        return " ".join(parts + changed)

    # -- drawing

    def update(self, f: Features, dt: float) -> None:
        self.level = self.drive_level(f)

    def layers(self, x: np.ndarray) -> list[tuple]:
        """Return a list of layers: ("curve", y, style) | ("hline", y0, style) |
        ("vline", x0, style) | ("bars", heights, style) | ("points", y, style). style is "main" or "aux"."""
        return []

    def live(self) -> str:
        return " ".join(f"{k}={v:.2f}" for k, v in self.p.items())


def _lerp(lo: float, hi: float, t: float) -> float:
    return lo + (hi - lo) * t


class Exp808(Panel):
    key = "808"
    title = "808 · exponential"
    formula = "y = A·e^(−kx)·sin(2π·f(x)·x)"
    identity = "f(x) = f₁ + (f₀ − f₁)·e^(−x/τ)"
    default_drive = "max(sub, 0.7*bass)"
    sound = "An 808 kick: a sine whose pitch and volume both decay exponentially."
    KNOBS = {
        "amp_min": (0.08, "height A when the level is 0"),
        "f0_min": (9.0, "start pitch f₀ at level 0 (cycles across the panel)"),
        "f0_max": (16.0, "start pitch f₀ at level 1"),
        "f1": (3.0, "end pitch f₁ (cycles across the panel)"),
        "tau": (0.22, "pitch-drop time τ"),
        "k_min": (2.0, "decay rate k at level 1 (lower rings longer)"),
        "k_max": (5.0, "decay rate k at level 0"),
        "reveal": (0.35, "seconds to redraw the curve after each kick (0 = no redraw)"),
    }

    def __init__(self):
        super().__init__()
        self.reveal = 1.0

    def update(self, f, dt):
        super().update(f, dt)
        lv, k = self.level, self.k
        # On every kick the curve is redrawn from x=0, like the note being played.
        self.reveal = 1.0 if k["reveal"] <= 0 else min(1.0, f.since_onset / k["reveal"])
        self.p = {
            "A": _lerp(k["amp_min"], 1.0, lv),
            "k": _lerp(k["k_max"], k["k_min"], lv),
            "f₀": _lerp(k["f0_min"], k["f0_max"], lv),
            "f₁": k["f1"],
            "τ": max(1e-3, k["tau"]),
        }

    def _y(self, x):
        p = self.p
        fx = p["f₁"] + (p["f₀"] - p["f₁"]) * np.exp(-x / p["τ"])
        return p["A"] * np.exp(-p["k"] * x) * np.sin(TAU * fx * x)

    def layers(self, x):
        y = self._y(x)
        y[x > self.reveal] = np.nan
        env = self.p["A"] * np.exp(-self.p["k"] * x)
        return [("curve", env, "aux"), ("curve", -env, "aux"), ("hline", 0.0, "aux"), ("curve", y, "main")]

    def live(self):
        p = self.p
        return f"A={p['A']:.2f} k={p['k']:.2f} f₀={p['f₀']:.1f} f₁={p['f₁']:g} τ={p['τ']:.2f}"


class DiveBomb(Panel):
    key = "dive"
    title = "dive bomb · 1/x"
    formula = "y = a / (x − p)"
    identity = "pitch: f(t) = f₁ + c/(t + t₀)"
    default_drive = "bass"
    sound = "A dive bomb: pitch follows c/(t+t₀), falling fast and then leveling off."
    domain = (-1.0, 1.0)
    yrange = (-1.05, 1.05)
    KNOBS = {
        "a_min": (0.03, "steepness a at level 0"),
        "a_max": (0.45, "steepness a at level 1"),
        "sweep": (0.75, "how far the asymptote p swings from the center"),
        "beats": (2.0, "beats per asymptote sweep cycle"),
    }

    def __init__(self):
        super().__init__()
        self.phase = 0.0

    def update(self, f, dt):
        super().update(f, dt)
        k = self.k
        self.phase += dt * TAU / max(0.2, max(0.1, k["beats"]) * f.beat_period)
        self.p = {"a": _lerp(k["a_min"], k["a_max"], self.level), "p": k["sweep"] * math.sin(self.phase)}

    def layers(self, x):
        a, p = self.p["a"], self.p["p"]
        with np.errstate(divide="ignore", invalid="ignore"):
            y = a / (x - p)
        y = _breaks(np.clip(y, -1.2, 1.2), x > p)
        return [("hline", 0.0, "aux"), ("vline", p, "aux"), ("curve", y, "main")]

    def live(self):
        return f"a={self.p['a']:.3f}  p={self.p['p']:+.2f}  asymptote x={self.p['p']:+.2f}"


class Reese(Panel):
    key = "reese"
    title = "Reese · trig identity"
    formula = "y = sin(ax) + sin(bx)"
    identity = "  = 2·sin((a+b)x/2)·cos((a−b)x/2)"
    default_drive = "max(lowmid, mid)"
    sound = "A Reese bass: two detuned waves beat against each other and growl."
    domain = (0.0, TAU)
    yrange = (-1.1, 1.1)
    KNOBS = {
        "a": (14.0, "base frequency a"),
        "detune_min": (0.6, "detune b − a at level 0"),
        "detune_max": (5.6, "detune b − a at level 1"),
        "amp_min": (0.1, "height A at level 0"),
        "drift": (6.0, "extra scroll speed at level 1"),
    }

    def __init__(self):
        super().__init__()
        self.phase = 0.0

    def update(self, f, dt):
        super().update(f, dt)
        k, lv = self.k, self.level
        self.phase += dt * (2.0 + k["drift"] * lv)
        a = k["a"]
        self.p = {"a": a, "b": a + _lerp(k["detune_min"], k["detune_max"], lv), "A": _lerp(k["amp_min"], 1.0, lv)}

    def layers(self, x):
        a, b, amp = self.p["a"], self.p["b"], self.p["A"]
        y = amp * 0.5 * (np.sin(a * x + self.phase) + np.sin(b * x + 1.3 * self.phase))
        env = amp * np.cos((a - b) * x / 2.0 - 0.15 * self.phase)
        return [("curve", env, "aux"), ("curve", -env, "aux"), ("curve", y, "main")]

    def live(self):
        p = self.p
        return f"a={p['a']:.1f} b={p['b']:.2f} Δ={p['b'] - p['a']:.2f} A={p['A']:.2f}"


class TanScreech(Panel):
    key = "tan"
    title = "screech · tan(x)"
    formula = "y = clip(A·tan(ωx + φ), −1, 1)"
    identity = "asymptotes at ωx + φ = π/2 + nπ"
    default_drive = "max(presence, 0.6*highmid)"
    sound = "A screech: tan(x) blows up at its asymptotes, and clipping turns that into buzz."
    yrange = (-1.1, 1.1)
    KNOBS = {
        "amp_min": (0.12, "height A at level 0"),
        "freq_min": (1.5, "ω at level 0, in half-periods across the panel"),
        "freq_max": (10.5, "ω at level 1, in half-periods across the panel"),
        "scroll": (1.5, "extra scroll speed at level 1"),
    }

    def __init__(self):
        super().__init__()
        self.phi = 0.0

    def update(self, f, dt):
        super().update(f, dt)
        k, lv = self.k, self.level
        self.phi = (self.phi + dt * TAU * (0.15 + k["scroll"] * lv)) % math.pi
        self.p = {"A": _lerp(k["amp_min"], 1.0, lv), "ω": math.pi * _lerp(k["freq_min"], k["freq_max"], lv), "φ": self.phi}

    def layers(self, x):
        arg = self.p["ω"] * x + self.p["φ"]
        y = np.clip(self.p["A"] * np.tan(arg), -1.0, 1.0)
        branch = np.floor((arg + math.pi / 2) / math.pi)
        return [("hline", 0.0, "aux"), ("curve", _breaks(y, branch), "main")]

    def live(self):
        p = self.p
        return f"A={p['A']:.2f} ω={p['ω']:.1f} φ={p['φ']:.2f}"


class FMWobble(Panel):
    key = "fm"
    title = "wobble · FM synthesis"
    formula = "y = sin(ωc·x + I·sin(ωm·x + φ))"
    identity = "φ advances one cycle per beat"
    default_drive = "max(highmid, 0.5*mid)"
    sound = "Dubstep wobble: frequency modulation, with the index I swept by an LFO."
    domain = (0.0, 1.0)
    KNOBS = {
        "index_min": (0.3, "modulation index I at level 0"),
        "index_max": (8.3, "modulation index I at level 1"),
        "carrier": (5.0, "carrier ωc, in cycles across the panel"),
        "mod": (1.5, "modulator ωm, in cycles across the panel"),
        "beats": (1.0, "beats per φ cycle (0.5 = twice per beat)"),
    }

    def __init__(self):
        super().__init__()
        self.phi = 0.0
        self.bpm = 0.0

    def update(self, f, dt):
        super().update(f, dt)
        k = self.k
        self.phi = (self.phi + dt * TAU / max(0.1, max(0.05, k["beats"]) * f.beat_period)) % TAU
        self.bpm = f.bpm
        self.p = {"I": _lerp(k["index_min"], k["index_max"], self.level), "ωc": TAU * k["carrier"], "ωm": TAU * k["mod"]}

    def layers(self, x):
        p = self.p
        mod = np.sin(p["ωm"] * x + self.phi)
        y = np.sin(p["ωc"] * x + p["I"] * mod)
        return [("curve", 0.35 * mod, "aux"), ("curve", y, "main")]

    def live(self):
        p = self.p
        return f"I={p['I']:.2f} ωc={p['ωc']:.1f} ωm={p['ωm']:.1f} BPM≈{self.bpm:.0f}"


class FourierSaw(Panel):
    key = "saw"
    title = "saw · Fourier series"
    formula = "y = (2/π)·Σₖ₌₁ᴺ(−1)^(k+1)·sin(kx)/k"
    identity = "N grows with brightness, Gibbs ≈ 9%"
    default_drive = "centroid"
    sound = "A saw lead built one harmonic at a time; brighter music adds harmonics."
    domain = (-2.0 * math.pi, 2.0 * math.pi)
    yrange = (-1.3, 1.3)
    KNOBS = {
        "n_min": (1.0, "harmonics N at level 0"),
        "n_max": (30.0, "harmonics N at level 1"),
        "speed": (1.2, "scroll speed"),
    }
    INT_KNOBS = {"n_min", "n_max"}

    def __init__(self):
        super().__init__()
        self.phase = 0.0
        self.hz = 0.0

    def update(self, f, dt):
        super().update(f, dt)
        k = self.k
        self.phase = (self.phase + dt * k["speed"]) % TAU
        self.hz = f.centroid_hz
        n = round(_lerp(k["n_min"], k["n_max"], self.level))
        self.p = {"N": float(min(200, max(1, n)))}

    def layers(self, x):
        n = int(self.p["N"])
        k = np.arange(1, n + 1)[:, None]
        xs = x[None, :] + self.phase
        y = (2.0 / np.pi) * (((-1.0) ** (k + 1)) * np.sin(k * xs) / k).sum(axis=0)
        ideal = ((x + self.phase + np.pi) % TAU - np.pi) / np.pi
        ideal = _breaks(ideal, np.floor((x + self.phase + np.pi) / TAU))
        return [("curve", ideal, "aux"), ("curve", y, "main")]

    def live(self):
        return f"N={int(self.p['N'])} harmonics  centroid≈{self.hz / 1000:.1f} kHz"


class Sinc(Panel):
    key = "sinc"
    title = "sinc · impulse"
    formula = "y = A·sin(πωx)/(πωx)"
    identity = "transform of a brick-wall filter"
    default_drive = "air"
    sound = "A click with no aliasing: sinc is a perfect impulse limited to a band."
    domain = (-1.0, 1.0)
    yrange = (-0.3, 1.05)
    KNOBS = {
        "amp_min": (0.15, "height A at level 0"),
        "width_min": (2.0, "ω at level 0 (higher is narrower)"),
        "width_max": (18.0, "ω at level 1"),
    }

    def update(self, f, dt):
        super().update(f, dt)
        k, lv = self.k, self.level
        self.p = {"A": _lerp(k["amp_min"], 1.0, lv), "ω": max(0.1, _lerp(k["width_min"], k["width_max"], lv))}

    def layers(self, x):
        y = self.p["A"] * np.sinc(self.p["ω"] * x)  # numpy sinc is sin(πx)/(πx)
        return [("hline", 0.0, "aux"), ("curve", y, "main")]

    def live(self):
        return f"A={self.p['A']:.2f} ω={self.p['ω']:.1f} zeros every {1 / self.p['ω']:.3f}"


class Logistic(Panel):
    key = "logistic"
    title = "chaos · logistic map"
    formula = "xₙ₊₁ = r·xₙ·(1 − xₙ)"
    identity = "chaos once r > 3.5699"
    default_drive = "rms"
    sound = "Iterated, the logistic map goes from a steady tone to noise as r rises."
    yrange = (-0.02, 1.02)
    HISTORY = 600
    KNOBS = {
        "r_min": (2.8, "r at level 0 (below 3 is a fixed point)"),
        "r_max": (4.0, "r at level 1 (4 is full chaos)"),
        "steps": (4.0, "iterations added per frame (scroll speed)"),
    }
    INT_KNOBS = {"steps"}

    def __init__(self):
        super().__init__()
        self.hist = np.full(self.HISTORY, 0.5)
        self.x = 0.4
        self.p = {"r": 2.8}

    @staticmethod
    def regime(r: float) -> str:
        if r < 3.0:
            return "fixed point"
        if r < 3.4495:
            return "period 2"
        if r < 3.5441:
            return "period 4"
        if r < 3.5699:
            return "period 2ⁿ"
        return "chaos"

    def update(self, f, dt):
        super().update(f, dt)
        k = self.k
        r = float(np.clip(_lerp(k["r_min"], k["r_max"], self.level), 0.5, 4.0))
        self.p = {"r": r}
        steps = int(min(50, max(1, k["steps"])))
        new = np.empty(steps)
        for i in range(steps):
            self.x = r * self.x * (1.0 - self.x)
            if not (0.0 < self.x < 1.0):
                self.x = 0.37
            new[i] = self.x
        self.hist = np.concatenate([self.hist[steps:], new])

    def layers(self, x):
        w = len(x)
        y = self.hist[-w:] if w <= len(self.hist) else np.r_[np.full(w - len(self.hist), np.nan), self.hist]
        r = self.p["r"]
        fixed = 1.0 - 1.0 / r if r > 1 else 0.0
        return [("hline", fixed, "aux"), ("points", y.copy(), "main")]

    def live(self):
        r = self.p["r"]
        fixed = 1 - 1 / r if r > 1 else 0.0
        return f"r={r:.3f}  ({self.regime(r)})  x*={fixed:.3f}"


class Spectrum(Panel):
    """The tuning shape (gain, threshold, curve, range) and smoothing apply to every bar."""

    key = "spectrum"
    title = "spectrum · DFT"
    formula = "X[k] = Σₙ x[n]·w[n]·e^(−2πikn/N)"
    identity = "w[n] = Hann window, log-spaced bars"
    default_drive = "rms"
    sound = "The classic Cava view: the input's own spectrum."
    yrange = (0.0, 1.0)

    def update(self, f, dt):
        super().update(f, dt)
        t = self.tune
        if t.own_smoothing:
            target = t.shape(f.raw_spectrum)
            prev = getattr(self, "_bars", None)
            if prev is None or len(prev) != len(target):
                prev = target.copy()
            attack = t.attack if t.attack is not None else self.global_attack
            decay = t.decay if t.decay is not None else self.global_decay
            self._bars = np.where(target > prev, prev + (target - prev) * attack, np.maximum(target, prev * decay))
            self.bars = self._bars
        else:
            self.bars = f.spectrum if t.is_default else t.shape(f.spectrum)
        self.n = f.fft_size
        self.rate = f.rate

    def layers(self, x):
        return [("bars", getattr(self, "bars", np.zeros(8)), "main")]

    def live(self):
        return f"N={getattr(self, 'n', 0)}  fs={getattr(self, 'rate', 0) / 1000:.1f} kHz  bars={len(getattr(self, 'bars', []))}"


# ---------------------------------------------------------------- custom panels


class ExprPanel(Panel):
    """A user-defined panel from the config file, for example:

    [panel.bell]
    title = bell · gaussian
    expr = A*exp(-(x/s)^2)
    params = A = 0.2 + 0.8*level, s = 0.1 + 0.5*mid
    domain = -1, 1
    yrange = -0.1, 1.1
    drive = bass

    In params, `level` is the panel's tuned level: its drive after any [tune.<key>] shaping.
    """

    drive_names = DRIVE_NAMES | {"t", "beat", "level"}

    def __init__(self, key: str, expr: str, params: dict[str, str], title: str | None = None,
                 domain=(-1.0, 1.0), yrange=(-1.1, 1.1), drive: str = "rms", sound: str | None = None):
        self.default_drive = drive
        super().__init__()
        self.key = key
        self.title = title or f"{key} · custom"
        self.expr_src = expr
        self.formula = "y = " + expr.replace("**", "^").replace("*", "·")
        self.identity = ", ".join(f"{k}={v.replace(' ', '')}" for k, v in params.items()).replace("**", "^").replace("*", "·")
        self.sound = sound or "User-defined formula from the config file."
        self.domain = tuple(domain)
        self.yrange = tuple(yrange)
        self.param_code = {k: compile_expr(v, self.drive_names) for k, v in params.items()}
        self.code = compile_expr(expr, set(params) | {"x", "t"})

    def update(self, f, dt):
        super().update(f, dt)
        env = dict(f.levels, t=f.t, beat=f.beat_period, level=self.level)
        self.p = {k: _eval(c, env) for k, c in self.param_code.items()}
        self.t = f.t

    def layers(self, x):
        env = dict(_FUNCS, **_CONSTS, **self.p, x=x, t=getattr(self, "t", 0.0))
        with np.errstate(all="ignore"):
            y = np.asarray(eval(self.code, {"__builtins__": {}}, env), dtype=np.float64)
        if y.shape != x.shape:
            y = np.broadcast_to(y, x.shape).copy()
        y[~np.isfinite(y)] = np.nan
        return [("hline", 0.0, "aux"), ("curve", y, "main")]


BUILTIN = [Exp808, DiveBomb, Reese, TanScreech, FMWobble, FourierSaw, Sinc, Logistic, Spectrum]
BY_KEY = {cls.key: cls for cls in BUILTIN}
DEFAULT_ORDER = [cls.key for cls in BUILTIN]


def make_panels(keys: list[str], custom: dict[str, ExprPanel] | None = None,
                tune: dict[str, dict[str, str]] | None = None,
                global_attack: float = 0.65, global_decay: float = 0.90) -> list[Panel]:
    custom = custom or {}
    tune = tune or {}
    known = set(BY_KEY) | set(custom)
    for k in tune:
        if k not in known:
            raise SystemExit(f"mathwave: [tune.{k}] does not match any panel. Panels: {', '.join(DEFAULT_ORDER + list(custom))}")
    out = []
    for k in keys:
        k = k.strip()
        if not k:
            continue
        if k in custom:
            p = custom[k]
        elif k in BY_KEY:
            p = BY_KEY[k]()
        else:
            raise SystemExit(f"mathwave: unknown panel '{k}'. Built-in panels: {', '.join(DEFAULT_ORDER)}")
        try:
            p.configure(tune.get(k, {}), global_attack, global_decay)
        except (ValueError, SyntaxError) as exc:
            raise SystemExit(f"mathwave: [tune.{k}]: {exc}")
        out.append(p)
    return out

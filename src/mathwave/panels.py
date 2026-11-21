"""The function panels. Each one is a formula whose parameters are driven by the music.

A panel exposes:
  formula   the symbolic formula that is plotted (printed in the panel footer)
  identity  an optional second line (an identity or the matching pitch law)
  drive     which analysis signal moves it (band, loudness or brightness)
  sound     what the same formula sounds like when you synthesize it
  layers()  numeric curves to draw for the current frame
  live()    the formula's parameters right now, printed under the formula
"""

from __future__ import annotations

import ast
import math

import numpy as np

from .dsp import Features

TAU = 2.0 * np.pi


def _breaks(y: np.ndarray, where: np.ndarray) -> np.ndarray:
    """Insert NaNs where a discontinuity happens so the renderer lifts the pen."""
    y = y.copy()
    jump = np.zeros_like(where, dtype=bool)
    jump[1:] = where[1:] != where[:-1]
    y[jump] = np.nan
    return y


class Panel:
    key = "base"
    title = "panel"
    formula = ""
    identity = ""
    drive = "rms"
    sound = ""
    domain = (0.0, 1.0)
    yrange = (-1.1, 1.1)

    def __init__(self):
        self.p: dict[str, float] = {}
        self.level = 0.0

    def update(self, f: Features, dt: float) -> None:
        self.level = f[self.drive]

    def layers(self, x: np.ndarray) -> list[tuple]:
        """Return a list of layers: ("curve", y, style) | ("hline", y0, style) |
        ("vline", x0, style) | ("bars", heights, style) | ("points", y, style). style is "main" or "aux"."""
        return []

    def live(self) -> str:
        return " ".join(f"{k}={v:.2f}" for k, v in self.p.items())


class Exp808(Panel):
    key = "808"
    title = "808 · exponential"
    formula = "y = A·e^(−kx)·sin(2π·f(x)·x)"
    identity = "f(x) = f₁ + (f₀ − f₁)·e^(−x/τ)"
    drive = "sub"
    sound = "An 808 kick: a sine whose pitch and volume both decay exponentially."

    def __init__(self):
        super().__init__()
        self.reveal = 1.0

    def update(self, f, dt):
        super().update(f, dt)
        hit = max(f["sub"], 0.7 * f["bass"])
        a = 0.08 + 0.92 * hit
        k = 2.0 + 3.0 * (1.0 - hit)
        # On every kick the curve is redrawn from x=0, like the note being played.
        self.reveal = min(1.0, f.since_onset / 0.35)
        self.p = {"A": a, "k": k, "f₀": 9.0 + 7.0 * hit, "f₁": 3.0, "τ": 0.22}

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
        return f"A={p['A']:.2f} k={p['k']:.2f} f₀={p['f₀']:.1f} f₁={p['f₁']:.0f} τ={p['τ']:.2f}"


class DiveBomb(Panel):
    key = "dive"
    title = "dive bomb · 1/x"
    formula = "y = a / (x − p)"
    identity = "pitch: f(t) = f₁ + c/(t + t₀)"
    drive = "bass"
    sound = "A dive bomb: pitch follows c/(t+t₀), falling fast and then leveling off."
    domain = (-1.0, 1.0)
    yrange = (-1.05, 1.05)

    def __init__(self):
        super().__init__()
        self.phase = 0.0

    def update(self, f, dt):
        super().update(f, dt)
        self.phase += dt * TAU / max(0.2, 2.0 * f.beat_period)
        self.p = {"a": 0.03 + 0.42 * self.level, "p": 0.75 * math.sin(self.phase)}

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
    drive = "lowmid"
    sound = "A Reese bass: two detuned waves beat against each other and growl."
    domain = (0.0, TAU)
    yrange = (-1.1, 1.1)

    def __init__(self):
        super().__init__()
        self.phase = 0.0

    def update(self, f, dt):
        super().update(f, dt)
        mid = f["mid"]
        self.phase += dt * (2.0 + 6.0 * self.level)
        a = 14.0
        self.p = {"a": a, "b": a + 0.6 + 5.0 * max(self.level, mid), "A": 0.1 + 0.9 * self.level}

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
    drive = "presence"
    sound = "A screech: tan(x) blows up at its asymptotes, and clipping turns that into buzz."
    yrange = (-1.1, 1.1)

    def __init__(self):
        super().__init__()
        self.phi = 0.0

    def update(self, f, dt):
        super().update(f, dt)
        lvl = max(self.level, 0.6 * f["highmid"])
        self.phi = (self.phi + dt * TAU * (0.15 + 1.5 * lvl)) % math.pi
        self.p = {"A": 0.12 + 0.88 * lvl, "ω": math.pi * (1.5 + 9.0 * lvl), "φ": self.phi}

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
    drive = "highmid"
    sound = "Dubstep wobble: frequency modulation, with the index I swept by an LFO."
    domain = (0.0, 1.0)

    def __init__(self):
        super().__init__()
        self.phi = 0.0
        self.bpm = 0.0

    def update(self, f, dt):
        super().update(f, dt)
        self.phi = (self.phi + dt * TAU / max(0.2, f.beat_period)) % TAU
        self.bpm = f.bpm
        self.p = {"I": 0.3 + 8.0 * max(self.level, 0.5 * f["mid"]), "ωc": TAU * 5.0, "ωm": TAU * 1.5}

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
    drive = "centroid"
    sound = "A saw lead built one harmonic at a time; brighter music adds harmonics."
    domain = (-2.0 * math.pi, 2.0 * math.pi)
    yrange = (-1.3, 1.3)

    def __init__(self):
        super().__init__()
        self.phase = 0.0
        self.hz = 0.0

    def update(self, f, dt):
        super().update(f, dt)
        self.phase = (self.phase + dt * 1.2) % TAU
        self.hz = f.centroid_hz
        self.p = {"N": float(1 + round(29 * self.level))}

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
    drive = "air"
    sound = "A click with no aliasing: sinc is a perfect impulse limited to a band."
    domain = (-1.0, 1.0)
    yrange = (-0.3, 1.05)

    def update(self, f, dt):
        super().update(f, dt)
        self.p = {"A": 0.15 + 0.85 * self.level, "ω": 2.0 + 16.0 * self.level}

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
    drive = "rms"
    sound = "Iterated, the logistic map goes from a steady tone to noise as r rises."
    yrange = (-0.02, 1.02)

    HISTORY = 600

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
        r = 2.8 + 1.2 * self.level
        self.p = {"r": r}
        steps = 4
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
        return [("hline", 1.0 - 1.0 / r, "aux"), ("points", y.copy(), "main")]

    def live(self):
        r = self.p["r"]
        return f"r={r:.3f}  ({self.regime(r)})  x*={1 - 1 / r:.3f}"


class Spectrum(Panel):
    key = "spectrum"
    title = "spectrum · DFT"
    formula = "X[k] = Σₙ x[n]·w[n]·e^(−2πikn/N)"
    identity = "w[n] = Hann window, log-spaced bars"
    drive = "rms"
    sound = "The classic Cava view: the input's own spectrum."
    yrange = (0.0, 1.0)

    def update(self, f, dt):
        super().update(f, dt)
        self.bars = f.spectrum
        self.n = f.fft_size
        self.rate = f.rate

    def layers(self, x):
        return [("bars", getattr(self, "bars", np.zeros(8)), "main")]

    def live(self):
        return f"N={getattr(self, 'n', 0)}  fs={getattr(self, 'rate', 0) / 1000:.1f} kHz  bars={len(getattr(self, 'bars', []))}"


# ---------------------------------------------------------------- custom panels

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


class ExprPanel(Panel):
    """A user-defined panel from the config file, for example:

    [panel.bell]
    title = bell · gaussian
    expr = A*exp(-(x/s)^2)
    params = A = 0.2 + 0.8*bass, s = 0.1 + 0.5*mid
    domain = -1, 1
    yrange = -0.1, 1.1
    """

    drive_names = {"sub", "bass", "lowmid", "mid", "highmid", "presence", "air", "rms", "centroid", "t", "beat"}

    def __init__(self, key: str, expr: str, params: dict[str, str], title: str | None = None,
                 domain=(-1.0, 1.0), yrange=(-1.1, 1.1), drive: str = "rms", sound: str | None = None):
        super().__init__()
        self.key = key
        self.title = title or f"{key} · custom"
        self.expr_src = expr
        self.formula = "y = " + expr.replace("**", "^").replace("*", "·")
        self.identity = ", ".join(f"{k}={v.replace(' ', '')}" for k, v in params.items()).replace("**", "^").replace("*", "·")
        self.drive = drive
        self.sound = sound or "User-defined formula from the config file."
        self.domain = tuple(domain)
        self.yrange = tuple(yrange)
        self.param_code = {k: compile_expr(v, self.drive_names) for k, v in params.items()}
        self.code = compile_expr(expr, set(params) | {"x", "t"})

    def update(self, f, dt):
        super().update(f, dt)
        env = dict(_FUNCS, **_CONSTS, **f.levels, t=f.t, beat=f.beat_period)
        self.p = {k: float(eval(c, {"__builtins__": {}}, env)) for k, c in self.param_code.items()}
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


def make_panels(keys: list[str], custom: dict[str, ExprPanel] | None = None) -> list[Panel]:
    custom = custom or {}
    out = []
    for k in keys:
        k = k.strip()
        if not k:
            continue
        if k in custom:
            out.append(custom[k])
        elif k in BY_KEY:
            out.append(BY_KEY[k]())
        else:
            raise SystemExit(f"mathwave: unknown panel '{k}'. Built-in panels: {', '.join(DEFAULT_ORDER)}")
    return out

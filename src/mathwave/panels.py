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

from .dsp import Features, Follower, Smoothing, drive_names

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
DRIVE_NAMES = drive_names()


def _num(node) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _num(node.operand)
        return None if v is None else -v
    return None


def band_name(kind: str, lo: float, hi: float) -> str:
    fmt = lambda v: f"{v:g}".replace(".", "p")  # noqa: E731
    return f"{kind}_{fmt(lo)}_{fmt(hi)}"


class _InlineBands(ast.NodeTransformer):
    """Rewrite band(lo, hi) and peak(lo, hi) into names the analyzer will provide."""

    def __init__(self, src: str):
        self.src = src
        self.found: dict[str, tuple[float, float, str]] = {}

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in ("band", "peak"):
            if len(node.args) != 2 or node.keywords:
                raise ValueError(f"'{self.src}': {node.func.id}() takes two numbers, e.g. {node.func.id}(40, 90)")
            lo, hi = _num(node.args[0]), _num(node.args[1])
            if lo is None or hi is None:
                raise ValueError(f"'{self.src}': {node.func.id}() needs plain numbers, e.g. {node.func.id}(40, 90)")
            if not (0 < lo < hi <= 24000):
                raise ValueError(f"'{self.src}': {node.func.id}({lo:g}, {hi:g}) needs 0 < lo < hi <= 24000 Hz")
            mode = "peak" if node.func.id == "peak" else "mean"
            name = band_name(node.func.id, lo, hi)
            self.found[name] = (lo, hi, mode)
            return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)
        return node


def compile_expr(src: str, names: set[str], inline: dict | None = None):
    """Compile a math expression after checking it only uses safe syntax and names.

    band(lo, hi) / peak(lo, hi) are allowed when `inline` is given; the bands they need
    are added to it as name -> (lo, hi, mode)."""
    tree = ast.parse(src.replace("^", "**"), mode="eval")
    if inline is not None:
        tr = _InlineBands(src)
        tree = ast.fix_missing_locations(tr.visit(tree))
        inline.update(tr.found)
        names = set(names) | set(tr.found)
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
    "drive": "what it listens to: a band, a feature, band(lo, hi), or e.g. max(sub, 0.7*bass)",
    "gain": "multiply the level before shaping (default 1)",
    "threshold": "ignore everything below this level, 0..1 (default 0)",
    "curve": "exponent: 1 linear, 2 punchier, 0.5 more sensitive (default 1)",
    "min": "level the panel rests at, 0..1 (default 0)",
    "max": "level the panel peaks at, 0..1 (default 1); min > max inverts",
    "attack": "this panel's rise speed, 0..1, higher is snappier (default: global)",
    "decay": "this panel's fall speed, 0..1, higher is slower (default: global)",
    "attack_ms": "time to rise 90% of the way, in ms (frame-rate independent)",
    "decay_ms": "time to fall 90% of the way, in ms (frame-rate independent)",
    "hold_ms": "hold each peak this long before falling, in ms",
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
    attack_ms: float | None = None
    decay_ms: float | None = None
    hold_ms: float | None = None

    @property
    def own_smoothing(self) -> bool:
        return any(v is not None for v in (self.attack, self.decay, self.attack_ms, self.decay_ms, self.hold_ms))

    def smoothing(self, base: Smoothing) -> Smoothing:
        """This panel's smoothing: its own keys over the global [smoothing] values.
        Setting a per-frame factor here also overrides a global *_ms value."""
        attack_ms = self.attack_ms if self.attack_ms is not None else (None if self.attack is not None else base.attack_ms)
        decay_ms = self.decay_ms if self.decay_ms is not None else (None if self.decay is not None else base.decay_ms)
        return Smoothing(
            attack=self.attack if self.attack is not None else base.attack,
            decay=self.decay if self.decay is not None else base.decay,
            attack_ms=attack_ms,
            decay_ms=decay_ms,
            hold_ms=self.hold_ms if self.hold_ms is not None else base.hold_ms,
        )

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
        for k in ("attack_ms", "decay_ms", "hold_ms"):
            v = getattr(self, k)
            if v is not None:
                parts.append(f"{k}={v:g}")
        return " ".join(parts)


# Shaping keys that may also be expressions, with the range their value is held to.
SHAPE_KEYS = {"gain": (1e-3, 100.0), "threshold": (0.0, 0.99), "curve": (0.05, 20.0), "min": (0.0, 1.0), "max": (0.0, 1.0)}
SHAPE_ATTR = {"min": "lo", "max": "hi"}


def _maybe_float(val: str) -> float | None:
    try:
        return float(val)
    except ValueError:
        return None


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
    # name -> (lo, hi, step): the slider range tune mode (k) offers for each knob
    RANGES: dict[str, tuple[float, float, float]] = {}

    def __init__(self, names: set[str] | None = None):
        self.p: dict[str, float] = {}
        self.level = 0.0
        self.k = {name: v for name, (v, _) in self.KNOBS.items()}
        self.k_expr: dict[str, object] = {}
        self.k_src: dict[str, str] = {}
        self.tune = Tuning()
        self.names = set(names) if names is not None else set(DRIVE_NAMES)
        self.inline: dict[str, tuple[float, float, str]] = {}
        self.base_smoothing = Smoothing()
        self._follow: Follower | None = None
        self.section: dict[str, str] = {}
        self.t_expr: dict[str, object] = {}
        self.t_src: dict[str, str] = {}
        self.set_drive(self.default_drive)

    # -- tuning

    def set_drive(self, src: str) -> None:
        self._drive_code = compile_expr(src, self.names, self.inline)
        self.drive = src.strip()

    @property
    def global_attack(self) -> float:  # kept for older callers
        return self.base_smoothing.attack

    @property
    def global_decay(self) -> float:
        return self.base_smoothing.decay

    def configure(self, section: dict[str, str], global_attack: float | Smoothing = 0.65, global_decay: float = 0.90) -> None:
        """Apply a [tune.<key>] section. Raises ValueError with a readable message.
        The second argument is the global Smoothing (or, for older callers, attack)."""
        if isinstance(global_attack, Smoothing):
            self.base_smoothing = global_attack
        else:
            self.base_smoothing = Smoothing(attack=global_attack, decay=global_decay)
        t = Tuning()
        t_expr: dict[str, object] = {}
        t_src: dict[str, str] = {}
        for raw_key, val in section.items():
            key = raw_key.strip().lower()
            val = val.strip()
            if key == "drive":
                self.set_drive(val)
            elif key in SHAPE_KEYS and _maybe_float(val) is None:
                # gain = 0.5 + punch/5: an expression of drives and sliders, evaluated each frame
                t_expr[key] = compile_expr(val, self.names, self.inline)
                t_src[key] = val
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
            elif key in ("attack_ms", "decay_ms", "hold_ms"):
                setattr(t, key, _float(key, val, 0.0, 60000.0))
            elif key in self.k:
                try:
                    v = float(val)
                except ValueError:
                    # A knob can be an expression of drives, e.g. a = 8 + 16*pitch
                    self.k_expr[key] = compile_expr(val, self.names | {"level"}, self.inline)
                    self.k_src[key] = val
                    continue
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
        self.t_expr, self.t_src = t_expr, t_src
        self._follow = Follower(t.smoothing(self.base_smoothing)) if t.own_smoothing else None
        self.section = {k.strip().lower(): v.strip() for k, v in section.items()}

    def retune(self, section: dict[str, str]) -> None:
        """Replace the whole [tune.<key>] section live (tune mode). Unlike configure(),
        knobs and drive start from their defaults first, and the smoothing state carries
        over so the panel does not jump."""
        old = self._follow
        self.k = {name: v for name, (v, _) in self.KNOBS.items()}
        self.k_expr, self.k_src = {}, {}
        self.set_drive(self.default_drive)
        self.configure(section, self.base_smoothing)
        if old is not None and self._follow is not None:
            old.s = self._follow.s
            self._follow = old

    def drive_level(self, f: Features, dt: float = 1 / 60) -> float:
        """The panel's tuned 0..1 level for this frame."""
        own = self._follow is not None
        levels = f.raw if own else f.levels
        if self.t_expr:
            env = _with_defaults(f.levels, self.inline)
            for key, code in self.t_expr.items():
                v = _eval(code, env)
                if math.isfinite(v):
                    lo, hi = SHAPE_KEYS[key]
                    setattr(self.tune, SHAPE_ATTR.get(key, key), min(hi, max(lo, v)))
        x = float(self.tune.shape(_eval(self._drive_code, _with_defaults(levels, self.inline))))
        return float(self._follow(x, dt)) if own else x

    def refresh_knobs(self, f: Features) -> None:
        """Evaluate knobs that are expressions of drives (smoothed levels)."""
        if not self.k_expr:
            return
        env = _with_defaults(dict(f.levels, level=self.level), self.inline)
        for name, code in self.k_expr.items():
            v = _eval(code, env)
            if not math.isfinite(v):
                continue
            if name in self.INT_KNOBS:
                v = float(max(1, int(round(v))))
            self.k[name] = v

    def tuning_summary(self) -> str:
        parts = []
        if self.drive != self.default_drive:
            parts.append(f"drive={self.drive}")
        s = " ".join(w for w in self.tune.summary().split()
                     if w.split("=")[0] not in self.t_src and not (w.startswith("range=") and ({"min", "max"} & set(self.t_src))))
        if s:
            parts.append(s)
        parts += [f"{k}={v.replace(' ', '')}" for k, v in self.t_src.items()]
        changed = [f"{n}={self.k_src[n].replace(' ', '')}" if n in self.k_src else f"{n}={v:g}"
                   for n, v in self.k.items() if n in self.k_src or v != self.KNOBS[n][0]]
        return " ".join(parts + changed)

    # -- drawing

    def update(self, f: Features, dt: float) -> None:
        self.level = self.drive_level(f, dt)
        self.refresh_knobs(f)

    def layers(self, x: np.ndarray) -> list[tuple]:
        """Return a list of layers: ("curve", y, style) | ("hline", y0, style) |
        ("vline", x0, style) | ("bars", heights, style) | ("points", y, style). style is "main" or "aux"."""
        return []

    def live(self) -> str:
        return " ".join(f"{k}={v:.2f}" for k, v in self.p.items())


def _with_defaults(levels: dict, inline: dict) -> dict:
    """Inline bands read 0 until the analyzer has produced them."""
    if not inline:
        return levels
    missing = {k: 0.0 for k in inline if k not in levels}
    return {**levels, **missing} if missing else levels


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

    RANGES = {
        "amp_min": (0.0, 1.0, 0.01), "f0_min": (1.0, 30.0, 0.5), "f0_max": (1.0, 40.0, 0.5),
        "f1": (0.5, 15.0, 0.25), "tau": (0.02, 1.0, 0.01), "k_min": (0.2, 10.0, 0.1),
        "k_max": (0.2, 12.0, 0.1), "reveal": (0.0, 1.5, 0.05),
    }

    def __init__(self, names=None):
        super().__init__(names)
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

    RANGES = {"a_min": (0.005, 0.5, 0.005), "a_max": (0.01, 1.0, 0.01), "sweep": (0.0, 1.0, 0.05),
              "beats": (0.25, 8.0, 0.25)}

    def __init__(self, names=None):
        super().__init__(names)
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

    RANGES = {"a": (2.0, 40.0, 0.5), "detune_min": (0.0, 10.0, 0.1), "detune_max": (0.0, 20.0, 0.1),
              "amp_min": (0.0, 1.0, 0.01), "drift": (0.0, 20.0, 0.5)}

    def __init__(self, names=None):
        super().__init__(names)
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

    RANGES = {"amp_min": (0.0, 1.0, 0.01), "freq_min": (0.5, 20.0, 0.25), "freq_max": (0.5, 30.0, 0.25),
              "scroll": (0.0, 8.0, 0.1)}

    def __init__(self, names=None):
        super().__init__(names)
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

    RANGES = {"index_min": (0.0, 10.0, 0.1), "index_max": (0.0, 20.0, 0.1), "carrier": (1.0, 15.0, 0.5),
              "mod": (0.25, 8.0, 0.25), "beats": (0.25, 8.0, 0.25)}

    def __init__(self, names=None):
        super().__init__(names)
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

    RANGES = {"n_min": (1.0, 20.0, 1.0), "n_max": (1.0, 80.0, 1.0), "speed": (0.0, 5.0, 0.1)}

    def __init__(self, names=None):
        super().__init__(names)
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
    RANGES = {"amp_min": (0.0, 1.0, 0.01), "width_min": (0.5, 20.0, 0.5), "width_max": (1.0, 40.0, 0.5)}

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

    RANGES = {"r_min": (0.5, 4.0, 0.01), "r_max": (0.5, 4.0, 0.01), "steps": (1.0, 50.0, 1.0)}

    def __init__(self, names=None):
        super().__init__(names)
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
        if self._follow is not None:
            self.bars = self._follow(t.shape(f.raw_spectrum), dt)
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

    knobs = height = 0.85 (0..1), width = 0.3 (0.05..1) declares the panel's own knobs:
    names its params can use, set from [tune.<key>] and shown as sliders in tune mode.
    """

    def __init__(self, key: str, expr: str, params: dict[str, str], title: str | None = None,
                 domain=(-1.0, 1.0), yrange=(-1.1, 1.1), drive: str = "rms", sound: str | None = None,
                 names: set[str] | None = None, knobs: dict[str, tuple[float, float, float, float]] | None = None):
        self.default_drive = drive
        knobs = knobs or {}
        self.KNOBS = {n: (d, f"custom knob, {lo:g} to {hi:g}") for n, (d, lo, hi, _) in knobs.items()}
        self.RANGES = {n: (lo, hi, st) for n, (_, lo, hi, st) in knobs.items()}
        self.INT_KNOBS = set()
        super().__init__(names)
        self.key = key
        self.title = title or f"{key} · custom"
        self.expr_src = expr
        self.formula = "y = " + expr.replace("**", "^").replace("*", "·")
        self.identity = ", ".join(f"{k}={v.replace(' ', '')}" for k, v in params.items()).replace("**", "^").replace("*", "·")
        self.sound = sound or "User-defined formula from the config file."
        self.domain = tuple(domain)
        self.yrange = tuple(yrange)
        pnames = self.names | {"t", "beat", "level"} | set(knobs)
        self.param_src = dict(params)
        self.param_code = {k: compile_expr(v, pnames, self.inline) for k, v in params.items()}
        self.code = compile_expr(expr, set(params) | {"x", "t"})

    def update(self, f, dt):
        super().update(f, dt)
        env = _with_defaults(dict(f.levels, t=f.t, beat=f.beat_period, level=self.level, **self.k), self.inline)
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


class Controls(Panel):
    """Your [slider.*] values as a panel. Add `controls` to panels to put it in the grid."""

    key = "controls"
    title = "controls · sliders"
    formula = "s = your [slider.*] values"
    identity = "↑↓ pick · ←→ move · d default"
    default_drive = "0"
    sound = "No formula: these are your hands on the dials. Any drive, knob or param can read them."
    yrange = (0.0, 1.0)

    def __init__(self, names=None):
        super().__init__(names)
        self.bank = None  # set by the app: a controls.SliderBank

    def update(self, f, dt):
        name = self.bank.current() if self.bank is not None else None
        self.level = self.bank.norm(name) if name else 0.0

    def live(self):
        if self.bank is None or not self.bank.specs:
            return "no sliders: add a [slider.NAME] section"
        return "  ".join(f"{n}={self.bank.values[n]:g}" for n in self.bank.names)


BUILTIN = [Exp808, DiveBomb, Reese, TanScreech, FMWobble, FourierSaw, Sinc, Logistic, Spectrum]
BY_KEY = {cls.key: cls for cls in BUILTIN + [Controls]}
DEFAULT_ORDER = [cls.key for cls in BUILTIN]


def make_panels(keys: list[str], custom: dict[str, ExprPanel] | None = None,
                tune: dict[str, dict[str, str]] | None = None,
                global_attack: float = 0.65, global_decay: float = 0.90,
                names: set[str] | None = None, smoothing: Smoothing | None = None) -> list[Panel]:
    """Build the panels in display order and apply their [tune.*] sections.

    names: every drive name the analyzer provides (built-ins plus [bands]).
    smoothing: the global [smoothing] settings, the default for per-panel smoothing."""
    custom = custom or {}
    tune = tune or {}
    base = smoothing or Smoothing(attack=global_attack, decay=global_decay)
    known = set(BY_KEY) | set(custom)
    for k in tune:
        if k not in known or k == "controls":
            raise SystemExit(f"mathwave: [tune.{k}] does not match any panel. Panels: {', '.join(DEFAULT_ORDER + list(custom))}")
    out = []
    for k in keys:
        k = k.strip()
        if not k:
            continue
        if k in custom:
            p = custom[k]
        elif k in BY_KEY:
            p = BY_KEY[k](names)
        else:
            raise SystemExit(f"mathwave: unknown panel '{k}'. Built-in panels: {', '.join(DEFAULT_ORDER + ['controls'])}")
        try:
            p.configure(tune.get(k, {}), base)
        except (ValueError, SyntaxError) as exc:
            raise SystemExit(f"mathwave: [tune.{k}]: {exc}")
        out.append(p)
    return out


def inline_bands(panels: list[Panel]) -> dict[str, tuple[float, float, str]]:
    """Every band(lo, hi) / peak(lo, hi) the panels' expressions need."""
    out: dict[str, tuple[float, float, str]] = {}
    for p in panels:
        out.update(p.inline)
    return out

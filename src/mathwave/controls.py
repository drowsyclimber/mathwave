"""Manual controls: user sliders, live panel tuning, config writing and input parsing.

Two kinds of control share one look:

  sliders   [slider.NAME] sections in the config. Each one is a value in its own
            range (0-10 by default) that any drive, knob or custom-panel param can
            read by name. They show on the `controls` panel and in the full-screen
            controls view (c). Positions are remembered per profile in
            $XDG_STATE_HOME/mathwave/sliders.json.

  tune mode k on a focused panel lists everything its [tune.<panel>] section can
            set as sliders: the drive, the level shaping, smoothing and every
            formula knob. Changes apply live; w writes them into the config file.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field

from .panels import Panel

# ---------------------------------------------------------------- numbers


def decimals(step: float) -> int:
    if step <= 0 or step >= 1:
        return 0
    return min(6, max(0, -int(math.floor(math.log10(step) + 1e-9))))


def fmt(v: float, step: float) -> str:
    """A value as it should be written to the config: no float noise, no trailing zeros."""
    d = decimals(step)
    s = f"{round(v, d):.{d}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def snap(v: float, lo: float, hi: float, step: float) -> float:
    a, b = min(lo, hi), max(lo, hi)
    v = min(b, max(a, v))
    if step > 0:
        v = a + round((v - a) / step) * step
        v = min(b, max(a, v))
    return round(v, decimals(step) + 2)


_RANGE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(?:\.\.|to|,|-)\s*(-?\d+(?:\.\d+)?)\s*$")


def parse_span(text: str, what: str) -> tuple[float, float]:
    """'0-10', '-5-5', '0..1', '0, 1' or '0 to 1' -> (lo, hi)."""
    m = _RANGE.match(text)
    if not m:
        raise ValueError(f"{what} = {text}: use LO-HI, for example 0-10")
    lo, hi = float(m.group(1)), float(m.group(2))
    if not lo < hi:
        raise ValueError(f"{what} = {text}: LO must be below HI")
    return lo, hi


# ---------------------------------------------------------------- user sliders


@dataclass
class SliderSpec:
    name: str
    lo: float = 0.0
    hi: float = 10.0
    step: float = 0.1
    default: float = 0.0
    label: str = ""


def parse_slider(name: str, sec: dict[str, str], reserved: set[str]) -> SliderSpec:
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
        raise ValueError(f"slider name '{name}': use lowercase letters, digits and _")
    if name in reserved or name.startswith(("band_", "peak_")):
        raise ValueError(f"slider name '{name}' is already a drive or function name; pick another")
    allowed = {"range", "default", "step", "label"}
    for k in sec:
        if k not in allowed:
            raise ValueError(f"[slider.{name}]: unknown key '{k}' (use range, default, step, label)")
    lo, hi = parse_span(sec.get("range", "0-10"), "range")
    try:
        step = float(sec["step"]) if sec.get("step") else (hi - lo) / 100.0
        default = float(sec["default"]) if sec.get("default") else lo
    except ValueError as exc:
        raise ValueError(f"[slider.{name}]: {exc}")
    if not 0 < step <= hi - lo:
        raise ValueError(f"[slider.{name}]: step must be above 0 and at most the range width")
    if not lo <= default <= hi:
        raise ValueError(f"[slider.{name}]: default {default:g} is outside {lo:g}-{hi:g}")
    return SliderSpec(name, lo, hi, step, snap(default, lo, hi, step), sec.get("label", "").strip())


def state_path() -> str:
    base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return os.path.join(base, "mathwave", "sliders.json")


def load_state(path: str | None = None) -> dict:
    try:
        with open(path or state_path()) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


class SliderBank:
    """The live values of every [slider.*], with selection and persistence."""

    def __init__(self, specs: dict[str, SliderSpec] | None = None, profile: str | None = None,
                 state_file: str | None = None):
        self.specs = dict(specs or {})
        self.values = {n: s.default for n, s in self.specs.items()}
        self.selected = 0
        self.profile = profile or "default"
        self.state_file = state_file  # None: never touch the disk (headless runs, tests)
        self.dirty = False
        if state_file:
            saved = load_state(state_file).get("profiles", {}).get(self.profile, {})
            for n, v in saved.items():
                if n in self.specs and isinstance(v, (int, float)):
                    self.set(n, float(v), mark=False)

    @property
    def names(self) -> list[str]:
        return list(self.specs)

    def current(self) -> str | None:
        names = self.names
        if not names:
            return None
        self.selected %= len(names)
        return names[self.selected]

    def select(self, delta: int) -> None:
        if self.specs:
            self.selected = (self.selected + delta) % len(self.specs)

    def set(self, name: str, v: float, mark: bool = True) -> None:
        s = self.specs[name]
        new = snap(v, s.lo, s.hi, s.step)
        if mark and new != self.values.get(name):
            self.dirty = True
        self.values[name] = new

    def nudge(self, name: str, steps: float) -> None:
        self.set(name, self.values[name] + steps * self.specs[name].step)

    def reset(self, name: str) -> None:
        self.set(name, self.specs[name].default)

    def norm(self, name: str) -> float:
        s = self.specs[name]
        return (self.values[name] - s.lo) / (s.hi - s.lo)

    def adopt(self, old: "SliderBank") -> None:
        """Keep the positions of sliders that still exist after a reload."""
        for n, v in old.values.items():
            if n in self.specs:
                self.set(n, v, mark=False)
        self.selected = min(old.selected, max(0, len(self.specs) - 1))

    def save(self) -> None:
        if not self.state_file or not self.dirty:
            return
        data = load_state(self.state_file)
        data.setdefault("profiles", {})[self.profile] = dict(self.values)
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.state_file), prefix=".sliders")
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
        os.replace(tmp, self.state_file)
        self.dirty = False


def slider_usage(panels: list[Panel], names: list[str]) -> dict[str, list[str]]:
    """Which panels read each slider: {'punch': ['808 gain', 'reese a']}."""
    out: dict[str, list[str]] = {n: [] for n in names}
    if not names:
        return out
    pat = {n: re.compile(rf"\b{re.escape(n)}\b") for n in names}
    for p in panels:
        sources = {"drive": p.drive, **getattr(p, "t_src", {}), **p.k_src}
        for k, src in getattr(p, "param_src", {}).items():
            sources[k] = src
        for key, src in sources.items():
            for n in names:
                if pat[n].search(src or ""):
                    out[n].append(f"{p.key} {key}")
    return out


# ---------------------------------------------------------------- tune mode


@dataclass
class Param:
    name: str
    lo: float
    hi: float
    step: float
    default: float | None  # None: falls back to the global [smoothing] value
    desc: str
    kind: str = "common"     # drive | common | knob


COMMON_PARAMS = [
    Param("gain", 0.05, 5.0, 0.05, 1.0, "multiply the level before shaping"),
    Param("threshold", 0.0, 0.95, 0.01, 0.0, "ignore everything below this level"),
    Param("curve", 0.2, 5.0, 0.05, 1.0, "exponent: above 1 punchier, below 1 more sensitive"),
    Param("min", 0.0, 1.0, 0.01, 0.0, "level the panel rests at"),
    Param("max", 0.0, 1.0, 0.01, 1.0, "level the panel peaks at (below min inverts)"),
    Param("attack_ms", 0.0, 500.0, 5.0, None, "time to rise 90% of the way"),
    Param("decay_ms", 0.0, 3000.0, 10.0, None, "time to fall 90% of the way"),
    Param("hold_ms", 0.0, 1000.0, 10.0, None, "hold each peak this long before it falls"),
]
COMMON_DEFAULT_TEXT = {"gain": "1", "threshold": "0", "curve": "1", "min": "0", "max": "1"}


def factor_to_ms(factor: float, fps: float, rising: bool) -> float:
    """Per-frame attack/decay factor -> time to cover 90% of a change at this frame rate."""
    keep = 1.0 - factor if rising else factor
    if keep <= 0:
        return 0.0
    if keep >= 1:
        return 60000.0
    return math.log(0.1) / math.log(keep) / fps * 1000.0


def _num(text) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    x, y = _num(a), _num(b)
    if x is not None and y is not None:
        return abs(x - y) < 1e-9
    return str(a).replace(" ", "") == str(b).replace(" ", "")


class TuneSession:
    """Live editing of one panel's [tune.<panel>] section."""

    def __init__(self, panel: Panel, drive_choices: list[str], fps: float = 60.0,
                 base_section: dict[str, str] | None = None, profile: str | None = None,
                 path: str | None = None):
        self.panel = panel
        self.fps = fps
        self.profile = profile
        self.path = path
        self.base_section = dict(base_section or {})
        self.section = dict(getattr(panel, "section", {}) or {})
        self.loaded = dict(self.section)
        self.selected = 0
        self.offset = 0
        choices = [panel.default_drive] + [c for c in drive_choices if c != panel.default_drive]
        if panel.drive not in choices:
            choices.insert(1, panel.drive)
        self.drive_choices = choices
        self.params: list[Param] = [Param("drive", 0, len(choices) - 1, 1, None, "what the panel listens to", "drive")]
        self.params += [p for p in COMMON_PARAMS]
        ranges = getattr(panel, "RANGES", {})
        for name, (default, desc) in panel.KNOBS.items():
            lo, hi, step = ranges.get(name, _auto_range(default))
            self.params.append(Param(name, lo, hi, step, default, desc, "knob"))

    # -- reading

    @property
    def dirty(self) -> bool:
        return self.section != self.loaded

    def current(self) -> Param:
        self.selected %= len(self.params)
        return self.params[self.selected]

    def global_ms(self, name: str) -> float:
        base = self.panel.base_smoothing
        if name == "attack_ms":
            if "attack" in self.section and _num(self.section["attack"]) is not None:
                return factor_to_ms(float(self.section["attack"]), self.fps, True)
            return base.attack_ms if base.attack_ms is not None else factor_to_ms(base.attack, self.fps, True)
        if name == "decay_ms":
            if "decay" in self.section and _num(self.section["decay"]) is not None:
                return factor_to_ms(float(self.section["decay"]), self.fps, False)
            return base.decay_ms if base.decay_ms is not None else factor_to_ms(base.decay, self.fps, False)
        return base.hold_ms

    def value(self, p: Param) -> float:
        """The parameter's effective numeric value right now."""
        if p.kind == "drive":
            return float(self.drive_choices.index(self.panel.drive)) if self.panel.drive in self.drive_choices else 0.0
        if p.kind == "knob":
            return float(self.panel.k.get(p.name, p.default or 0.0))
        v = _num(self.section.get(p.name))
        if v is not None:
            return v
        if p.name in self.panel.t_src:  # an expression: its value this frame
            t = self.panel.tune
            return float(getattr(t, {"min": "lo", "max": "hi"}.get(p.name, p.name)))
        return self.global_ms(p.name) if p.default is None else p.default

    def is_set(self, p: Param) -> bool:
        if p.kind == "drive":
            return self.panel.drive != self.panel.default_drive
        return p.name in self.section

    def expression(self, p: Param) -> str | None:
        """A knob that is an expression (a = 8 + 16*pitch) shows its source instead of a slider."""
        if p.kind in ("knob", "common") and p.name in self.section and _num(self.section[p.name]) is None:
            return self.section[p.name]
        return None

    def norm(self, p: Param) -> float:
        v = self.value(p)
        return 0.0 if p.hi == p.lo else max(0.0, min(1.0, (v - p.lo) / (p.hi - p.lo)))

    def text(self, p: Param) -> str:
        if p.kind == "drive":
            return self.panel.drive
        expr = self.expression(p)
        if expr is not None:
            return expr
        v = self.value(p)
        s = fmt(v, p.step)
        if p.default is None and not self.is_set(p):
            return f"global {s}"
        return s

    # -- editing

    def _apply(self) -> None:
        self.panel.retune(self.section)

    def set(self, p: Param, v: float) -> None:
        if p.kind == "drive":
            i = int(round(min(p.hi, max(p.lo, v))))
            choice = self.drive_choices[i]
            if choice == self.panel.default_drive:
                self.section.pop("drive", None)
            else:
                self.section["drive"] = choice
        else:
            v = snap(v, p.lo, p.hi, p.step)
            self.section[p.name] = fmt(v, p.step)
            if p.name == "attack_ms":
                self.section.pop("attack", None)
            elif p.name == "decay_ms":
                self.section.pop("decay", None)
        self._apply()

    def nudge(self, steps: float, p: Param | None = None) -> None:
        p = p or self.current()
        if p.kind == "drive":
            n = len(self.drive_choices)
            self.set(p, (self.value(p) + (1 if steps > 0 else -1)) % n)
            return
        self.set(p, self.value(p) + steps * p.step)

    def set_fraction(self, p: Param, frac: float) -> None:
        self.set(p, p.lo + max(0.0, min(1.0, frac)) * (p.hi - p.lo))

    def reset(self, p: Param | None = None) -> None:
        p = p or self.current()
        self.section.pop(p.name, None)
        if p.name == "attack_ms":
            self.section.pop("attack", None)
        elif p.name == "decay_ms":
            self.section.pop("decay", None)
        self._apply()

    def undo(self) -> None:
        self.section = dict(self.loaded)
        self._apply()

    def select(self, delta: int) -> None:
        self.selected = (self.selected + delta) % len(self.params)

    # -- writing

    @property
    def section_name(self) -> str:
        return f"tune.{self.panel.key}" + (f"@{self.profile}" if self.profile else "")

    def default_text(self, key: str) -> str | None:
        if key == "drive":
            return self.panel.default_drive
        if key in COMMON_DEFAULT_TEXT:
            return COMMON_DEFAULT_TEXT[key]
        if key in self.panel.KNOBS:
            p = next((q for q in self.params if q.name == key), None)
            return fmt(self.panel.KNOBS[key][0], p.step if p else 0.001)
        return None

    def updates(self) -> dict[str, str | None]:
        """Key -> new text (None removes it) for the section this session writes to.

        Without a profile, keys back at their built-in default are removed. With one,
        the profile section only keeps what differs from the plain [tune.<panel>]."""
        keys = list(dict.fromkeys([p.name for p in self.params] + list(self.section) + list(self.loaded)
                                  + list(self.base_section)))
        out: dict[str, str | None] = {}
        for key in keys:
            live = self.section.get(key)
            if self.profile:
                ref = self.base_section.get(key)
                if _same(live, ref):
                    out[key] = None
                elif live is None:
                    out[key] = self.default_text(key)
                else:
                    out[key] = live
            else:
                out[key] = None if live is None or _same(live, self.default_text(key)) else live
        return out

    def write(self) -> str:
        """Write the changes into the config file. Returns the path."""
        path = self.path
        if not path:
            raise ValueError("no config file path")
        update_section(path, self.section_name, self.updates())
        self.loaded = dict(self.section)
        if not self.profile:
            self.base_section = dict(self.section)
        return path


def _auto_range(default: float) -> tuple[float, float, float]:
    hi = max(1.0, abs(default) * 2.0)
    lo = 0.0 if default >= 0 else -hi
    return lo, hi, (hi - lo) / 100.0


# ---------------------------------------------------------------- config writer

_HEADER = re.compile(r"^\s*\[([^\]]+)\]\s*(?:[#;].*)?$")
_KEYLINE = re.compile(r"^(\s*)([A-Za-z0-9_]+)(\s*[=:]\s*)(.*?)(\s+[#;].*)?$")


def update_section(path: str, section: str, updates: dict[str, str | None]) -> None:
    """Set or remove keys in one [section] of an INI file, keeping comments and order.

    Existing keys are edited in place (a trailing comment stays), new keys go after the
    section's last key, and a missing section is added at the end of the file. The
    previous file is kept as PATH.bak."""
    try:
        with open(path) as fh:
            lines = fh.read().splitlines()
        existed = True
    except FileNotFoundError:
        lines = ["# mathwave config. Tune mode (k, then w) writes [tune.<panel>] sections here."]
        existed = False

    start = None
    for i, line in enumerate(lines):
        m = _HEADER.match(line)
        if m and m.group(1).strip() == section:
            start = i
            break
    todo = {k: v for k, v in updates.items()}
    if start is not None:
        end = len(lines)
        for j in range(start + 1, len(lines)):
            if _HEADER.match(lines[j]):
                end = j
                break
        body = lines[start + 1:end]
        kept: list[str] = []
        last_key = -1
        for line in body:
            m = _KEYLINE.match(line)
            if m and not line.lstrip().startswith(("#", ";")) and m.group(2).lower() in todo:
                key = m.group(2).lower()
                new = todo.pop(key)
                if new is None:
                    continue
                kept.append(f"{m.group(1)}{m.group(2)}{m.group(3)}{new}{m.group(5) or ''}")
                last_key = len(kept) - 1
            else:
                kept.append(line)
                if m and not line.lstrip().startswith(("#", ";")):
                    last_key = len(kept) - 1
        add = [f"{k} = {v}" for k, v in todo.items() if v is not None]
        at = last_key + 1 if last_key >= 0 else 0
        kept[at:at] = add
        lines = lines[:start + 1] + kept + lines[end:]
    else:
        add = [f"{k} = {v}" for k, v in todo.items() if v is not None]
        if add:
            while lines and not lines[-1].strip():
                lines.pop()
            lines += ["", f"[{section}]"] + add

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if existed:
        with open(path) as src, open(path + ".bak", "w") as bak:
            bak.write(src.read())
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), prefix=".mathwave")
    with os.fdopen(fd, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------- input


@dataclass
class Mouse:
    button: int      # 0 left, 1 middle, 2 right, 64 wheel up, 65 wheel down
    col: int         # 0-based
    row: int
    pressed: bool    # False on release
    motion: bool = False


@dataclass
class Hit:
    kind: str        # "slider" | "tune"
    name: str
    row: int
    c0: int          # first column of the bar
    c1: int          # last column of the bar
    span: tuple = field(default=(0, 0))  # whole clickable row: (first col, last col)


_CSI = {
    "A": "up", "B": "down", "C": "right", "D": "left", "H": "home", "F": "end", "Z": "backtab",
}
_TILDE = {"1": "home", "4": "end", "5": "pgup", "6": "pgdn", "7": "home", "8": "end"}
_MOUSE = re.compile(r"\x1b\[<(\d+);(\d+);(\d+)([Mm])")
_SEQ = re.compile(r"\x1b(?:\[(\d+(?:;\d+)*)?([A-Za-z~])|O([A-Za-z]))")


def parse_input(data: str) -> list:
    """Split raw terminal input into keys ('q', 'up', 'S-left', 'esc', ...) and Mouse events."""
    out: list = []
    i = 0
    while i < len(data):
        ch = data[i]
        if ch != "\x1b":
            out.append("enter" if ch in "\r\n" else "backspace" if ch in "\x7f\x08" else "tab" if ch == "\t" else ch)
            i += 1
            continue
        m = _MOUSE.match(data, i)
        if m:
            b, x, y, kind = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
            out.append(Mouse(b & ~(4 | 8 | 16 | 32), x - 1, y - 1, kind == "M", bool(b & 32)))
            i = m.end()
            continue
        m = _SEQ.match(data, i)
        if m:
            params, final, ss3 = m.group(1), m.group(2), m.group(3)
            if ss3:
                key = _CSI.get(ss3)
            elif final == "~":
                key = _TILDE.get((params or "").split(";")[0])
            else:
                key = _CSI.get(final)
            if key and params and ";" in params:
                mod = int(params.split(";")[1]) - 1
                if mod & 1:
                    key = "S-" + key
            if key:
                out.append(key)
            i = m.end()
            continue
        out.append("esc")
        i += 1
    return out

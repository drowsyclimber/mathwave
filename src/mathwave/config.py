"""Config file loading. Format and search path follow Cava's conventions.

Profiles: any section can have an @name suffix, for example [tune.808@phonk] or
[smoothing@ambient]. When a profile is active, its sections are layered key by key
over the plain sections. Choose one with `profile = phonk` under [general],
`--profile phonk`, or cycle through them with Shift+P while running.
"""

from __future__ import annotations

import configparser
import os
import re
from dataclasses import dataclass, field

from .dsp import BANDS, FEATURES, VALID_FFT, AnalysisSettings, BandSpec, Smoothing, drive_names
from .controls import SliderSpec, parse_slider
from .panels import _CONSTS, _FUNCS, DEFAULT_ORDER, ExprPanel


def default_path() -> str:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "mathwave", "config")


@dataclass
class Config:
    framerate: int = 60
    sensitivity: float = 1.0
    panels: list[str] = field(default_factory=lambda: list(DEFAULT_ORDER))
    pages: list[list[str]] = field(default_factory=list)  # empty: [panels, the other group]
    layout: str = "auto"
    formulas: int = 2
    bars: int = 48
    method: str = "auto"
    source: str = "auto"
    rate: int = 44100
    channels: int = 2
    fifo: str = "/tmp/mpd.fifo"
    theme: str = "phonk"
    gradient: list[str] | None = None
    color_mode: str = "truecolor"
    # [smoothing]
    attack: float = 0.65
    decay: float = 0.90
    attack_ms: float | None = None
    decay_ms: float | None = None
    hold_ms: float = 0.0
    noise_floor_db: float = -75.0
    dynamic_range_db: float = 40.0
    fall_db_per_s: float = 6.0
    # [analysis]
    fft_size: int = 2048
    multires: bool = False
    low_fft: int = 8192
    high_fft: int = 1024
    split_hz: float = 250.0
    tilt: float = 0.0
    weighting: str = "none"
    pitch_range: tuple = (30.0, 300.0)
    # [bands]
    bands: dict[str, BandSpec] = field(default_factory=dict)
    # [onsets]
    kick_bands: tuple = ((30.0, 150.0),)
    snare_bands: tuple = ((180.0, 400.0), (2000.0, 5000.0))
    hat_bands: tuple = ((7000.0, 16000.0),)
    onset_sensitivity: float = 1.5
    refractory_ms: float = 120.0
    envelope_ms: float = 200.0
    # [tempo]
    bpm: float | None = None
    bpm_multiplier: float = 1.0
    bpm_range: tuple = (70.0, 180.0)
    # manual controls
    sliders: dict[str, SliderSpec] = field(default_factory=dict)
    mouse: bool = True
    # panels and profiles
    custom: dict[str, ExprPanel] = field(default_factory=dict)
    tune: dict[str, dict[str, str]] = field(default_factory=dict)
    tune_base: dict[str, dict[str, str]] = field(default_factory=dict)  # [tune.*] without any profile
    profile: str | None = None
    profiles: list[str] = field(default_factory=list)
    path: str | None = None

    @property
    def names(self) -> set[str]:
        """Every drive name available to expressions: bands, features and sliders."""
        return drive_names({k: v for k, v in self.bands.items() if k not in BANDS}) | set(self.sliders)

    @property
    def smoothing(self) -> Smoothing:
        return Smoothing(self.attack, self.decay, self.attack_ms, self.decay_ms, self.hold_ms)

    def analysis(self) -> AnalysisSettings:
        return AnalysisSettings(
            rate=self.rate, fft_size=self.fft_size, n_bars=self.bars, sensitivity=self.sensitivity,
            noise_floor_db=self.noise_floor_db, dynamic_range_db=self.dynamic_range_db,
            fall_db_per_s=self.fall_db_per_s, smoothing=self.smoothing, bands=dict(self.bands),
            tilt_db_per_oct=self.tilt, weighting=self.weighting, multires=self.multires,
            low_fft=self.low_fft, high_fft=self.high_fft, split_hz=self.split_hz,
            kick_bands=self.kick_bands, snare_bands=self.snare_bands, hat_bands=self.hat_bands,
            onset_sensitivity=self.onset_sensitivity, refractory_ms=self.refractory_ms,
            envelope_ms=self.envelope_ms, bpm=self.bpm, bpm_multiplier=self.bpm_multiplier,
            bpm_range=self.bpm_range, pitch_range=self.pitch_range,
        )


# ---------------------------------------------------------------- value parsers


def _pairs(text: str) -> tuple[float, float]:
    a, b = (float(v) for v in text.split(","))
    return a, b


def check_fft_size(n: int) -> int:
    if n not in VALID_FFT:
        raise SystemExit(f"mathwave: fft_size must be one of {', '.join(map(str, VALID_FFT))} (got {n})")
    return n


_RANGE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)")


def parse_range(text: str, what: str) -> tuple[float, float]:
    m = _RANGE.match(text)
    if not m or m.end() != len(text.rstrip()):
        raise ValueError(f"{what} = {text}: expected LO-HI in Hz, for example 40-90")
    lo, hi = float(m.group(1)), float(m.group(2))
    if not (0 < lo < hi <= 24000):
        raise ValueError(f"{what} = {text}: needs 0 < LO < HI <= 24000")
    return lo, hi


def parse_ranges(text: str, what: str) -> tuple:
    """'180-400 + 2000-5000' -> ((180, 400), (2000, 5000))"""
    return tuple(parse_range(part, what) for part in text.split("+"))


_RESERVED = set(_FUNCS) | set(_CONSTS) | set(FEATURES) | {"x", "t", "beat", "level", "band", "peak"}


def parse_band(name: str, text: str) -> BandSpec:
    """'40-90 peak range=25 floor=-70 fall=4 fft=8192 norm=shared' (all after LO-HI optional;
    LO-HI may be left out when overriding a built-in band)."""
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
        raise ValueError(f"band name '{name}': use lowercase letters, digits and _")
    if name in _RESERVED or name.startswith(("band_", "peak_")):
        raise ValueError(f"band name '{name}' is reserved; pick another name")
    m = _RANGE.match(text)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        rest = text[m.end():].split()
    elif name in BANDS:
        lo, hi = BANDS[name]
        rest = text.split()
    else:
        raise ValueError(f"{name} = {text}: a new band needs a range, for example {name} = 40-90")
    if not (0 < lo < hi <= 24000):
        raise ValueError(f"{name} = {text}: needs 0 < LO < HI <= 24000")
    spec = BandSpec(lo, hi)
    for tok in rest:
        if tok in ("mean", "peak"):
            spec.mode = tok
            continue
        if "=" not in tok:
            raise ValueError(f"{name}: unknown option '{tok}' (use mean, peak, range=, floor=, fall=, fft=, norm=)")
        k, v = tok.split("=", 1)
        try:
            if k == "range":
                spec.range_db = float(v)
                if spec.range_db < 3:
                    raise ValueError
            elif k == "floor":
                spec.floor_db = float(v)
            elif k == "fall":
                spec.fall_db = float(v)
            elif k == "fft":
                spec.fft = check_fft_size(int(v))
            elif k == "norm":
                if v not in ("own", "shared"):
                    raise ValueError
                spec.norm = v
            else:
                raise KeyError(k)
        except KeyError:
            raise ValueError(f"{name}: unknown option '{k}' (use range=, floor=, fall=, fft=, norm=)")
        except (ValueError, SystemExit):
            raise ValueError(f"{name}: bad value {tok}")
    return spec


def _bool(text: str, what: str) -> bool:
    t = text.strip().lower()
    if t in ("1", "on", "yes", "true"):
        return True
    if t in ("0", "off", "no", "false"):
        return False
    raise ValueError(f"{what} = {text}: use on or off")


def _split_top(text: str) -> list[str]:
    """Split on commas that are not inside parentheses, so max(sub, bass) stays whole."""
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _params(text: str) -> dict[str, str]:
    out = {}
    for part in _split_top(text):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
        elif part.strip():
            raise ValueError(f"params entry '{part.strip()}' needs the form name = expression")
    return out


# ---------------------------------------------------------------- loading


def _apply_profile(cp: configparser.ConfigParser, profile: str | None) -> tuple[str | None, list[str]]:
    profiles = sorted({s.split("@", 1)[1].strip() for s in cp.sections() if "@" in s})
    if profile is None:
        profile = cp.get("general", "profile", fallback="").strip() or None
    if profile in ("", "default", "none"):
        profile = None
    if profile is not None and profile not in profiles:
        known = ", ".join(profiles) if profiles else "none defined"
        raise SystemExit(f"mathwave: config error: unknown profile '{profile}' (profiles: {known})")
    for sec in [s for s in cp.sections() if "@" in s]:
        base, prof = (x.strip() for x in sec.split("@", 1))
        if prof == profile:
            if not cp.has_section(base):
                cp.add_section(base)
            for k, v in cp[sec].items():
                cp[base][k] = v
        cp.remove_section(sec)
    return profile, profiles


def load(path: str | None = None, profile: str | None = None) -> Config:
    cfg = Config()
    path = path or default_path()
    if not os.path.exists(path):
        if profile not in (None, "", "default", "none"):
            raise SystemExit(f"mathwave: config error: unknown profile '{profile}' (no config file)")
        return cfg
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None)
    try:
        cp.read(path)
    except configparser.Error as exc:
        # Duplicate keys or sections, a key outside any section, and so on.
        msg = " ".join(str(exc).split())
        raise SystemExit(f"mathwave: config error: {msg}")
    cfg.path = path
    cfg.tune_base = {sec.split(".", 1)[1].strip(): {k: v for k, v in cp[sec].items()}
                     for sec in cp.sections() if sec.startswith("tune.") and "@" not in sec}
    cfg.profile, cfg.profiles = _apply_profile(cp, profile)
    sec = lambda name: cp[name] if cp.has_section(name) else {}  # noqa: E731
    try:
        _read_values(cfg, sec("general"), sec("input"), sec("color"), sec("smoothing"))
        _read_analysis(cfg, sec("analysis"), sec("bands"), sec("onsets"), sec("tempo"))
        _read_sliders(cfg, cp)
    except ValueError as exc:
        raise SystemExit(f"mathwave: config error in {path}: {exc}")
    _read_panels(cfg, cp, path)
    return cfg


def _read_values(cfg: Config, g, i, c, s) -> None:
    cfg.framerate = int(g.get("framerate", cfg.framerate))
    cfg.sensitivity = float(g.get("sensitivity", cfg.sensitivity))
    if g.get("panels"):
        cfg.panels = [p.strip() for p in g["panels"].split(",") if p.strip()]
    if g.get("pages"):
        cfg.pages = [[p.strip() for p in page.split(",") if p.strip()] for page in g["pages"].split("|")]
        cfg.pages = [page for page in cfg.pages if page]
    cfg.layout = g.get("layout", cfg.layout) or "auto"
    cfg.formulas = int(g.get("formulas", cfg.formulas))
    cfg.bars = int(g.get("bars", cfg.bars))
    if g.get("mouse"):
        cfg.mouse = _bool(g["mouse"], "mouse")

    cfg.method = i.get("method", cfg.method) or "auto"
    cfg.source = i.get("source", cfg.source) or "auto"
    cfg.rate = int(i.get("rate", cfg.rate))
    cfg.channels = int(i.get("channels", cfg.channels))
    cfg.fifo = i.get("fifo", cfg.fifo)

    cfg.theme = c.get("theme", cfg.theme) or "phonk"
    if c.get("gradient"):
        cfg.gradient = [v.strip() for v in c["gradient"].split(",") if v.strip()]
    cfg.color_mode = c.get("mode", cfg.color_mode) or "truecolor"

    cfg.attack = float(s.get("attack", cfg.attack))
    cfg.decay = float(s.get("decay", cfg.decay))
    for key in ("attack_ms", "decay_ms"):
        if s.get(key):
            v = float(s[key])
            if v < 0:
                raise ValueError(f"{key} = {s[key]}: must be >= 0")
            setattr(cfg, key, v)
    cfg.hold_ms = float(s.get("hold_ms", cfg.hold_ms))
    cfg.noise_floor_db = float(s.get("noise_floor_db", cfg.noise_floor_db))
    cfg.dynamic_range_db = float(s.get("dynamic_range_db", cfg.dynamic_range_db))
    cfg.fall_db_per_s = float(s.get("fall_db_per_s", cfg.fall_db_per_s))
    cfg.fft_size = check_fft_size(int(s.get("fft_size", cfg.fft_size)))  # 0.2 location, still read


def _read_analysis(cfg: Config, a, b, o, t) -> None:
    if a.get("fft_size"):
        cfg.fft_size = check_fft_size(int(a["fft_size"]))
    if a.get("multires"):
        cfg.multires = _bool(a["multires"], "multires")
    if a.get("low_fft"):
        cfg.low_fft = check_fft_size(int(a["low_fft"]))
    if a.get("high_fft"):
        cfg.high_fft = check_fft_size(int(a["high_fft"]))
    cfg.split_hz = float(a.get("split_hz", cfg.split_hz))
    cfg.tilt = float(a.get("tilt", cfg.tilt))
    w = a.get("weighting", cfg.weighting).strip().lower()
    if w not in ("none", "a"):
        raise ValueError(f"weighting = {w}: use none or a")
    cfg.weighting = w
    if a.get("pitch_range"):
        cfg.pitch_range = parse_range(a["pitch_range"], "pitch_range")

    for name, text in b.items():
        cfg.bands[name] = parse_band(name, text)

    if o.get("kick"):
        cfg.kick_bands = parse_ranges(o["kick"], "kick")
    if o.get("snare"):
        cfg.snare_bands = parse_ranges(o["snare"], "snare")
    if o.get("hat"):
        cfg.hat_bands = parse_ranges(o["hat"], "hat")
    cfg.onset_sensitivity = float(o.get("sensitivity", cfg.onset_sensitivity))
    cfg.refractory_ms = float(o.get("refractory_ms", cfg.refractory_ms))
    cfg.envelope_ms = float(o.get("envelope_ms", cfg.envelope_ms))

    bpm = t.get("bpm", "auto").strip().lower()
    if bpm != "auto":
        cfg.bpm = float(bpm)
        if not (20 <= cfg.bpm <= 400):
            raise ValueError(f"bpm = {bpm}: use auto or a tempo between 20 and 400")
    cfg.bpm_multiplier = float(t.get("multiplier", cfg.bpm_multiplier))
    if not (0.125 <= cfg.bpm_multiplier <= 8):
        raise ValueError(f"multiplier = {cfg.bpm_multiplier:g}: use a value between 0.125 and 8")
    if t.get("range"):
        cfg.bpm_range = parse_range(t["range"], "range")


def _read_sliders(cfg: Config, cp: configparser.ConfigParser) -> None:
    reserved = _RESERVED | set(BANDS) | set(cfg.bands) | {"controls"}
    for sec in cp.sections():
        if sec.startswith("slider."):
            name = sec.split(".", 1)[1].strip()
            cfg.sliders[name] = parse_slider(name, dict(cp[sec]), reserved)


_KNOB = re.compile(r"^\s*([a-z_][a-z0-9_]*)\s*=\s*(-?\d+(?:\.\d+)?)\s*"
                   r"(?:\(\s*(-?\d+(?:\.\d+)?)\s*(?:\.\.|to|-)\s*(-?\d+(?:\.\d+)?)\s*(?:,\s*(\d+(?:\.\d+)?)\s*)?\))?\s*$")


def _knobs(text: str, taken: set[str]) -> dict[str, tuple[float, float, float, float]]:
    """'height = 0.85 (0..1), width = 0.3 (0.05..1, 0.01)' -> {name: (default, lo, hi, step)}."""
    out = {}
    for part in _split_top(text):
        if not part.strip():
            continue
        m = _KNOB.match(part)
        if not m and _KNOB.match(part.lower()):
            raise ValueError(f"knobs entry '{part.strip()}': knob names are lowercase (they are keys in [tune.<panel>])")
        if not m:
            raise ValueError(f"knobs entry '{part.strip()}': use name = default (lo..hi), e.g. width = 0.3 (0.05..1)")
        name, default = m.group(1), float(m.group(2))
        if name in taken or name in _RESERVED:
            raise ValueError(f"knob name '{name}' is already a drive, param or function name")
        if m.group(3) is not None:
            lo, hi = float(m.group(3)), float(m.group(4))
        else:
            hi = max(1.0, abs(default) * 2.0)
            lo = 0.0 if default >= 0 else -hi
        if not lo < hi or not lo <= default <= hi:
            raise ValueError(f"knob '{name}': needs lo < hi and the default inside the range")
        step = float(m.group(5)) if m.group(5) else (hi - lo) / 100.0
        out[name] = (default, lo, hi, step)
    return out


def _read_panels(cfg: Config, cp: configparser.ConfigParser, path: str) -> None:
    names = cfg.names
    for name in cp.sections():
        if name.startswith("tune."):
            key = name.split(".", 1)[1].strip()
            cfg.tune[key] = {k: v for k, v in cp[name].items()}
            continue
        if not name.startswith("panel."):
            continue
        key = name.split(".", 1)[1]
        sec = cp[name]
        if key == "controls":
            raise SystemExit(f"mathwave: [{name}] in {path}: 'controls' is the built-in slider panel; pick another key")
        try:
            params = _params(sec.get("params", ""))
            knobs = _knobs(sec.get("knobs", ""), names | set(params))
            cfg.custom[key] = ExprPanel(
                key,
                sec["expr"],
                params,
                title=sec.get("title"),
                domain=_pairs(sec.get("domain", "-1, 1")),
                yrange=_pairs(sec.get("yrange", "-1.1, 1.1")),
                drive=sec.get("drive", "rms"),
                sound=sec.get("sound"),
                names=names,
                knobs=knobs,
            )
        except KeyError as exc:
            raise SystemExit(f"mathwave: [{name}] in {path} is missing {exc}")
        except (ValueError, SyntaxError) as exc:
            raise SystemExit(f"mathwave: bad [{name}] in {path}: {exc}")

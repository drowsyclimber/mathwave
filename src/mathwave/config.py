"""Config file loading. Format and search path follow Cava's conventions."""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass, field

from .panels import DEFAULT_ORDER, ExprPanel


def default_path() -> str:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "mathwave", "config")


@dataclass
class Config:
    framerate: int = 60
    sensitivity: float = 1.0
    panels: list[str] = field(default_factory=lambda: list(DEFAULT_ORDER))
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
    attack: float = 0.65
    decay: float = 0.90
    noise_floor_db: float = -75.0
    dynamic_range_db: float = 40.0
    fft_size: int = 2048
    custom: dict[str, ExprPanel] = field(default_factory=dict)
    tune: dict[str, dict[str, str]] = field(default_factory=dict)
    path: str | None = None


def _pairs(text: str) -> tuple[float, float]:
    a, b = (float(v) for v in text.split(","))
    return a, b


def check_fft_size(n: int) -> int:
    if n not in (1024, 2048, 4096, 8192):
        raise SystemExit(f"mathwave: fft_size must be 1024, 2048, 4096 or 8192 (got {n})")
    return n


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


def load(path: str | None = None) -> Config:
    cfg = Config()
    path = path or default_path()
    if not os.path.exists(path):
        return cfg
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None)
    cp.read(path)
    cfg.path = path
    g = cp["general"] if cp.has_section("general") else {}
    i = cp["input"] if cp.has_section("input") else {}
    c = cp["color"] if cp.has_section("color") else {}
    s = cp["smoothing"] if cp.has_section("smoothing") else {}

    cfg.framerate = int(g.get("framerate", cfg.framerate))
    cfg.sensitivity = float(g.get("sensitivity", cfg.sensitivity))
    if g.get("panels"):
        cfg.panels = [p.strip() for p in g["panels"].split(",") if p.strip()]
    cfg.layout = g.get("layout", cfg.layout) or "auto"
    cfg.formulas = int(g.get("formulas", cfg.formulas))
    cfg.bars = int(g.get("bars", cfg.bars))

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
    cfg.noise_floor_db = float(s.get("noise_floor_db", cfg.noise_floor_db))
    cfg.dynamic_range_db = float(s.get("dynamic_range_db", cfg.dynamic_range_db))
    cfg.fft_size = check_fft_size(int(s.get("fft_size", cfg.fft_size)))

    for name in cp.sections():
        if name.startswith("tune."):
            key = name.split(".", 1)[1].strip()
            cfg.tune[key] = {k: v for k, v in cp[name].items()}
            continue
        if not name.startswith("panel."):
            continue
        key = name.split(".", 1)[1]
        sec = cp[name]
        try:
            cfg.custom[key] = ExprPanel(
                key,
                sec["expr"],
                _params(sec.get("params", "")),
                title=sec.get("title"),
                domain=_pairs(sec.get("domain", "-1, 1")),
                yrange=_pairs(sec.get("yrange", "-1.1, 1.1")),
                drive=sec.get("drive", "rms"),
                sound=sec.get("sound"),
            )
        except KeyError as exc:
            raise SystemExit(f"mathwave: [{name}] in {path} is missing {exc}")
        except (ValueError, SyntaxError) as exc:
            raise SystemExit(f"mathwave: bad [{name}] in {path}: {exc}")
    return cfg

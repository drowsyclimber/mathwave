"""Command line entry point and the real-time loop."""

from __future__ import annotations

import argparse
import os
import re
import select
import signal
import sys
import termios
import time
import tty

import numpy as np

from . import __version__, synth
from .audio import DemoSource, build_source
from .config import Config, check_fft_size, default_path, load
from .controls import Hit, Mouse, SliderBank, TuneSession, parse_input, slider_usage, state_path
from .dsp import BANDS, FEATURE_INFO, Analyzer, Features
from .blackhole import HOLE
from .cern import CERN
from .panels import BUILTIN, COMMON_KEYS, DEFAULT_ORDER, GROUPS, Controls, expand, inline_bands, make_panels
from .render import DYNAMIC_THEMES, THEMES, Renderer, drive_label

THEME_ORDER = list(THEMES) + DYNAMIC_THEMES
KEY_HELP = [
    ("1-9, 0", "show that panel full screen (0 is panel 10); the same key again or g returns"),
    (". ,", "focus the next or previous panel (reaches panel 11 and beyond)"),
    ("n, N", "next or previous page: the math panels, the CERN page and the black hole page"),
    ("k", "tune mode: the focused panel's settings as sliders (opens panel 1 from the grid)"),
    ("c", "full-screen controls: your [slider.*] values"),
    ("↑ ↓, Tab", "pick a slider (in tune mode, the controls view or a controls panel)"),
    ("← →", "move it; Shift+← → or [ ] for ten steps, Home/End for the ends"),
    ("d, u, w", "slider back to default; undo all tuning; write tuning to the config"),
    ("mouse", "click or drag a slider bar, scroll the wheel over one"),
    ("m", "toggle the live meter view of every band and feature"),
    ("r", "reload the config file"),
    ("P", "cycle through config profiles (Shift+P)"),
    ("b", "tap tempo (tap at least twice); B goes back to the configured or detected tempo"),
    ("f", "cycle formula detail: full, formula plus values, hidden"),
    ("t", "next color theme"),
    ("+ -", "raise or lower gain"),
    ("p, space", "pause"),
    ("Esc", "close tune mode, controls or meters; quit from the grid"),
    ("q", "quit"),
]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="mathwave",
        description="Cava-style terminal visualizer: music drives a grid of math functions, "
        "and each panel prints its formula with live parameter values.",
    )
    ap.add_argument("-p", "--config", help=f"config file (default {default_path()})")
    ap.add_argument("-i", "--input", dest="method", choices=["auto", "pulse", "pipewire", "fifo", "stdin", "demo"],
                    help="audio input (default auto: parec, then pw-record)")
    ap.add_argument("-s", "--source", help="PulseAudio device or PipeWire target (default: monitor of the default output)")
    ap.add_argument("--fifo", help="raw s16le FIFO path for --input fifo (for example MPD's fifo output)")
    ap.add_argument("--demo", action="store_true", help="play the built-in formula track instead of capturing audio")
    ap.add_argument("--audible", action="store_true", help="with --demo, also play the track through your speakers")
    ap.add_argument("--profile", help="activate a config profile (sections named like [tune.808@NAME])")
    ap.add_argument("--panels", help=f"comma list of panels (default: {','.join(DEFAULT_ORDER)})")
    ap.add_argument("--layout", help="auto or COLSxROWS, for example 3x3")
    ap.add_argument("--theme", choices=THEME_ORDER, help="color theme")
    ap.add_argument("--fps", type=int, help="frames per second (default 60)")
    ap.add_argument("--sensitivity", type=float, help="gain applied after normalization (default 1.0)")
    ap.add_argument("--fft-size", type=int, help="FFT size: 1024, 2048 (default), 4096, 8192 or 16384")
    ap.add_argument("--bpm", help="fixed tempo, or auto to detect it from kicks")
    ap.add_argument("--focus", type=int, help="start with panel N full screen")
    ap.add_argument("--page", type=int, help="start on page N (n flips pages; by default 2 is CERN, 3 is the black hole)")
    ap.add_argument("--meters", action="store_true", help="start in the live meter view")
    ap.add_argument("--controls", action="store_true", help="start in the full-screen controls view")
    ap.add_argument("--tune", action="store_true", help="start in tune mode (on --focus N, or panel 1)")
    ap.add_argument("--set", action="append", metavar="NAME=VALUE", help="set a slider at start (repeatable)")
    ap.add_argument("--no-mouse", action="store_true", help="do not capture the mouse")
    ap.add_argument("--keys-in", metavar="KEYS", help=argparse.SUPPRESS)  # --dump: keys to feed first
    ap.add_argument("--no-color", action="store_true", help="disable color")
    ap.add_argument("--formulas", action="store_true", help="print every panel's formula and exit")
    ap.add_argument("--list-panels", action="store_true", help="list panel keys and exit")
    ap.add_argument("--list-knobs", nargs="?", const="all", metavar="PANEL",
                    help="list the [tune.<panel>] keys for every panel (or one) and exit")
    ap.add_argument("--list-features", action="store_true",
                    help="list every drive name (bands and features) usable in expressions and exit")
    ap.add_argument("--keys", action="store_true", help="list the keyboard controls and exit")
    ap.add_argument("--render-wav", metavar="PATH", help="write the demo track to a stereo WAV file and exit")
    ap.add_argument("--seconds", type=float, default=synth.BEAT * synth.SECTION_BEATS * synth.N_LAYERS,
                    help="length for --render-wav (default: one full loop)")
    ap.add_argument("--dump", type=int, metavar="FRAMES",
                    help="headless: simulate FRAMES frames of the demo and print the last one")
    ap.add_argument("--size", default="160x48", help="terminal size for --dump, COLSxROWS")
    ap.add_argument("--at", type=float, help="with --dump, end the simulation at this demo time in seconds")
    ap.add_argument("-v", "--version", action="version", version=f"mathwave {__version__}")
    return ap.parse_args(argv)


def apply_args(cfg: Config, a) -> Config:
    if a.method:
        cfg.method = a.method
    if a.demo:
        cfg.method = "demo"
    if a.source:
        cfg.source = a.source
    if a.fifo:
        cfg.fifo = a.fifo
    if a.panels:
        cfg.panels = [p.strip() for p in a.panels.split(",") if p.strip()]
        cfg.pages = []
    if a.layout:
        cfg.layout = a.layout
    if a.theme:
        cfg.theme = a.theme
        cfg.gradient = None
    if a.fps:
        cfg.framerate = max(1, a.fps)
    if a.sensitivity:
        cfg.sensitivity = a.sensitivity
    if a.fft_size:
        cfg.fft_size = check_fft_size(a.fft_size)
    if getattr(a, "bpm", None):
        if a.bpm.strip().lower() == "auto":
            cfg.bpm = None
        else:
            try:
                cfg.bpm = float(a.bpm)
            except ValueError:
                raise SystemExit(f"mathwave: --bpm {a.bpm}: use auto or a number")
    if a.no_color:
        cfg.color_mode = "none"
    if getattr(a, "no_mouse", False):
        cfg.mouse = False
    return cfg


def load_config(a, profile: str | None = None) -> Config:
    return apply_args(load(a.config, profile if profile is not None else getattr(a, "profile", None)), a)


def build_panels(cfg: Config):
    return make_panels(cfg.panels, cfg.custom, cfg.tune, names=cfg.names, smoothing=cfg.smoothing)


def page_list(cfg: Config) -> list[list[str]]:
    """The pages n flips through: [general] pages, or the panels line followed by the
    other built-in pages (math, cern, hole)."""
    if cfg.pages:
        return cfg.pages
    first = expand(cfg.panels)
    groups = [GROUPS[g] for g in ("math", "cern", "hole")]
    others = [g for g in groups if g != first]
    if first not in groups:  # your own panels line stands in for the math page
        others = others[1:]
    return [cfg.panels] + others


def build_pages(cfg: Config) -> tuple[Analyzer, list[list]]:
    """One analyzer for every page, with all their inline bands registered."""
    pages = [make_panels(keys, cfg.custom, cfg.tune, names=cfg.names, smoothing=cfg.smoothing)
             for keys in page_list(cfg)]
    an = Analyzer(settings=cfg.analysis())
    for panels in pages:
        for name, (lo, hi, mode) in inline_bands(panels).items():
            an.add_band(name, lo, hi, mode)
    return an, pages


def build(cfg: Config) -> tuple[Analyzer, list]:
    """The analysis pipeline and panels for a config, with inline band(lo, hi) registered."""
    panels = build_panels(cfg)
    an = Analyzer(settings=cfg.analysis())
    for name, (lo, hi, mode) in inline_bands(panels).items():
        an.add_band(name, lo, hi, mode)
    return an, panels


def print_knobs(cfg: Config, which: str) -> int:
    panels = [cls(cfg.names) for cls in BUILTIN + CERN + HOLE] + list(cfg.custom.values())
    if which != "all":
        panels = [p for p in panels if p.key == which]
        if not panels:
            print(f"mathwave: no panel '{which}'", file=sys.stderr)
            return 2
    print("Keys every [tune.<panel>] section accepts:")
    for k, desc in COMMON_KEYS.items():
        print(f"  {k:10s} {desc}")
    print("  gain, threshold, curve, min, max and every knob can be a number or an expression")
    print("  of drives and sliders, e.g. gain = 0.5 + punch/5 or a = 8 + 16*pitch.")
    print("  Run mathwave --list-features for every drive name.")
    print()
    for p in panels:
        print(f"[tune.{p.key}]   {p.title}")
        print(f"  drive      default: {p.default_drive}")
        for name, (default, desc) in p.KNOBS.items():
            print(f"  {name:10s} default {default:g}: {desc}")
        if not p.KNOBS:
            print("  (no extra knobs; shape it with the common keys" + (", or use level in params)" if p.key in cfg.custom else ")"))
        print()
    return 0


def print_features(cfg: Config) -> int:
    an, panels = build(cfg)
    print("Bands (drive names; band(lo, hi) and peak(lo, hi) also work inline):")
    for name, spec in an.bands.items():
        origin = "built-in" if name in BANDS and name not in cfg.bands else (
            "inline" if name.startswith(("band_", "peak_")) else "[bands]")
        extra = []
        if spec.mode == "peak":
            extra.append("peak")
        if spec.norm == "shared":
            extra.append("shared norm")
        if spec.range_db is not None:
            extra.append(f"range {spec.range_db:g} dB")
        fft = an.band_fft(spec)
        if fft != an.n:
            extra.append(f"fft {fft}")
        print(f"  {name:12s} {spec.lo:g}–{spec.hi:g} Hz  ({origin}{', ' + ', '.join(extra) if extra else ''})")
    print()
    print("Features:")
    for name, desc in FEATURE_INFO.items():
        print(f"  {name:12s} {desc}")
    print()
    if cfg.sliders:
        print()
        print("Sliders (their own range, not 0..1; divide to scale, e.g. punch/10):")
        for name, spec in cfg.sliders.items():
            label = f"  {spec.label}" if spec.label else ""
            print(f"  {name:12s} {spec.lo:g} to {spec.hi:g}, default {spec.default:g}{label}")
    print()
    print("Also in custom panel params: t (seconds), beat (beat length in seconds), level (the panel's tuned drive),")
    print("and the panel's own knobs from its knobs = line.")
    return 0


def print_formulas(panels, f: Features | None = None) -> None:
    for i, p in enumerate(panels, 1):
        print(f"{i}. {p.title}")
        print(f"   {p.formula}")
        if p.identity:
            print(f"   {p.identity}")
        print(f"   driven by: {drive_label(p.drive, f)}")
        if p.tuning_summary():
            print(f"   tuned: {p.tuning_summary()}")
        print(f"   sounds like: {p.sound}")
        if getattr(p, "verdict", ""):
            print(f"   physics check: {p.verdict}. {p.truth}")
        print()


def _status(src_name: str, f: Features, cfg: Config, note: str | None, paused: bool,
            page: tuple[int, int] | None = None) -> str:
    tempo = f"BPM≈{f.bpm:.0f}" + ("" if f.tempo_source == "auto" else f" ({f.tempo_source})")
    parts = [f"input: {src_name}", tempo, f"gain {cfg.sensitivity:.2f}", f"theme {cfg.theme}"]
    if page and page[1] > 1:
        parts.append(f"page {page[0] + 1}/{page[1]}")
    if cfg.profile:
        parts.append(f"profile {cfg.profile}")
    if paused:
        parts.append("PAUSED")
    if note:
        parts.append(note)
    return " · ".join(parts)


def dump(cfg: Config, frames: int, size: str, focus: int | None = None, at: float | None = None,
         meters: bool = False, args=None, controls: bool = False, tune: bool = False, keys: str = "") -> int:
    """Headless: simulate the demo and print the last frame. `keys` are fed to the app
    first, written like the terminal sends them (for example a tune-mode session)."""
    cols, rows = (int(v) for v in size.lower().split("x"))
    app = App(cfg, args, focus, meters, controls, tune)
    for ev in parse_input(keys.encode().decode("unicode_escape") if keys else ""):
        app.key(ev, 0.0)
    src, _ = build_source("offline", rate=cfg.rate)
    dt = 1.0 / cfg.framerate
    start = 0.0 if at is None else max(0.0, at - frames * dt)
    for k in range(frames):
        t = start + (k + 1) * dt
        app.step(src.latest_stereo(app.an.need, t), t, dt)
    sys.stdout.write("\n".join(app.draw(cols, rows, "demo (offline)")) + "\n")
    return 0


class Terminal:
    def __init__(self, mouse: bool = True):
        self.mouse = mouse
        self.out = sys.stdout
        try:
            self.keys_fd = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            self.keys_fd = sys.stdin.fileno()
        self.old = None

    def __enter__(self):
        try:
            self.old = termios.tcgetattr(self.keys_fd)
            tty.setcbreak(self.keys_fd)
        except termios.error:
            self.old = None
        # SGR mouse reporting with drag (1002 + 1006). Shift+drag still selects text.
        self.out.write("\x1b[?1049h\x1b[?25l\x1b[2J" + ("\x1b[?1002h\x1b[?1006h" if self.mouse else ""))
        self.out.flush()
        return self

    def __exit__(self, *exc):
        if self.old is not None:
            termios.tcsetattr(self.keys_fd, termios.TCSADRAIN, self.old)
        self.out.write(("\x1b[?1006l\x1b[?1002l" if self.mouse else "") + "\x1b[0m\x1b[?25h\x1b[?1049l")
        self.out.flush()

    def keys(self) -> str:
        r, _, _ = select.select([self.keys_fd], [], [], 0)
        if not r:
            return ""
        try:
            return os.read(self.keys_fd, 4096).decode(errors="ignore")
        except BlockingIOError:
            return ""

    def size(self) -> tuple[int, int]:
        try:
            s = os.get_terminal_size(self.out.fileno())
            return s.columns, s.lines
        except OSError:
            return 120, 40

    def draw(self, lines: list[str]) -> None:
        self.out.write("\x1b[?2026h\x1b[H" + "\r\n".join(lines) + "\x1b[?2026l")
        self.out.flush()


def _error_note(exc: SystemExit) -> str:
    """'mathwave: config error in /path: detail; hint' -> 'config error: detail' for the status line."""
    msg = re.sub(r"^config error( in [^:]+)?: ", "", str(exc).removeprefix("mathwave: "))
    return "config error: " + msg.split(";")[0]


RESTART_ONLY = ("method", "source", "rate", "channels", "fifo")
ADJUST = {"left": -1, "right": 1, "S-left": -10, "S-right": 10, "[": -10, "]": 10, "pgdn": -10, "pgup": 10}


def drive_choices(cfg: Config) -> list[str]:
    """What tune mode's drive row cycles through: bands, features, then sliders scaled to 0..1."""
    out = list(BANDS) + [b for b in cfg.bands if b not in BANDS] + list(FEATURE_INFO)
    for name, spec in cfg.sliders.items():
        out.append(f"{name}/{spec.hi:g}" if spec.lo == 0 else f"({name} - {spec.lo:g})/{spec.hi - spec.lo:g}")
    return out


class App:
    """Everything the live view does, without the terminal: analysis, panels, sliders,
    tune mode, keys and mouse. run() drives it from a real terminal, --dump and the
    tests drive it headlessly."""

    def __init__(self, cfg: Config, args=None, focus: int | None = None, meters: bool = False,
                 controls: bool = False, tune: bool = False, persist: bool = False):
        self.args = args
        self.persist = persist
        self.page = max(0, (getattr(args, "page", None) or 1) - 1)
        self.focus, self.meters, self.controls = focus, meters, controls
        self.tune = None
        self.sessions: dict[str, TuneSession] = {}
        self.drag = None
        self.note: str | None = None
        self.paused = False
        self.f = Features()
        self.bank = None
        self.last_save = 0.0
        self._setup(cfg)
        self.theme_i = THEME_ORDER.index(cfg.theme) if cfg.theme in THEME_ORDER else 0
        for item in getattr(args, "set", None) or []:
            self.set_slider(item)
        if tune:
            self.open_tune()

    # -- setup

    def _setup(self, cfg: Config, old_bank: SliderBank | None = None) -> None:
        self.cfg = cfg
        self.an, self.pages = build_pages(cfg)
        self.page = min(getattr(self, "page", 0), len(self.pages) - 1)
        self.bank = SliderBank(cfg.sliders, cfg.profile, state_path() if self.persist else None)
        if old_bank is not None and old_bank.profile == self.bank.profile:
            self.bank.adopt(old_bank)
        for page in self.pages:
            for p in page:
                if isinstance(p, Controls):
                    p.bank = self.bank
        self.rend = Renderer(cfg.theme, cfg.gradient, cfg.color_mode)
        self.footer = cfg.formulas
        self.usage = slider_usage(self.all_panels, self.bank.names)
        self.sessions = {}
        if self.focus is not None and self.focus >= len(self.panels):
            self.focus = None

    @property
    def panels(self) -> list:
        return self.pages[self.page]

    @property
    def all_panels(self) -> list:
        seen, out = set(), []
        for page in self.pages:
            for p in page:
                if id(p) not in seen:
                    seen.add(id(p))
                    out.append(p)
        return out

    def flip(self, delta: int = 1) -> None:
        if len(self.pages) < 2:
            self.note = "only one page (set pages = ... | ... under [general])"
            return
        self.page = (self.page + delta) % len(self.pages)
        self.focus, self.tune = None, None
        self.meters = self.controls = False

    def set_slider(self, item: str) -> None:
        name, _, value = item.partition("=")
        name = name.strip()
        if name not in self.bank.specs:
            known = ", ".join(self.bank.names) or "none defined"
            raise SystemExit(f"mathwave: --set {item}: no slider '{name}' (sliders: {known})")
        try:
            self.bank.set(name, float(value), mark=False)
        except ValueError:
            raise SystemExit(f"mathwave: --set {item}: use NAME=NUMBER")

    def reload(self, profile: str | None = None) -> None:
        """r (profile None) or Shift+P. Everything rebuilds except the audio input."""
        old, was_tuning, dirty = self.cfg, self.tune is not None, any(s.dirty for s in self.sessions.values())
        try:
            cfg = load_config(self.args, profile if profile is not None else (old.profile or ""))
        except SystemExit as exc:
            self.note = _error_note(exc)
            return
        pending = [name for name in RESTART_ONLY if getattr(cfg, name) != getattr(old, name)]
        for name in RESTART_ONLY:
            setattr(cfg, name, getattr(old, name))
        self.bank.save()
        try:
            self._setup(cfg, self.bank)
        except SystemExit as exc:
            self.note = _error_note(exc)
            self._setup(old, self.bank)
            return
        self.tune = None
        if was_tuning:
            self.open_tune()
        what = f"profile {cfg.profile or 'default'}" if profile is not None else "config reloaded"
        extra = []
        if pending:
            extra.append(", ".join(pending) + " apply on restart")
        if dirty:
            extra.append("unsaved tuning discarded")
        self.note = what + (f" ({'; '.join(extra)})" if extra else "")

    # -- frames

    def step(self, samples, t: float, dt: float) -> Features:
        f = self.an.analyze(samples, t)
        if self.bank.values:
            f.levels.update(self.bank.values)
            f.raw.update(self.bank.values)
        for p in self.panels:
            p.update(f, dt)
        self.f = f
        return f

    def draw(self, cols: int, rows: int, src_name: str, hints: bool = True) -> list[str]:
        tune = self.tune if self.focus is not None and not self.meters and not self.controls else None
        return self.rend.render(self.panels, self.f, cols, rows, layout=self.cfg.layout, footer=self.footer,
                                focus=self.focus, status=_status(src_name, self.f, self.cfg, self.note, self.paused,
                                              (self.page, len(self.pages))),
                                hints=hints, meters=self.meters, bank=self.bank, tune=tune,
                                controls=self.controls, usage=self.usage)

    def autosave(self, now: float) -> None:
        if self.bank.dirty and now - self.last_save > 2.0:
            self.bank.save()
            self.last_save = now

    # -- tune mode

    def open_tune(self) -> None:
        if self.focus is None:
            self.focus = next((i for i, p in enumerate(self.panels) if not isinstance(p, Controls)), None)
            if self.focus is None:
                self.note = "no panel to tune"
                return
        p = self.panels[self.focus]
        if isinstance(p, Controls):
            self.note = "the controls panel has no tuning: ←→ moves its sliders"
            self.tune = None
            return
        s = self.sessions.get(p.key)
        if s is None or s.panel is not p:
            s = TuneSession(p, drive_choices(self.cfg), fps=self.cfg.framerate,
                            base_section=self.cfg.tune_base.get(p.key, {}), profile=self.cfg.profile,
                            path=self.cfg.path or default_path())
            self.sessions[p.key] = s
        self.tune = s
        self.meters = self.controls = False

    def write_tune(self) -> None:
        s = self.tune
        try:
            path = s.write()
        except (OSError, ValueError) as exc:
            self.note = f"could not write the config: {exc}"
            return
        self.cfg.path = path
        self.cfg.tune[s.panel.key] = dict(s.section)
        if not s.profile:
            self.cfg.tune_base[s.panel.key] = dict(s.section)
        self.note = f"wrote [{s.section_name}] to {path.replace(os.path.expanduser('~'), '~')}"

    def _tuned(self) -> None:
        self.usage = slider_usage(self.all_panels, self.bank.names)

    def slider_mode(self) -> bool:
        if self.controls:
            return True
        if self.tune is not None or self.meters:
            return False
        if self.focus is not None:
            return isinstance(self.panels[self.focus], Controls)
        return any(isinstance(p, Controls) for p in self.panels)

    def adjust(self, k: str) -> bool:
        """Arrow keys and friends. Returns True when the key was used."""
        if self.tune is not None and self.focus is not None and not self.meters and not self.controls:
            s = self.tune
            if k in ("up", "backtab"):
                s.select(-1)
            elif k in ("down", "tab"):
                s.select(1)
            elif k in ADJUST:
                s.nudge(ADJUST[k])
            elif k in ("home", "end"):
                p = s.current()
                s.set(p, p.lo if k == "home" else p.hi)
            elif k == "d":
                s.reset()
            elif k == "u":
                s.undo()
                self.note = "tuning back to the config file"
            elif k == "w":
                self.write_tune()
                return True
            else:
                return False
            self._tuned()
            return True
        if self.slider_mode():
            b = self.bank
            name = b.current()
            if name is None:
                return False
            spec = b.specs[name]
            if k in ("up", "backtab"):
                b.select(-1)
            elif k in ("down", "tab"):
                b.select(1)
            elif k in ADJUST:
                b.nudge(name, ADJUST[k])
            elif k in ("home", "end"):
                b.set(name, spec.lo if k == "home" else spec.hi)
            elif k == "d":
                b.reset(name)
            else:
                return False
            return True
        return False

    def mouse(self, ev: Mouse) -> None:
        hits = self.rend.hits
        if ev.button in (64, 65):
            hit = next((h for h in hits if h.row == ev.row and h.span[0] <= ev.col <= h.span[1]), None)
            if hit:
                self._hit(hit, None, 1 if ev.button == 64 else -1)
            return
        if not ev.pressed:
            self.drag = None
            return
        if ev.button != 0:
            return
        if ev.motion:
            if self.drag is not None:
                h = self.drag
                self._hit(h, (ev.col - h.c0) / max(1, h.c1 - h.c0), 0)
            return
        hit = next((h for h in hits if h.row == ev.row and h.span[0] <= ev.col <= h.span[1]), None)
        if hit is None:
            return
        frac = (ev.col - hit.c0) / max(1, hit.c1 - hit.c0) if hit.c0 <= ev.col <= hit.c1 else None
        self._hit(hit, frac, 0)
        self.drag = hit if frac is not None else None

    def _hit(self, hit: Hit, frac: float | None, steps: int) -> None:
        if hit.kind == "slider" and hit.name in self.bank.specs:
            b = self.bank
            b.selected = b.names.index(hit.name)
            if steps:
                b.nudge(hit.name, steps)
            elif frac is not None:
                spec = b.specs[hit.name]
                b.set(hit.name, spec.lo + max(0.0, min(1.0, frac)) * (spec.hi - spec.lo))
        elif hit.kind == "tune" and self.tune is not None:
            s = self.tune
            i = next((j for j, p in enumerate(s.params) if p.name == hit.name), None)
            if i is None:
                return
            s.selected = i
            p = s.params[i]
            if steps:
                s.nudge(steps, p)
            elif frac is not None:
                if p.kind == "drive":
                    s.nudge(-1 if frac < 0.5 else 1, p)
                else:
                    s.set_fraction(p, frac)
            self._tuned()

    # -- keys

    def key(self, k, now: float) -> bool:
        """Handle one key or Mouse event. Returns False to quit."""
        if isinstance(k, Mouse):
            self.mouse(k)
            return True
        if k in ("q", "Q"):
            return False
        if k == "esc":
            if self.tune is not None:
                self.tune = None
            elif self.controls:
                self.controls = False
            elif self.meters:
                self.meters = False
            else:
                return False
            return True
        if self.adjust(k):
            return True
        if len(k) == 1 and k.isdigit():
            n = 10 if k == "0" else int(k)  # 0 selects panel 10
            if self.focus == n - 1 or n > len(self.panels):
                self.focus, self.tune = None, None
            else:
                self.focus = n - 1
                if self.tune is not None:
                    self.open_tune()
            self.meters = self.controls = False
        elif k in ("g", "G"):
            self.focus, self.tune = None, None
            self.meters = self.controls = False
        elif k in ("n", "N"):
            self.flip(1 if k == "n" else -1)
        elif k in (".", ","):
            n = len(self.panels)
            if self.focus is None:
                self.focus = 0 if k == "." else n - 1
            else:
                self.focus = (self.focus + (1 if k == "." else -1)) % n
            self.meters = self.controls = False
            if self.tune is not None:
                self.open_tune()
        elif k == "k":
            if self.tune is not None:
                self.tune = None
            else:
                self.open_tune()
        elif k in ("c", "C"):
            self.controls = not self.controls
            if self.controls:
                self.meters = False
        elif k in ("m", "M"):
            self.meters = not self.meters
            if self.meters:
                self.controls = False
        elif k in ("r", "R") and self.args is not None:
            self.reload()
        elif k == "P" and self.args is not None:
            names = [""] + self.cfg.profiles
            if len(names) == 1:
                self.note = "no profiles in the config (add sections like [tune.808@phonk])"
            else:
                self.reload(names[(names.index(self.cfg.profile or "") + 1) % len(names)])
        elif k == "b":
            self.an.tap(now)
            self.note = "tap tempo" if len(self.an.taps) >= 2 else "tap again to set the tempo"
        elif k == "B":
            self.an.reset_tempo()
            self.note = "tempo: " + ("fixed" if self.cfg.bpm else "auto")
        elif k in ("f", "F"):
            self.footer = (self.footer - 1) % 3
        elif k in ("t", "T"):
            self.theme_i = (self.theme_i + 1) % len(THEME_ORDER)
            self.cfg.theme = THEME_ORDER[self.theme_i]
            self.rend = Renderer(self.cfg.theme, None, self.cfg.color_mode)
        elif k in ("p", " "):
            self.paused = not self.paused
        elif k in ("+", "="):
            self.cfg.sensitivity = self.an.s.sensitivity = min(4.0, self.cfg.sensitivity * 1.15)
        elif k in ("-", "_"):
            self.cfg.sensitivity = self.an.s.sensitivity = max(0.2, self.cfg.sensitivity / 1.15)
        return True


def reload_config(args, old: Config, profile: str | None = None):
    """Re-read the config and rebuild panels and analysis. Returns (cfg, analyzer, panels, note).
    Kept for scripts; the app itself uses App.reload()."""
    cfg = load_config(args, profile if profile is not None else (old.profile or ""))
    an, panels = build(cfg)
    pending = [name for name in RESTART_ONLY if getattr(cfg, name) != getattr(old, name)]
    for name in RESTART_ONLY:
        setattr(cfg, name, getattr(old, name))
    what = f"profile {cfg.profile or 'default'}" if profile is not None else "config reloaded"
    note = what + (f" ({', '.join(pending)} apply on restart)" if pending else "")
    return cfg, an, panels, note


def run(cfg: Config, audible: bool, focus: int | None, args=None, meters: bool = False,
        controls: bool = False, tune: bool = False) -> int:
    app = App(cfg, args, focus, meters, controls, tune, persist=True)
    src, note = build_source(cfg.method, cfg.source, cfg.rate, cfg.channels, cfg.fifo, audible)
    app.note = app.note or note

    def _term(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _term)
    t0 = time.monotonic()
    last = t0
    lines: list[str] = []
    try:
        with Terminal(mouse=cfg.mouse) as term:
            while True:
                start = time.monotonic()
                for ev in parse_input(term.keys()):
                    if not app.key(ev, start - t0):
                        return 0

                if not src.alive and src.error and not isinstance(src, DemoSource):
                    app.note = f"{src.error} (showing demo)"
                    src = DemoSource(app.cfg.rate).start()

                now = time.monotonic()
                dt, last = now - last, now
                cols, rows = term.size()
                if not app.paused:
                    app.step(src.latest_stereo(app.an.need), now - t0, dt)
                lines = app.draw(cols, rows, src.name)
                term.draw(lines)
                app.autosave(now)
                spent = time.monotonic() - start
                frame_dt = 1.0 / app.cfg.framerate
                if spent < frame_dt:
                    time.sleep(frame_dt - spent)
    except KeyboardInterrupt:
        return 0
    finally:
        src.stop()
        app.bank.save()


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.keys:
        for k, desc in KEY_HELP:
            print(f"  {k:10s} {desc}")
        return 0
    cfg = load_config(a)
    if a.list_knobs:
        return print_knobs(cfg, a.list_knobs)
    if a.list_features:
        return print_features(cfg)
    if a.list_panels:
        for cls in BUILTIN:
            print(f"{cls.key:10s} {cls.title}")
        print("controls   your [slider.*] values")
        print()
        print("CERN page (panels = cern):")
        for cls in CERN:
            print(f"{cls.key:10s} {cls.title}")
        print()
        print("Black hole page (panels = hole):")
        for cls in HOLE:
            print(f"{cls.key:10s} {cls.title}  [{cls.verdict}]")
        print()
        for k, p in cfg.custom.items():
            print(f"{k:10s} {p.title} (config)")
        return 0
    if a.formulas:
        an, pages = build_pages(cfg)
        f = an.analyze(np.zeros(an.need), 0.0)
        print_formulas(pages[min(len(pages), max(1, a.page or 1)) - 1], f)
        return 0
    if a.render_wav:
        synth.write_wav(a.render_wav, a.seconds, cfg.rate)
        print(f"wrote {a.render_wav} ({a.seconds:.1f} s, stereo)")
        return 0
    if a.dump:
        return dump(cfg, a.dump, a.size, None if a.focus is None else a.focus - 1, a.at, a.meters, a,
                    a.controls, a.tune, a.keys_in or "")
    if not sys.stdout.isatty():
        print("mathwave: stdout is not a terminal (use --dump for headless output)", file=sys.stderr)
        return 2
    return run(cfg, a.audible, None if a.focus is None else a.focus - 1, a, a.meters, a.controls, a.tune)

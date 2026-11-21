"""Command line entry point and the real-time loop."""

from __future__ import annotations

import argparse
import os
import select
import signal
import sys
import termios
import time
import tty

from . import __version__, synth
from .audio import DemoSource, build_source
from .config import Config, check_fft_size, default_path, load
from .dsp import Analyzer, Features
from .panels import BUILTIN, COMMON_KEYS, DEFAULT_ORDER, make_panels
from .render import DRIVE_LABEL, THEMES, Renderer

THEME_ORDER = list(THEMES) + ["wal"]


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
    ap.add_argument("--panels", help=f"comma list of panels (default: {','.join(DEFAULT_ORDER)})")
    ap.add_argument("--layout", help="auto or COLSxROWS, for example 3x3")
    ap.add_argument("--theme", choices=THEME_ORDER, help="color theme")
    ap.add_argument("--fps", type=int, help="frames per second (default 60)")
    ap.add_argument("--sensitivity", type=float, help="gain applied after normalization (default 1.0)")
    ap.add_argument("--fft-size", type=int, help="FFT size: 1024, 2048 (default), 4096 or 8192; larger separates sub-bass better")
    ap.add_argument("--focus", type=int, help="start with panel N full screen")
    ap.add_argument("--no-color", action="store_true", help="disable color")
    ap.add_argument("--formulas", action="store_true", help="print every panel's formula and exit")
    ap.add_argument("--list-panels", action="store_true", help="list panel keys and exit")
    ap.add_argument("--list-knobs", nargs="?", const="all", metavar="PANEL",
                    help="list the [tune.<panel>] keys for every panel (or one) and exit")
    ap.add_argument("--render-wav", metavar="PATH", help="write the demo track to a WAV file and exit")
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
    if a.no_color:
        cfg.color_mode = "none"
    return cfg


def build_panels(cfg: Config):
    return make_panels(cfg.panels, cfg.custom, cfg.tune, cfg.attack, cfg.decay)


def print_knobs(cfg: Config, which: str) -> int:
    panels = [cls() for cls in BUILTIN] + list(cfg.custom.values())
    if which != "all":
        panels = [p for p in panels if p.key == which]
        if not panels:
            print(f"mathwave: no panel '{which}'", file=sys.stderr)
            return 2
    print("Keys every [tune.<panel>] section accepts:")
    for k, desc in COMMON_KEYS.items():
        print(f"  {k:10s} {desc}")
    print("  drive names: " + " ".join(sorted(DRIVE_LABEL)))
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


def print_formulas(panels) -> None:
    for i, p in enumerate(panels, 1):
        print(f"{i}. {p.title}")
        print(f"   {p.formula}")
        if p.identity:
            print(f"   {p.identity}")
        print(f"   driven by: {DRIVE_LABEL.get(p.drive, p.drive)}")
        if p.tuning_summary():
            print(f"   tuned: {p.tuning_summary()}")
        print(f"   sounds like: {p.sound}")
        print()


def _status(src_name: str, f: Features, cfg: Config, note: str | None, paused: bool) -> str:
    parts = [f"input: {src_name}", f"BPM≈{f.bpm:.0f}", f"gain {cfg.sensitivity:.2f}", f"theme {cfg.theme}"]
    if paused:
        parts.append("PAUSED")
    if note:
        parts.append(note)
    return " · ".join(parts)


def dump(cfg: Config, frames: int, size: str, focus: int | None = None, at: float | None = None) -> int:
    cols, rows = (int(v) for v in size.lower().split("x"))
    panels = build_panels(cfg)
    an = Analyzer(cfg.rate, fft_size=cfg.fft_size, n_bars=cfg.bars, attack=cfg.attack, decay=cfg.decay, noise_floor_db=cfg.noise_floor_db,
                  dynamic_range_db=cfg.dynamic_range_db, sensitivity=cfg.sensitivity)
    src, _ = build_source("offline", rate=cfg.rate)
    rend = Renderer(cfg.theme, cfg.gradient, cfg.color_mode)
    dt = 1.0 / cfg.framerate
    f = None
    start = 0.0 if at is None else max(0.0, at - frames * dt)
    for k in range(frames):
        t = start + (k + 1) * dt
        f = an.analyze(src.latest(an.n, t), t)
        for p in panels:
            p.update(f, dt)
    lines = rend.render(panels, f, cols, rows, layout=cfg.layout, footer=cfg.formulas, focus=focus,
                        status=_status("demo (offline)", f, cfg, None, False), hints=True)
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


class Terminal:
    def __init__(self):
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
        self.out.write("\x1b[?1049h\x1b[?25l\x1b[2J")
        self.out.flush()
        return self

    def __exit__(self, *exc):
        if self.old is not None:
            termios.tcsetattr(self.keys_fd, termios.TCSADRAIN, self.old)
        self.out.write("\x1b[0m\x1b[?25h\x1b[?1049l")
        self.out.flush()

    def keys(self) -> str:
        r, _, _ = select.select([self.keys_fd], [], [], 0)
        if not r:
            return ""
        try:
            return os.read(self.keys_fd, 64).decode(errors="ignore")
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


def reload_config(path: str | None, args, old: Config, an: Analyzer):
    """Re-read the config and rebuild the panels. Returns (cfg, panels, note).

    Settings that need a new audio pipeline (input, rate, fft_size, dynamic range,
    noise floor, bars) keep their running values until restart."""
    cfg = apply_args(load(path), args)
    panels = build_panels(cfg)
    an.attack, an.decay, an.sensitivity = cfg.attack, cfg.decay, cfg.sensitivity
    pending = [name for name in ("method", "source", "rate", "fft_size", "dynamic_range_db", "noise_floor_db", "bars")
               if getattr(cfg, name) != getattr(old, name)]
    note = "config reloaded" + (f" ({', '.join(pending)} apply on restart)" if pending else "")
    for name in ("method", "source", "rate", "fft_size", "dynamic_range_db", "noise_floor_db", "bars"):
        setattr(cfg, name, getattr(old, name))
    return cfg, panels, note


def run(cfg: Config, audible: bool, focus: int | None, args=None) -> int:
    panels = build_panels(cfg)
    an = Analyzer(cfg.rate, fft_size=cfg.fft_size, n_bars=cfg.bars, attack=cfg.attack, decay=cfg.decay, noise_floor_db=cfg.noise_floor_db,
                  dynamic_range_db=cfg.dynamic_range_db, sensitivity=cfg.sensitivity)
    src, note = build_source(cfg.method, cfg.source, cfg.rate, cfg.channels, cfg.fifo, audible)
    rend = Renderer(cfg.theme, cfg.gradient, cfg.color_mode)
    theme_i = THEME_ORDER.index(cfg.theme) if cfg.theme in THEME_ORDER else 0

    def _term(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _term)
    frame_dt = 1.0 / cfg.framerate
    paused = False
    footer = cfg.formulas
    t0 = time.monotonic()
    last = t0
    lines: list[str] = []
    f = Features()
    try:
        with Terminal() as term:
            while True:
                start = time.monotonic()
                for ch in term.keys():
                    if ch in "qQ\x1b":
                        return 0
                    if ch.isdigit():
                        n = 10 if ch == "0" else int(ch)  # 0 selects panel 10
                        focus = None if focus == n - 1 or n > len(panels) else n - 1
                    elif ch in "gG":
                        focus = None
                    elif ch in "rR" and args is not None:
                        try:
                            cfg, panels, note = reload_config(args.config, args, cfg, an)
                            rend = Renderer(cfg.theme, cfg.gradient, cfg.color_mode)
                            footer = cfg.formulas
                            if focus is not None and focus >= len(panels):
                                focus = None
                        except SystemExit as exc:
                            note = "config error: " + str(exc).removeprefix("mathwave: ").split(";")[0]
                    elif ch in "fF":
                        footer = (footer - 1) % 3
                    elif ch in "tT":
                        theme_i = (theme_i + 1) % len(THEME_ORDER)
                        cfg.theme = THEME_ORDER[theme_i]
                        rend = Renderer(cfg.theme, None, cfg.color_mode)
                    elif ch in "pP ":
                        paused = not paused
                    elif ch in "+=":
                        cfg.sensitivity = an.sensitivity = min(4.0, cfg.sensitivity * 1.15)
                    elif ch in "-_":
                        cfg.sensitivity = an.sensitivity = max(0.2, cfg.sensitivity / 1.15)

                if not src.alive and src.error and not isinstance(src, DemoSource):
                    note = f"{src.error} (showing demo)"
                    src = DemoSource(cfg.rate).start()

                now = time.monotonic()
                dt, last = now - last, now
                if not paused:
                    f = an.analyze(src.latest(an.n), now - t0)
                    for p in panels:
                        p.update(f, dt)
                    cols, rows = term.size()
                    lines = rend.render(panels, f, cols, rows, layout=cfg.layout, footer=footer, focus=focus,
                                        status=_status(src.name, f, cfg, note, paused))
                else:
                    cols, rows = term.size()
                    if lines and len(lines) != rows:
                        lines = rend.render(panels, f, cols, rows, layout=cfg.layout, footer=footer, focus=focus,
                                            status=_status(src.name, f, cfg, note, paused))
                term.draw(lines)
                spent = time.monotonic() - start
                if spent < frame_dt:
                    time.sleep(frame_dt - spent)
    except KeyboardInterrupt:
        return 0
    finally:
        src.stop()


def main(argv=None) -> int:
    a = parse_args(argv)
    cfg = apply_args(load(a.config), a)
    if a.list_knobs:
        return print_knobs(cfg, a.list_knobs)
    if a.list_panels:
        for cls in BUILTIN:
            print(f"{cls.key:10s} {cls.title}")
        for k, p in cfg.custom.items():
            print(f"{k:10s} {p.title} (config)")
        return 0
    if a.formulas:
        print_formulas(build_panels(cfg))
        return 0
    if a.render_wav:
        synth.write_wav(a.render_wav, a.seconds, cfg.rate)
        print(f"wrote {a.render_wav} ({a.seconds:.1f} s)")
        return 0
    if a.dump:
        return dump(cfg, a.dump, a.size, None if a.focus is None else a.focus - 1, a.at)
    if not sys.stdout.isatty():
        print("mathwave: stdout is not a terminal (use --dump for headless output)", file=sys.stderr)
        return 2
    return run(cfg, a.audible, None if a.focus is None else a.focus - 1, a)

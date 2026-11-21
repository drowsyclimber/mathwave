"""Terminal rendering: braille plots, panel boxes, grid layout and ANSI color."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np

import colorsys

from .dsp import BANDS, FEATURE_INFO, Features
from .panels import Panel

RESET = "\x1b[0m"

THEMES: dict[str, list[str]] = {
    "phonk": ["#8a3ffc", "#c23cff", "#ff3fa4", "#ff6b4a", "#ffb23f", "#ffe14d"],
    "ocean": ["#3d7cff", "#2aa8ff", "#22d3c5", "#41f0a0", "#a6ff6b"],
    "ember": ["#ff2e2e", "#ff6a2e", "#ff9f2e", "#ffd12e"],
    "matrix": ["#00ff6a", "#00d257", "#00a845"],
    "mono": ["#d8d8d8"],
}
# "chroma" is computed per frame from the dominant note; "wal" reads a pywal palette.
DYNAMIC_THEMES = ["chroma", "wal"]

# Braille dot weights indexed [row 0..3][col 0..1].
_BRAILLE_W = np.array([[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]], dtype=np.uint32)


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.strip().lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def gradient(colors: list[str], n: int) -> list[tuple[int, int, int]]:
    rgb = [hex_rgb(c) for c in colors]
    if n <= 1 or len(rgb) == 1:
        return [rgb[0]] * max(n, 1)
    out = []
    for i in range(n):
        u = i / (n - 1) * (len(rgb) - 1)
        j = min(int(u), len(rgb) - 2)
        f = u - j
        a, b = rgb[j], rgb[j + 1]
        out.append(tuple(int(round(a[k] + (b[k] - a[k]) * f)) for k in range(3)))
    return out


def wal_colors() -> list[str] | None:
    """Pick up a pywal palette if one exists (works with most Hyprland rices)."""
    path = os.path.expanduser("~/.cache/wal/colors.json")
    try:
        with open(path) as fh:
            c = json.load(fh)["colors"]
        return [c[f"color{i}"] for i in range(1, 7)]
    except Exception:
        return None


def chroma_palette(pc: int, tonality: float, n: int) -> list[tuple[int, int, int]]:
    """Twelve hues around the color wheel, one per pitch class; panels fan out from it.
    Tonal music gets saturated colors, noisy music drifts towards pastel."""
    base = (pc % 12) / 12.0
    sat = 0.45 + 0.45 * max(0.0, min(1.0, tonality))
    out = []
    for i in range(max(n, 1)):
        h = (base + (i / max(n, 1)) * 0.33) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, sat, 1.0)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def theme_colors(name: str, custom: list[str] | None = None) -> list[str]:
    if custom:
        return custom
    if name == "wal":
        return wal_colors() or THEMES["phonk"]
    return THEMES.get(name, THEMES["phonk"])


def _rgb256(r: int, g: int, b: int) -> int:
    q = lambda v: 0 if v < 48 else 1 if v < 115 else (v - 35) // 40  # noqa: E731
    return 16 + 36 * q(r) + 6 * q(g) + q(b)


class Styles:
    """Interns SGR sequences so a screen cell only needs a small integer."""

    def __init__(self, mode: str = "truecolor"):
        self.mode = mode
        self.sgr = [RESET]
        self.ids: dict[tuple, int] = {(None, False, False): 0}

    def get(self, rgb: tuple[int, int, int] | None, bold: bool = False, dim: bool = False) -> int:
        key = (rgb, bold, dim)
        if key in self.ids:
            return self.ids[key]
        parts = ["0"]
        if bold:
            parts.append("1")
        if rgb is not None:
            r, g, b = rgb
            if dim:
                r, g, b = int(r * 0.42), int(g * 0.42), int(b * 0.42)
            if self.mode == "256":
                parts.append(f"38;5;{_rgb256(r, g, b)}")
            elif self.mode != "none":
                parts.append(f"38;2;{r};{g};{b}")
        elif dim:
            parts.append("2")
        self.ids[key] = len(self.sgr)
        self.sgr.append("\x1b[" + ";".join(parts) + "m")
        return self.ids[key]


class Screen:
    def __init__(self, cols: int, rows: int):
        self.cols, self.rows = cols, rows
        self.ch = np.full((rows, cols), 32, dtype=np.uint32)
        self.st = np.zeros((rows, cols), dtype=np.uint16)

    def text(self, r: int, c: int, s: str, style: int, maxw: int | None = None) -> int:
        if r < 0 or r >= self.rows or c >= self.cols or not s:
            return 0
        room = self.cols - c if maxw is None else min(maxw, self.cols - c)
        if room <= 0:
            return 0
        if len(s) > room:
            s = s[: max(0, room - 1)] + "…"
        codes = np.frombuffer(s.encode("utf-32-le"), dtype=np.uint32)
        self.ch[r, c : c + len(codes)] = codes
        self.st[r, c : c + len(codes)] = style
        return len(codes)

    def lines(self, styles: Styles, color: bool = True) -> list[str]:
        out = []
        for r in range(self.rows):
            s = self.ch[r].tobytes().decode("utf-32-le")
            if not color:
                out.append(s)
                continue
            st = self.st[r]
            cuts = list(np.flatnonzero(np.diff(st)) + 1) + [self.cols]
            parts, start = [], 0
            for cut in cuts:
                parts.append(styles.sgr[int(st[start])])
                parts.append(s[start:cut])
                start = cut
            out.append("".join(parts) + RESET)
        return out


# ------------------------------------------------------------------ plotting


def plot_curve(grid: np.ndarray, y: np.ndarray, yrange: tuple[float, float]) -> None:
    """Draw y (one value per dot column) into a boolean dot grid, joining neighbors."""
    h, w = grid.shape
    y = np.asarray(y, dtype=np.float64)[:w]
    lo_y, hi_y = yrange
    with np.errstate(invalid="ignore"):
        r = (hi_y - y) / (hi_y - lo_y) * (h - 1)
    valid = np.isfinite(r)
    rc = np.where(valid, np.clip(np.round(np.nan_to_num(r)), -1, h), 0).astype(np.int64)
    prev = np.roll(rc, 1)
    prev_ok = np.roll(valid, 1)
    prev_ok[0] = False
    use_prev = valid & prev_ok
    # Walk half way toward the previous point so steep segments stay one dot thick.
    mid = np.where(use_prev, (rc + prev) / 2.0, rc)
    lo = np.floor(np.minimum(rc, mid)).astype(np.int64)
    hi = np.ceil(np.maximum(rc, mid)).astype(np.int64)
    nxt = np.roll(rc, -1)
    nxt_ok = np.roll(valid, -1)
    nxt_ok[-1] = False
    use_next = valid & nxt_ok
    mid2 = np.where(use_next, (rc + nxt) / 2.0, rc)
    lo = np.minimum(lo, np.floor(np.minimum(rc, mid2)).astype(np.int64))
    hi = np.maximum(hi, np.ceil(np.maximum(rc, mid2)).astype(np.int64))
    rows = np.arange(h)[:, None]
    mask = (rows >= lo[None, :]) & (rows <= hi[None, :]) & valid[None, :]
    grid[:, : len(y)] |= mask


def plot_points(grid: np.ndarray, y: np.ndarray, yrange: tuple[float, float]) -> None:
    """Draw unconnected dots, one per column (used for iterated maps)."""
    h, w = grid.shape
    y = np.asarray(y, dtype=np.float64)[:w]
    lo_y, hi_y = yrange
    with np.errstate(invalid="ignore"):
        r = np.round((hi_y - y) / (hi_y - lo_y) * (h - 1))
    ok = np.isfinite(r) & (r >= 0) & (r < h)
    cols = np.flatnonzero(ok)
    grid[r[ok].astype(np.int64), cols] = True


def plot_hline(grid, y0, yrange):
    h, w = grid.shape
    lo_y, hi_y = yrange
    r = int(round((hi_y - y0) / (hi_y - lo_y) * (h - 1)))
    if 0 <= r < h:
        grid[r, ::2] = True


def plot_vline(grid, x0, domain):
    h, w = grid.shape
    c = int(round((x0 - domain[0]) / (domain[1] - domain[0]) * (w - 1)))
    if 0 <= c < w:
        grid[::2, c] = True


def plot_bars(grid, heights):
    h, w = grid.shape
    n = len(heights)
    if n == 0:
        return
    col = np.arange(w)
    idx = np.minimum(col * n // w, n - 1)
    first = np.r_[True, idx[1:] != idx[:-1]]
    gap = first if w // n >= 3 else np.zeros(w, dtype=bool)
    tops = h - np.round(np.clip(np.asarray(heights)[idx], 0, 1) * h).astype(np.int64)
    rows = np.arange(h)[:, None]
    grid |= (rows >= tops[None, :]) & ~gap[None, :]


def braille_codes(grid: np.ndarray) -> np.ndarray:
    h4, w2 = grid.shape
    h, w = h4 // 4, w2 // 2
    g = grid[: h * 4, : w * 2].reshape(h, 4, w, 2).astype(np.uint32)
    return (g * _BRAILLE_W[None, :, None, :]).sum(axis=(1, 3)).astype(np.uint32)


# ------------------------------------------------------------------ layout


@dataclass
class Rect:
    r: int
    c: int
    h: int
    w: int


def grid_layout(n: int, cols: int, rows: int, spec: str = "auto", min_w: int = 40, min_h: int = 7) -> list[Rect] | None:
    if n <= 0:
        return []
    if spec and spec != "auto" and "x" in spec:
        gc, gr = (int(v) for v in spec.lower().split("x"))
        options = [(gc, gr)]
    else:
        options = []
        for gc in range(1, n + 1):
            gr = math.ceil(n / gc)
            pw, ph = cols / gc, rows / gr
            if pw < min_w or ph < min_h:
                continue
            score = abs(math.log((pw / ph) / 2.4)) + 0.2 * (gc * gr - n)
            options.append((score, gc, gr))
        if not options:
            return None
        options.sort()
        options = [(options[0][1], options[0][2])]
    gc, gr = options[0]
    rects = []
    heights = [rows // gr + (1 if i < rows % gr else 0) for i in range(gr)]
    r = 0
    k = 0
    for ri in range(gr):
        in_row = min(gc, n - k)
        if in_row <= 0:
            break
        widths = [cols // in_row + (1 if i < cols % in_row else 0) for i in range(in_row)]
        c = 0
        for wi in widths:
            rects.append(Rect(r, c, heights[ri], wi))
            c += wi
            k += 1
        r += heights[ri]
    return rects


# ------------------------------------------------------------------ renderer


DRIVE_LABEL = {**{k: f"{k} {int(lo)}–{int(hi)} Hz" for k, (lo, hi) in BANDS.items()}, **FEATURE_INFO}


def drive_label(name: str, f: Features | None = None) -> str:
    """Human label for a drive: band ranges (including custom and inline bands) or a feature."""
    if f is not None and name in f.band_ranges:
        lo, hi, mode = f.band_ranges[name]
        return f"{name} {lo:g}–{hi:g} Hz" + (" (peak)" if mode == "peak" else "")
    return DRIVE_LABEL.get(name, name)


class Renderer:
    def __init__(self, theme: str = "phonk", colors: list[str] | None = None, color_mode: str = "truecolor"):
        self.color_mode = color_mode
        self.theme = theme
        self.custom_colors = colors
        self.styles = Styles(color_mode)

    def palette(self, n: int, f: Features | None = None):
        if self.theme == "chroma" and not self.custom_colors:
            if f is None:
                return chroma_palette(0, 0.5, n)
            # 12 hues x 3 saturation steps keeps the style cache small
            tonal = round(f.levels.get("tonality", 0.5) * 2) / 2
            return chroma_palette(f.chroma_pc, tonal, n)
        return gradient(theme_colors(self.theme, self.custom_colors), n)

    def draw_panel(self, scr: Screen, rect: Rect, panel: Panel, idx: int, rgb, footer: int, focus: bool, f: Features):
        S = self.styles
        border = S.get(rgb, dim=True)
        title = S.get(rgb, bold=True)
        main = S.get(rgb)
        aux = S.get(rgb, dim=True)
        text = S.get((235, 235, 235), bold=True)
        sub = S.get((150, 150, 150))
        r0, c0, h, w = rect.r, rect.c, rect.h, rect.w
        if h < 3 or w < 6:
            return

        scr.text(r0, c0, "╭" + "─" * (w - 2) + "╮", border)
        for rr in range(r0 + 1, r0 + h - 1):
            scr.text(rr, c0, "│", border)
            scr.text(rr, c0 + w - 1, "│", border)
        scr.text(r0 + h - 1, c0, "╰" + "─" * (w - 2) + "╯", border)

        meter_w = 8 if w >= 40 else 0
        label = f" {idx + 1} {panel.title} "
        scr.text(r0, c0 + 2, label, title, maxw=w - 4 - (meter_w + 3 if meter_w else 0))
        if meter_w:
            filled = int(round(panel.level * meter_w))
            mc = c0 + w - 3 - meter_w
            scr.text(r0, mc - 1, " ", border)
            scr.text(r0, mc, "█" * filled, main)
            scr.text(r0, mc + filled, "░" * (meter_w - filled), aux)
            scr.text(r0, mc + meter_w, " ", border)

        inner_w = w - 4
        lines: list[tuple[str, int]] = []
        if footer >= 1:
            lines.append((panel.formula, text))
        if footer >= 2 and panel.identity:
            lines.append((panel.identity, sub))
        if footer >= 1:
            lines.append((panel.live(), main))
        if focus:
            lines += [("", sub), ("sound: " + panel.sound, sub), ("drive: " + drive_label(panel.drive, f)
                      + f"   level {panel.level:.2f}", sub)]
            tuned = panel.tuning_summary()
            lines.append(("tuned: " + tuned if tuned else "tuned: defaults (add a [tune." + panel.key + "] section)", sub))
        plot_h = h - 2 - len(lines) - (1 if lines else 0)
        while plot_h < 2 and lines:
            lines.pop(1 if len(lines) > 1 else 0)
            plot_h = h - 2 - len(lines) - (1 if lines else 0)
        if plot_h < 1:
            return

        pr, pc, pw = r0 + 1, c0 + 2, inner_w
        gh, gw = plot_h * 4, pw * 2
        x = np.linspace(panel.domain[0], panel.domain[1], gw)
        g_main = np.zeros((gh, gw), dtype=bool)
        g_aux = np.zeros((gh, gw), dtype=bool)
        for layer in panel.layers(x):
            kind, val, style = layer
            g = g_main if style == "main" else g_aux
            if kind == "curve":
                plot_curve(g, val, panel.yrange)
            elif kind == "hline":
                plot_hline(g, val, panel.yrange)
            elif kind == "vline":
                plot_vline(g, val, panel.domain)
            elif kind == "points":
                plot_points(g, val, panel.yrange)
            elif kind == "bars":
                plot_bars(g, val)
        cm, ca = braille_codes(g_main), braille_codes(g_aux)
        both = cm | ca
        ch = np.where(both > 0, 0x2800 + both, 32).astype(np.uint32)
        st = np.where(cm > 0, main, np.where(ca > 0, aux, 0)).astype(np.uint16)
        scr.ch[pr : pr + plot_h, pc : pc + pw] = ch
        scr.st[pr : pr + plot_h, pc : pc + pw] = st

        if lines:
            sep_r = pr + plot_h
            scr.text(sep_r, c0 + 1, "┄" * (w - 2), border)
            for i, (s, sty) in enumerate(lines):
                scr.text(sep_r + 1 + i, pc, s, sty, maxw=inner_w)

    # ------------------------------------------------------------ meters

    def meter_rows(self, f: Features) -> list[tuple[str, list[tuple[str, str, float, float, str]]]]:
        """Groups of (name, description, level, raw, extra) for the meter view."""
        lv, raw = f.levels, f.raw
        bands = []
        for name, (lo, hi, mode) in f.band_ranges.items():
            desc = f"{lo:g}–{hi:g} Hz" + (" peak" if mode == "peak" else "")
            db = f.band_db.get(name)
            bands.append((name, desc, lv.get(name, 0.0), raw.get(name, 0.0), f"{db:5.0f} dB" if db is not None else ""))
        spectrum = [
            ("rms", "loudness", lv["rms"], raw["rms"], ""),
            ("centroid", "brightness", lv["centroid"], raw["centroid"], f"{f.centroid_hz / 1000:4.1f} kHz"),
            ("rolloff", "85% rolloff", lv["rolloff"], raw["rolloff"], f"{f.rolloff_hz / 1000:4.1f} kHz"),
            ("flatness", "noisiness", lv["flatness"], raw["flatness"], ""),
            ("flux", "change", lv["flux"], raw["flux"], ""),
        ]
        hit = lambda k: "● hit" if f.onsets.get(k) else ""  # noqa: E731
        musical = [
            ("pitch", "bass note", lv["pitch"], raw["pitch"],
             f"{f.pitch_name} {f.pitch_hz:5.1f} Hz" if f.pitch_hz > 0 else "no note"),
            ("note", "pitch class", lv["note"], raw["note"], f.note_name),
            ("tonality", "one note dominates", lv["tonality"], raw["tonality"], ""),
            ("kick", "trigger", lv["kick"], raw["kick"], hit("kick")),
            ("snare", "trigger", lv["snare"], raw["snare"], hit("snare")),
            ("hat", "trigger", lv["hat"], raw["hat"], hit("hat")),
            ("phase", "beat position", lv["phase"], raw["phase"], f"BPM {f.bpm:.0f} {f.tempo_source}"),
        ]
        stereo = [(k, d, lv[k], raw[k], "") for k, d in
                  (("left", "loudness L"), ("right", "loudness R"), ("side", "L−R loudness"),
                   ("width", "0 mono · 1 wide"), ("balance", "0 L · 0.5 C · 1 R"))]
        return [("BANDS", bands), ("SPECTRUM", spectrum), ("MUSICAL", musical), ("STEREO", stereo)]

    def draw_meters(self, scr: Screen, rect: Rect, f: Features, pal) -> None:
        S = self.styles
        head = S.get(pal[0], bold=True)
        sub = S.get((150, 150, 150))
        txt = S.get((235, 235, 235))
        groups = self.meter_rows(f)
        lines: list[tuple] = []
        for gi, (title, rows) in enumerate(groups):
            lines.append(("head", title, gi))
            for row in rows:
                lines.append(("row", row, gi))
        usable = rect.h - 2
        ncols = max(1, -(-len(lines) // max(1, usable)))
        while ncols > 1 and rect.w // ncols < 58:
            ncols -= 1
        colw = rect.w // ncols
        per = -(-len(lines) // ncols)
        bar_w = max(8, min(30, colw - 52))
        scr.text(rect.r, rect.c + 1, "meters · live level (bar), smoothed and raw values · m to go back",
                 sub, maxw=rect.w - 2)
        for i, item in enumerate(lines):
            col, r = divmod(i, per)
            rr = rect.r + 1 + r
            if rr >= rect.r + rect.h:
                break
            c0 = rect.c + 1 + col * colw
            kind, data, gi = item
            color = pal[min(len(pal) - 1, gi * max(1, len(pal) // 4))]
            if kind == "head":
                scr.text(rr, c0, data, head)
                continue
            name, desc, level, raw, extra = data
            fill = int(round(max(0.0, min(1.0, level)) * bar_w))
            scr.text(rr, c0, f"{name[:10]:10s}", txt)
            scr.text(rr, c0 + 11, "█" * fill, S.get(color))
            scr.text(rr, c0 + 11 + fill, "░" * (bar_w - fill), S.get(color, dim=True))
            scr.text(rr, c0 + 12 + bar_w, f"{level:4.2f} {raw:4.2f}", txt)
            scr.text(rr, c0 + 23 + bar_w, f"{extra:16s} {desc}", sub, maxw=max(0, colw - 24 - bar_w))

    def render(self, panels: list[Panel], f: Features, cols: int, rows: int, *, layout: str = "auto",
               footer: int = 2, focus: int | None = None, status: str = "", hints: bool = True,
               meters: bool = False) -> list[str]:
        scr = Screen(cols, rows)
        S = self.styles
        pal = self.palette(len(panels), f)
        top = 1 if rows > 10 else 0
        if top:
            left = f" mathwave · {status}"
            right = "1-9,0 focus · m meters · r reload · P profile · b tap · t theme · q quit " if hints else ""
            scr.text(0, 0, left, S.get(pal[0], bold=True), maxw=cols)
            if right and len(left) + len(right) + 2 < cols:
                scr.text(0, cols - len(right), right, S.get((130, 130, 130)))
        if meters:
            self.draw_meters(scr, Rect(top, 0, rows - top, cols), f, pal)
        elif focus is not None and 0 <= focus < len(panels):
            self.draw_panel(scr, Rect(top, 0, rows - top, cols), panels[focus], focus, pal[focus], max(footer, 2), True, f)
        else:
            rects = grid_layout(len(panels), cols, rows - top, layout)
            if rects is None:
                msg = f"terminal too small for {len(panels)} panels: enlarge it, press 1-9, or set panels/layout"
                scr.text(rows // 2, max(0, (cols - len(msg)) // 2), msg, S.get(pal[0], bold=True))
            else:
                for i, (p, rc) in enumerate(zip(panels, rects)):
                    rc.r += top
                    self.draw_panel(scr, rc, p, i, pal[i], footer, False, f)
        return scr.lines(S, color=self.color_mode != "none")

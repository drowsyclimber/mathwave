"""The black hole page: your last 66.5 seconds inside Sagittarius A*, one formula per stage.

Based on the "Falling into a black hole: your last 67 seconds, as math" reel. Every panel
has two parts, like the reel's screen: a simulated black hole on top and the stage's
formula drawn as a trace underneath.

The simulated black hole is real optics, simplified. The accretion disk is a set of
rings from the innermost stable orbit (3 r_s) outward, seen at an angle, and each point
is moved to where its light actually lands on the sky using Beloborodov's light-bending
approximation (1 − cos α = (1 − cos ψ)(1 − r_s/r)). That lifts the far side of the disk
into the arc over the top of the hole. The side of the disk turning toward you is drawn
bright, from the Doppler and gravitational shift of an orbit at that radius. The ring at
b = (√27/2)·r_s is the photon ring, the edge of the shadow. The music ripples the disk:
each ring carries the spectrum around at its own orbital speed. Hot clumps orbit at Kepler
rates (inner ones faster), gas spirals in and plunges once inside 3 r_s, and background
stars drift behind the hole, each split into two images by the point-lens equation
β = θ − θ_E²/θ. Picture time is sped up by the music; a real orbit at 3 r_s around
Sgr A* takes about half an hour.

The fall is real too. A body dropped from rest at the horizon follows the cycloid
r = r_s·cos²(η/2), τ = (GM/c³)(η + sin η), which takes πGM/c³ = 66.5 s for Sgr A*,
the longest anyone can last inside. Every panel runs that clock in real time, so the
countdown, your radius and the tidal pull Δa = 2GM·L/r³ on a 2 m body are the actual
values at that moment.

Each panel is labeled. PHYSICS: the formula is a real law. MIXED: a real quantity is
inside, but how it becomes sound is a design choice. METAPHOR: the formula is sound
design and only the caption's physics is real. The focus view (1-9, 0) explains which
part is which.
"""

from __future__ import annotations

import math

import numpy as np

from .cern import C, dot, ring
from .panels import BY_KEY, GROUPS, TAU, Panel, _lerp

G = 6.67430e-11        # m³/(kg s²)
M_SUN = 1.98847e30     # kg
G0 = 9.80665           # m/s², one g
SGR_A = 4.297          # millions of suns (GRAVITY 2023)
BODY = 2.0             # m, head to feet
B_CRIT = math.sqrt(27.0) / 2.0   # photon ring radius in r_s
RADII = (3.0, 3.9, 4.9, 6.1)     # disk rings in r_s; 3 r_s is the innermost stable orbit
FINE_RADII = (3.0, 3.5, 4.1, 4.7, 5.4, 6.1, 6.9)  # the same disk with more rings, for big views
R_OUT = 7.0            # outer edge of the disk, in r_s
SCALE = 4.5            # r_s per scene unit
OFFSET = 1.3           # the image sits this far above the hole (the far-side arc), so center on it
THETA_E = 3.3          # Einstein radius of the background lens, in r_s


# ---------------------------------------------------------------- the physics


def hole(mass_millions: float) -> tuple[float, float, float]:
    """GM (m³/s²), r_s (m) and GM/c³ (s) for a mass in millions of suns."""
    gm = G * max(1e-6, mass_millions) * 1e6 * M_SUN
    return gm, 2.0 * gm / C**2, gm / C**3


def fall_time(mass_millions: float) -> float:
    """Longest proper time from the horizon to r = 0: π·GM/c³."""
    return math.pi * hole(mass_millions)[2]


def eta_of(frac):
    """Cycloid angle η in 0..π with (η + sin η)/π = frac, the fraction of the fall done."""
    if np.ndim(frac) == 0:  # the per-frame clock: plain floats are much faster
        target = min(1.0, max(0.0, float(frac))) * math.pi
        if target >= math.pi:
            return math.pi
        lo_s, hi_s = 0.0, math.pi
        for _ in range(48):
            mid = 0.5 * (lo_s + hi_s)
            if mid + math.sin(mid) < target:
                lo_s = mid
            else:
                hi_s = mid
        return 0.5 * (lo_s + hi_s)
    target = np.clip(np.asarray(frac, dtype=np.float64), 0.0, 1.0) * math.pi
    lo, hi = np.zeros_like(target), np.full_like(target, math.pi)
    for _ in range(48):  # η + sin η is monotone on 0..π
        mid = 0.5 * (lo + hi)
        low = mid + np.sin(mid) < target
        lo, hi = np.where(low, mid, lo), np.where(low, hi, mid)
    out = np.where(target >= math.pi, math.pi, 0.5 * (lo + hi))
    return float(out) if out.ndim == 0 else out


def r_of_eta(eta):
    """r / r_s on the fall from rest at the horizon."""
    return np.cos(np.asarray(eta) / 2.0) ** 2


def tidal_g(gm: float, r_m: float, length: float = BODY) -> float:
    """Head-to-feet tidal acceleration 2GM·L/r³, in g."""
    if r_m <= 0:
        return math.inf
    return 2.0 * gm * length / r_m**3 / G0


def r_at_tide(gm: float, dg: float, length: float = BODY) -> float:
    """The radius (m) where the head-to-feet tide is dg g."""
    return (2.0 * gm * length / (max(dg, 1e-30) * G0)) ** (1.0 / 3.0)


def g_text(dg: float) -> str:
    if not math.isfinite(dg):
        return "∞ g"
    if dg < 0.01:
        return f"{dg:.1e} g"
    if dg < 10:
        return f"{dg:.2f} g"
    if dg < 1e7:
        return f"{dg:,.0f} g"
    return f"{dg:.1e} g"


def km_text(r_m: float) -> str:
    km = r_m / 1000.0
    if km >= 1e6:
        return f"{km / 1e6:.2f}M km"
    if km >= 1:
        return f"{km:,.0f} km"
    return f"{r_m:,.0f} m"


def project(r, phi, tilt_deg: float):
    """Where light from disk points (r in r_s, azimuth phi) lands on the sky, seen at tilt_deg
    from face-on: (X, Y, shift), with light bending from Beloborodov's approximation and the
    shift g = √(1 − r_s/r) / (γ(1 − β·cos)) of a circular orbit there (g > 1 moves toward you)."""
    i = math.radians(min(89.5, max(1.0, tilt_deg)))
    si, ci = math.sin(i), math.cos(i)
    rr = np.maximum(np.asarray(r, dtype=np.float64), 1.05)
    phi = np.asarray(phi, dtype=np.float64)
    u = 1.0 / rr
    px, py = np.cos(phi), np.sin(phi)
    cos_psi = py * si
    cos_a = 1.0 - (1.0 - cos_psi) * (1.0 - u)
    sin_a = np.sqrt(np.clip(1.0 - cos_a**2, 0.0, 1.0))
    b = rr * sin_a / np.sqrt(1.0 - u)
    ex, ey, ez = px, py - cos_psi * si, -cos_psi * ci
    norm = np.sqrt(ex**2 + ey**2 + ez**2) + 1e-12
    X = b * ex / norm
    Y = b * (-ey * ci + ez * si) / norm
    beta = np.sqrt(np.clip(u / (2.0 * (1.0 - u)), 0.0, 0.99))
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    shift = np.sqrt(1.0 - u) / (gamma * (1.0 - beta * px * si))
    return X, Y, shift


def lens(bx, by, theta_e: float = THETA_E):
    """Point-lens images of a background star at sky position (bx, by): both solutions of
    the lens equation β = θ − θ_E²/θ, the bright one outside θ_E and the faint one inside."""
    bx, by = np.asarray(bx, dtype=np.float64), np.asarray(by, dtype=np.float64)
    beta = np.hypot(bx, by) + 1e-9
    root = np.sqrt(beta**2 + 4 * theta_e**2)
    outer, inner = (beta + root) / 2, (beta - root) / 2
    ux, uy = bx / beta, by / beta
    return (ux * outer, uy * outer), (ux * inner, uy * inner)


def disk_image(tilt_deg: float, radii=RADII, n: int = 181, ripple=None):
    """Sky positions (in r_s) of disk rings seen at tilt_deg from face-on, with light bending.

    Returns [(X, Y, shift)] per ring (see project). ripple(phi, r) -> fractional radius
    change, for the music."""
    phi = np.linspace(0.0, TAU, n)
    rr = np.concatenate([r0 * (1.0 + (ripple(phi, r0) if ripple is not None else np.zeros_like(phi))) for r0 in radii])
    X, Y, g = project(rr, np.tile(phi, len(radii)), tilt_deg)  # one call for every ring
    return [(X[k * n:(k + 1) * n], Y[k * n:(k + 1) * n], g[k * n:(k + 1) * n]) for k in range(len(radii))]


_SECONDARY: dict = {}


def secondary_image(tilt_deg: float, r0: float = RADII[0], n: int = 121):
    """The thin arc under the shadow: light from the far side that went under the hole.
    Approximated as the primary image pulled in toward the photon ring and flipped."""
    key = (round(tilt_deg * 20), r0, n)
    if key in _SECONDARY:
        return _SECONDARY[key]
    if len(_SECONDARY) > 256:
        _SECONDARY.clear()
    (X, Y, _), = disk_image(tilt_deg, (r0,), n)
    b = np.hypot(X, Y)
    far = Y > 0
    k = (B_CRIT + 0.12 * (b - B_CRIT)) / np.maximum(b, 1e-9)
    _SECONDARY[key] = np.where(far, -X * k, np.nan), np.where(far, -Y * k, np.nan)
    return _SECONDARY[key]


# ---------------------------------------------------------------- the base panel


_BASE_K = {
    "mass": (SGR_A, "black hole mass, in millions of suns (Sgr A*: 4.297)"),
    "tilt": (80.0, "view of the disk, degrees from face-on"),
}
_BASE_R = {"mass": (0.5, 100.0, 0.1), "tilt": (55.0, 89.0, 1.0)}


class _Hole(Panel):
    """A black hole scene over a trajectory trace. Subclasses define sweep(xs, f), which
    returns {channel: y} for the new x positions the pen crossed this frame."""

    scene_split = 0.6         # share of the plot rows the black hole gets in the grid
    scene_split_focus = 0.66  # and full screen
    verdict = "MIXED"
    truth = ""
    domain = (0.0, 4.0)       # in beats unless pen_on_fall
    pen_on_fall = False       # the pen follows the fall instead of the beat
    CHANNELS = ("main",)
    N = 600
    KNOBS = dict(_BASE_K)
    RANGES = dict(_BASE_R)
    # Per-frame values, readable before the first frame.
    amp, ratio, n, note, flash, d, pdg, pr = 0.4, 1.0, 0.0, "–", False, 0.05, 1e-4, 0.0

    def __init__(self, names=None):
        super().__init__(names)
        self.frac, self.hold, self.t = 0.0, 0.0, 0.0
        self.beat, self.pen = 0.0, self.domain[0]
        self.grid = np.linspace(self.domain[0], self.domain[1], self.N)
        self.traces = {c: np.full(self.N, np.nan) for c in self.CHANNELS}
        self.spec = np.zeros(48)
        self.onset = False
        self._fall()

    # -- the fall, in real proper time

    def _fall(self) -> None:
        gm, rs, _ = hole(self.k["mass"])
        self.gm, self.rs = gm, rs
        self.tau_max = fall_time(self.k["mass"])
        self.eta = eta_of(self.frac)
        self.r_rs = float(r_of_eta(self.eta))
        self.r_m = self.r_rs * rs
        self.tau_left = (1.0 - self.frac) * self.tau_max
        self.dg = tidal_g(gm, self.r_m)

    def probe(self, level: float) -> tuple[float, float]:
        """Music-chosen depth: level 0 is the horizon's tide, level 1 is 10,000 g.
        Returns (Δa in g, the radius in m where the tide is that strong)."""
        lo = math.log10(max(1e-12, tidal_g(self.gm, self.rs)))
        dg = 10 ** _lerp(lo, 4.0, max(0.0, min(1.0, level)))
        return dg, r_at_tide(self.gm, dg)

    def update(self, f, dt):
        super().update(f, dt)
        self.t += dt
        self.spec = np.asarray(f.spectrum, dtype=np.float64)
        self.onset = bool(f.onsets.get("kick"))
        self._animate(dt)
        if self.frac >= 1.0:
            self.hold += dt
            if self.hold > 2.0:  # you are gone; the next one falls in
                self.frac, self.hold = 0.0, 0.0
                if self.pen_on_fall:
                    self._clear()
        else:
            self.frac = min(1.0, self.frac + dt / max(1e-3, fall_time(self.k["mass"])))
        self._fall()
        self.react(f, dt)
        self._advance(f, dt)

    def react(self, f, dt) -> None:
        """Per-frame state before the pen moves (subclasses)."""

    def fall_pen(self) -> float:
        return self.domain[0] + (self.domain[1] - self.domain[0]) * self.frac

    # -- the trace

    def _clear(self) -> None:
        for c in self.traces:
            self.traces[c][:] = np.nan

    def _advance(self, f, dt) -> None:
        d0, d1 = self.domain
        step = (d1 - d0) / (self.N - 1)
        old = self.pen
        self.beat += dt / max(0.05, f.beat_period)
        if self.pen_on_fall:
            new = self.fall_pen()
            if new < old:
                self._clear()
                old = d0 - step
        else:
            new = old + dt / max(0.05, f.beat_period)
            if new >= d1:
                self._clear()
                new = d0 + (new - d1) % (d1 - d0)
                old = d0 - step
        i0 = max(0, int(math.floor((old - d0) / step)) + 1)
        i1 = min(self.N - 1, int(math.floor((new - d0) / step)))
        if i1 >= i0:
            xs = self.grid[i0 : i1 + 1]
            for c, y in self.sweep(xs, f).items():
                self.traces[c][i0 : i1 + 1] = y
        self.pen = new

    def sweep(self, xs, f) -> dict:
        return {}

    def trace(self, x, channel: str = "main"):
        d0, d1 = self.domain
        idx = np.clip(np.round((x - d0) / (d1 - d0) * (self.N - 1)).astype(int), 0, self.N - 1)
        return self.traces[channel][idx]

    def guides(self, x) -> list:
        return [("hline", 0.0, "aux")]

    def layers(self, x):
        out = self.guides(x)
        for c in self.CHANNELS:
            out.append(("curve", self.trace(x, c), "main"))
        out.append(("vline", self.pen, "aux"))
        return out

    # -- the scene

    def _setup_scene(self) -> None:
        rng = np.random.default_rng(1974)  # the year Sgr A* was found
        # Hot clumps riding the disk, and gas spiraling in from the outer edge.
        self.clump_r = rng.uniform(3.3, 6.6, 9)
        self.clump_phi = rng.uniform(0, TAU, 9)
        self.gas_r = rng.uniform(1.6, R_OUT, 36)
        self.gas_phi = rng.uniform(0, TAU, 36)
        # Background stars, in r_s on the sky, drifting behind the hole.
        self.star_x = rng.uniform(-22, 22, 70)
        self.star_y = rng.uniform(-4.5, 7.5, 70)
        self.star_b = rng.random(70) < 0.18
        self.flare = 0.0

    def _animate(self, dt: float) -> None:
        if not hasattr(self, "clump_r"):
            self._setup_scene()
        speed = 0.5 + 0.9 * self.level              # the music speeds the picture up (time is not to scale)
        omega = lambda r: 1.4 * speed * (3.0 / r) ** 1.5  # noqa: E731  Kepler: inner orbits turn faster
        self.clump_phi = (self.clump_phi + omega(self.clump_r) * dt) % TAU
        inside = self.gas_r < 3.0                   # inside the innermost stable orbit gas plunges
        self.gas_r = self.gas_r - dt * speed * np.where(inside, 2.2, 0.18)
        self.gas_phi = (self.gas_phi + omega(np.maximum(self.gas_r, 1.6)) * dt) % TAU
        gone = self.gas_r < 1.6
        if gone.any():
            self.gas_r[gone] = R_OUT
            self.gas_phi[gone] = np.random.default_rng(int(self.t * 1000) % 2**31).uniform(0, TAU, gone.sum())
        self.star_x = (self.star_x - dt * 0.45 * speed + 22) % 44 - 22
        self.flare = max(self.flare * math.exp(-dt / 0.35), self.level if self.onset else 0.0)

    def _ripple(self, phi, r):
        n = len(self.spec)
        if n == 0:
            return 0.0
        omega = 0.9 * (3.0 / r) ** 1.5  # inner rings turn faster, like orbits do
        idx = (((phi - omega * self.t) % TAU) / TAU * n).astype(int) % n
        return (0.05 + 0.06 * self.flare) * self.level * self.spec[idx]

    def you(self) -> tuple[float, float]:
        """Where you are drawn: inside the shadow, at your real r/r_s (not to scale)."""
        rad = 0.9 * B_CRIT / SCALE * self.r_rs
        return ring(np.array(math.radians(215.0)), rad)

    def scene(self, gh: int = 48, gw: int = 160) -> list:
        """The black hole, in scene units (y in about −1..1). gh and gw are the dot grid size,
        used to pick the level of detail."""
        if not hasattr(self, "clump_r"):
            self._setup_scene()
        big = gh >= 44
        tilt = self.k["tilt"] + 1.5 * math.sin(0.23 * self.t)  # a slow camera sway
        out = []
        # Stars behind the hole, bent by it: two images each, none inside the shadow.
        (ox, oy), (ix, iy) = lens(self.star_x, self.star_y)
        for (x, y), style in (((ox, oy), "aux"),) + ((((ix, iy), "aux"),) if big else ()):
            keep = np.hypot(x, y) > B_CRIT * 1.08
            out.append(("iso_points", (np.where(keep, x, np.nan), np.where(keep, y, np.nan)), style))
        bright = self.star_b & (np.hypot(ox, oy) > B_CRIT * 1.08)
        out.append(("iso_points", (np.where(bright, ox, np.nan), np.where(bright, oy, np.nan)), "main"))
        # The photon ring, which flares on the kick, and the thin image under the shadow.
        th = np.linspace(0, TAU, 241)
        out.append(("iso", ring(th, np.full_like(th, B_CRIT)), "main"))
        if self.flare > 0.08:
            out.append(("iso", ring(th, np.full_like(th, B_CRIT * (1.04 + 0.08 * self.flare))), "main"))
        sx, sy = secondary_image(tilt)
        out.append(("iso", (sx, sy), "aux"))
        # The disk: three brightness levels from the Doppler shift.
        hot = 1.1 - 0.15 * self.flare
        for X, Y, shift in disk_image(tilt, FINE_RADII if big else RADII, 241 if big else 161, self._ripple):
            solid = shift >= hot
            mid = (shift >= 0.8) & ~solid
            dim = shift < 0.8
            dim[1:] |= dim[:-1].copy()  # keep the dim side joined to the rest
            out.append(("iso", (np.where(solid, X, np.nan), np.where(solid, Y, np.nan)), "main"))
            m = mid & (np.arange(len(X)) % 2 == 0)
            out.append(("iso_points", (np.where(m, X, np.nan), np.where(m, Y, np.nan)), "main"))
            out.append(("iso", (np.where(dim, X, np.nan), np.where(dim, Y, np.nan)), "aux"))
        # Hot clumps: short thick arcs, brightest on the approaching side (one call for all).
        arc = np.linspace(-0.16, 0.16, 10)
        arc[-1] = np.nan  # pen up between arcs
        dr = np.array([-0.07, 0.0, 0.07])
        rr = (self.clump_r[:, None, None] + dr[None, :, None]) * np.ones_like(arc)
        pp = self.clump_phi[:, None, None] + np.nan_to_num(arc, nan=0.0) * np.ones((1, 3, 1))
        X, Y, g = project(rr.ravel(), pp.ravel(), tilt)
        cut = np.tile(np.isnan(arc), len(self.clump_r) * 3)
        X, Y = np.where(cut, np.nan, X), np.where(cut, np.nan, Y)
        hot_c = np.repeat(g.reshape(-1, 10).mean(axis=1) > 0.85, 10)
        out.append(("iso", (np.where(hot_c, X, np.nan), np.where(hot_c, Y, np.nan)), "main"))
        out.append(("iso", (np.where(~hot_c, X, np.nan), np.where(~hot_c, Y, np.nan)), "aux"))
        # Gas spiraling in.
        X, Y, g = project(self.gas_r, self.gas_phi, tilt)
        out.append(("iso_points", (np.where(g > 0.85, X, np.nan), np.where(g > 0.85, Y, np.nan)), "main"))
        out.append(("iso_points", (np.where(g <= 0.85, X, np.nan), np.where(g <= 0.85, Y, np.nan)), "aux"))
        # Everything above is in r_s on the sky; scale it and center the picture.
        out = [(k, (np.asarray(v[0]) / SCALE, (np.asarray(v[1]) - OFFSET) / SCALE), st) for k, v, st in out]
        # The stage's own drawing and you, already in scene units around the hole.
        extra = self.overlay()
        x, y = self.you()
        extra.append(("iso", dot(math.atan2(float(x), float(y)), float(math.hypot(x, y)), 0.035), "main"))
        out += [(k, (np.asarray(v[0]), np.asarray(v[1]) - OFFSET / SCALE), st) for k, v, st in extra]
        return _batch(out)

    def overlay(self) -> list:
        return []

    def tau_text(self) -> str:
        return f"τ left {self.tau_left:.1f} s" if self.tau_left >= 0.1 else f"τ left {self.tau_left * 1000:.1f} ms"


def _batch(layers: list) -> list:
    """Join layers of the same kind and style into one, with NaN between them, so the
    renderer draws a few long polylines instead of many short ones."""
    groups: dict[tuple[str, str], list] = {}
    for kind, (x, y), style in layers:
        groups.setdefault((kind, style), []).append((np.ravel(x), np.ravel(y)))
    out = []
    gap = np.array([np.nan])
    for (kind, style), parts in groups.items():
        xs = np.concatenate([np.concatenate([x, gap]) for x, _ in parts])
        ys = np.concatenate([np.concatenate([y, gap]) for _, y in parts])
        out.append((kind, (xs, ys), style))
    order = {"aux": 0, "main": 1}
    return sorted(out, key=lambda layer: order.get(layer[2], 0))


def _frac(x):
    return x - np.floor(x)


def _circle(cx, cy, r, n=72, sx=1.0, sy=1.0, angle=0.0):
    t = np.linspace(0, TAU, n)
    ex, ey = r * sx * np.cos(t), r * sy * np.sin(t)
    ca, sa = math.cos(angle), math.sin(angle)
    return cx + ex * ca - ey * sa, cy + ex * sa + ey * ca


# ---------------------------------------------------------------- 01 the horizon


class Horizon(_Hole):
    key = "horizon"
    title = "horizon · 808 fall"
    formula = "y = 3e^(−4{x})·sin(2π∫f₀·r/r_s dτ)"
    identity = "kick = whole fall, r/r_s from 1 to 0"
    default_drive = "kick"
    sound = ("You just crossed the event horizon and felt nothing. Every kick is the whole fall: "
             "the pitch follows r/r_s from 1 to 0 along the real cycloid, high to low in a flash.")
    verdict = "MIXED"
    truth = ("Real: at the horizon of a 4.3 million sun hole the tide on your body is about 1e-4 g, "
             "and r = 0 is at most πGM/c³ = 66.5 s away. Made up: pitch proportional to r/r_s.")
    KNOBS = {**_BASE_K, "f0": (7.0, "cycles at the top of each kick"), "damp": (4.0, "the 4 in e^(−4{x})")}
    RANGES = {**_BASE_R, "f0": (1.0, 20.0, 0.5), "damp": (0.5, 10.0, 0.1)}

    U = np.linspace(0.0, 1.0, 513)
    PHASE = np.concatenate([[0.0], np.cumsum(r_of_eta(eta_of(0.5 * (U[1:] + U[:-1]))) * np.diff(U))])

    def react(self, f, dt):
        self.amp = 0.6 + 0.4 * self.level

    def sweep(self, xs, f):
        u = _frac(xs)
        phase = self.k["f0"] * np.interp(u, self.U, self.PHASE)
        return {"main": self.amp * np.exp(-self.k["damp"] * u) * np.sin(TAU * phase)}

    def overlay(self):
        th = np.linspace(0, TAU, 73)
        hit = self.onset or self.level > 0.7
        return [("iso", ring(th, np.full_like(th, 0.9 * B_CRIT / SCALE)), "main" if hit else "aux")]

    def live(self):
        return f"{self.tau_text()}  r={self.r_rs:.2f} r_s"


# ---------------------------------------------------------------- 02 SOS


def _sos_gate(units):
    """··· ––– ··· as on/off: dot 1, dash 3, gaps 1, letters 3, then a 7-unit pause (34 units)."""
    pattern = [1, 1, 1, 1, 1, 3, 3, 1, 3, 1, 3, 3, 1, 1, 1, 1, 1, 7]
    on, edges, t = [], [], 0
    for k, w in enumerate(pattern):
        edges.append(t)
        on.append(k % 2 == 0)
        t += w
    edges = np.array(edges + [t], dtype=np.float64)
    u = np.mod(units, t)
    idx = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, len(on) - 1)
    return np.array(on)[idx].astype(np.float64)


class SOS(_Hole):
    key = "sos"
    title = "SOS · echoes"
    formula = "yₖ = bₖ + 2^(−k/2)·SOS(x − 3k/4)"
    identity = "×(1 + ⅕sin(32πx/2ᵏ)), k = echo"
    default_drive = "max(mid, hat)"
    sound = ("You call for help: dot dot dot, dash dash dash. Every echo comes back smaller and an "
             "octave lower, and nobody outside will ever hear it.")
    verdict = "METAPHOR"
    truth = ("Real: inside the horizon even light aimed outward moves to smaller r, so no signal ever "
             "gets out. Made up: SOS() is a gate spelling ··· ––– ···, and the echoes are an effect.")
    CHANNELS = ("e0", "e1", "e2", "e3")
    domain = (0.0, 6.0)
    KNOBS = {**_BASE_K, "echoes": (3.0, "echo rows k = 0, 1, 2 …"), "unit": (8.0, "Morse units per beat")}
    INT_KNOBS = {"echoes"}
    RANGES = {**_BASE_R, "echoes": (1.0, 4.0, 1.0), "unit": (2.0, 16.0, 1.0)}

    def __init__(self, names=None):
        super().__init__(names)
        self.calls: list[float] = []   # ages of the trapped light rings
        self._was_on = 0

    LETTERS = (0.0, 8.0, 22.0)  # where S, O and S start in the 34-unit word

    def react(self, f, dt):
        self.calls = [a + dt for a in self.calls if a < 1.6]
        u = (self.pen * self.k["unit"]) % 34.0
        letter = sum(u >= s for s in self.LETTERS)
        if letter != self._was_on:  # each letter leaves as one ring of light
            self.calls.append(0.0)
        self._was_on = letter

    def sweep(self, xs, f):
        n = int(self.k["echoes"])
        out = {}
        for k in range(4):
            if k >= n:
                out[f"e{k}"] = np.full_like(xs, np.nan)
                continue
            b = 0.75 - 1.5 * k / max(1, n - 1) if n > 1 else 0.0
            gate = _sos_gate((xs - 0.75 * k) * self.k["unit"])
            amp = (0.15 + 0.3 * self.level) * 2 ** (-k / 2)
            out[f"e{k}"] = b + amp * gate * (1 + 0.2 * np.sin(32 * math.pi * xs / 2**k)) - 0.5 * amp
        return out

    def guides(self, x):
        return []

    def overlay(self):
        # Each call leaves as a ring of light, and even that ring falls inward.
        out = []
        r0 = 0.9 * B_CRIT / SCALE
        for age in self.calls:
            r = r0 * max(0.0, 1.0 - age / 1.6)
            if r > 0.01:
                out.append(("iso", _circle(0.0, 0.0, r), "aux"))
        return out

    def live(self):
        return f"echoes {int(self.k['echoes'])}  −3 dB each  ↓ octave"


# ---------------------------------------------------------------- 03 the light


class Light(_Hole):
    key = "light"
    title = "the light · cot η/2"
    formula = "y=(1+n)e^(−2.2{4x})·cos(2πf/f₀·{4x})"
    identity = "f/f₀ = cot(η/2), η = fall angle"
    default_drive = "rms"
    sound = ("Light from the universe behind you keeps falling in with you. One spike per note of the "
             "arp: higher note, taller spike, and the pitch follows your cycloid angle η.")
    verdict = "MIXED"
    truth = ("Real: η is the cycloid angle of a radial fall, r = r_s·cos²(η/2), so cot(η/2) = √(r/(r_s − r)). "
             "Made up: that as a pitch ratio. And you do not watch the universe's whole future flash by.")
    KNOBS = {**_BASE_K, "cycles": (2.0, "cycles per spike when f/f₀ = 1")}
    RANGES = {**_BASE_R, "cycles": (0.5, 6.0, 0.25)}

    def react(self, f, dt):
        self.ratio = float(min(6.0, max(0.3, 1.0 / math.tan(max(1e-3, self.eta) / 2.0))))
        midi = 69 + 12 * math.log2(f.pitch_hz / 440.0) if f.pitch_hz > 0 else 57.0
        self.n = float(min(1.0, max(0.0, (midi - 40.0) / 40.0)))
        self.note = f.pitch_name
        self.flash = _frac(np.array(self.pen * 4.0)) < 0.15

    def sweep(self, xs, f):
        u = _frac(4.0 * xs)
        amp = (0.25 + 0.5 * self.n) * (0.4 + 0.6 * self.level)
        return {"main": -0.9 + 2.0 * amp * np.exp(-2.2 * u) * np.abs(np.cos(TAU * self.k["cycles"] * self.ratio * u))}

    def guides(self, x):
        return [("hline", -0.9, "aux")]

    def overlay(self):
        # Light from above keeps falling in: streaks moving down toward you.
        out = []
        k = int(2 + 6 * self.level)
        for j in range(k):
            a = math.radians(-60 + 120 * (j + 0.5) / k)
            s = _frac(np.array(self.t * 0.8 + j * 0.37))
            r1, r2 = 1.0 - 0.6 * float(s), 1.0 - 0.6 * float(s) - 0.12
            out.append(("iso", ring(np.array([a, a]), np.array([r1, max(0.3, r2)])), "aux"))
        if self.flash:
            th = np.linspace(0, TAU, 121)
            out.append(("iso", ring(th, np.full_like(th, B_CRIT / SCALE * 1.06)), "main"))
        return out

    def live(self):
        return f"η={self.eta:.2f}  f/f₀={self.ratio:.2f}  n={self.n:.2f} {self.note}"


# ---------------------------------------------------------------- 04 the engines


class Engines(_Hole):
    key = "engines"
    title = "engines · organ"
    formula = "y=E·[sin ωx + sin(3ωx/2) + ⅗sin 2ωx]"
    identity = "root, fifth 3:2 and octave 2:1"
    default_drive = "max(kick, bass)"
    sound = ("You fire the engines. Every burn is a full organ chord: root, fifth and octave, "
             "and none of it moves you outward.")
    verdict = "METAPHOR"
    truth = ("Real: no burn can stop the fall, and firing the engines only shortens your time; "
             "coasting lasts longest. Made up: the chord is additive synthesis.")
    KNOBS = {**_BASE_K, "omega": (3.0, "root cycles per beat (ω/2π)"), "burn": (1.5, "burn length, in beats")}
    RANGES = {**_BASE_R, "omega": (1.0, 8.0, 0.5), "burn": (0.25, 4.0, 0.25)}

    def __init__(self, names=None):
        super().__init__(names)
        self.since, self.burn_lv, self.E = 99.0, 0.0, 0.0

    def react(self, f, dt):
        self.since += dt / max(0.05, f.beat_period)
        if self.onset and self.level > 0.3:
            self.since, self.burn_lv = 0.0, self.level
        u = self.since / max(0.1, self.k["burn"])
        self.E = self.burn_lv * (1 - math.exp(-30 * u)) * math.exp(-2.5 * u)

    def sweep(self, xs, f):
        w = TAU * self.k["omega"]
        chord = np.sin(w * xs) + np.sin(1.5 * w * xs) + 0.6 * np.sin(2 * w * xs)
        return {"main": self.E * chord / 2.6}

    def overlay(self):
        x, y = self.you()
        a = math.atan2(float(x), float(y))
        r = float(math.hypot(x, y))
        plume = 0.05 + 0.35 * self.E
        return [("iso", ring(np.array([a, a]), np.array([r + 0.04, r + 0.04 + plume])), "main"),
                ("iso", ring(np.array([a + 0.25, a]), np.array([r + 0.04 + 0.6 * plume, r + 0.04 + plume])), "aux"),
                ("iso", ring(np.array([a - 0.25, a]), np.array([r + 0.04 + 0.6 * plume, r + 0.04 + plume])), "aux")]

    def live(self):
        return f"E={self.E:.2f}  ratios 1 : 3/2 : 2"


# ---------------------------------------------------------------- 05 no way out


class NoWayOut(_Hole):
    key = "noway"
    title = "no way out · r → 0"
    formula = "y = 3e^(−1.3(x−x₀))·cos 3π(x−x₀) → 0"
    identity = "inside, r = 0 is in every future"
    default_drive = "bass"
    sound = ("Every line spirals down to the same note: the center is in every future. Each kick "
             "starts a new line at x₀ and it rings down to zero.")
    verdict = "METAPHOR"
    truth = ("Real: inside the horizon r works like time, so r = 0 is in every future, as unavoidable "
             "as tomorrow. Made up: the damped cosine is a picture of that, not the motion.")
    KNOBS = {**_BASE_K, "spirals": (4.0, "paths drawn into the center")}
    INT_KNOBS = {"spirals"}
    RANGES = {**_BASE_R, "spirals": (1.0, 8.0, 1.0)}

    def __init__(self, names=None):
        super().__init__(names)
        self.x0, self.amp = 0.0, 0.0

    def react(self, f, dt):
        if (self.onset and self.level > 0.2) or self.pen < self.x0:
            self.x0, self.amp = self.pen, 0.3 + 0.7 * self.level

    def sweep(self, xs, f):
        d = np.maximum(0.0, xs - self.x0)
        return {"main": self.amp * np.exp(-1.3 * d) * np.cos(3 * math.pi * d)}

    def overlay(self):
        out = []
        n = int(self.k["spirals"])
        s = np.linspace(0.0, 1.0, 90)
        for j in range(n):
            a0 = TAU * j / n + 0.4 * self.t
            th = a0 + 2.4 * s
            r = (1.15 * (1 - s)) ** 1.5 * 0.95
            out.append(("iso", ring(th, r), "aux"))
        return out

    def live(self):
        return f"{self.tau_text()}  every path → r=0"


# ---------------------------------------------------------------- 06 tides


class Tides(_Hole):
    key = "tides"
    title = "tides · Δa = 2GML/r³"
    formula = "y = A(Δa)·sin(2π(τ₀ − τ(x)))"
    identity = "one wave per second you have left"
    default_drive = "rms"
    sound = ("The tides are rising. The window scrolls with your own clock: the last 6 seconds solid, "
             "the next 2 dim, one wave per second, until the wave runs into r = 0.")
    verdict = "MIXED"
    truth = ("Real: τ₀ − τ is your proper time left on a radial fall, and the tide across a 2 m body is "
             "Δa = 2GM·L/r³, only 1e-4 g at this horizon. Made up: the amplitude curve A(Δa).")
    domain = (-6.0, 2.0)  # seconds of proper time from now
    KNOBS = {**_BASE_K, "tick": (1.0, "seconds of proper time per wave")}
    RANGES = {**_BASE_R, "tick": (0.25, 4.0, 0.25)}

    def _wave(self, x):
        tau = self.frac * self.tau_max + x
        ok = (tau >= 0) & (tau <= self.tau_max)
        eta = eta_of(np.clip(tau / self.tau_max, 0.0, 1.0))
        r = np.maximum(r_of_eta(eta) * self.rs, 1.0)
        dg = 2.0 * self.gm * BODY / r**3 / G0
        a = np.clip((np.log10(dg) + 4.5) / 8.0, 0.12, 1.0) * (0.45 + 0.55 * self.level)
        return np.where(ok, a * np.sin(TAU * (self.tau_max - tau) / self.k["tick"]), np.nan)

    def layers(self, x):
        y = self._wave(x)
        past = np.where(x <= 0, y, np.nan)
        future = np.where(x > 0, y, np.nan)
        return [("hline", 0.0, "aux"), ("curve", future, "aux"), ("curve", past, "main"), ("vline", 0.0, "aux")]

    def overlay(self):
        x, y = self.you()
        a = math.atan2(float(x), float(y))
        e = min(0.85, max(0.0, (math.log10(max(self.dg, 1e-12)) + 4.0) / 8.0))
        return [("iso", _circle(float(x), float(y), 0.09, sx=1 + e, sy=1 - e / 2, angle=math.pi / 2 - a), "aux")]

    def live(self):
        return f"{self.tau_text()}  Δa={g_text(self.dg)}"


# ---------------------------------------------------------------- 07 head and feet


class HeadFeet(_Hole):
    key = "headfeet"
    title = "head/feet · g/4"
    formula = "y_feet(x) = y_head(x + g/4) at f/2"
    identity = "g = 2·log₁₀(Δa / 0.01 g)"
    default_drive = "rms"
    sound = ("Your head plays the melody up top and your feet play it early below, an octave down. "
             "Louder music takes you deeper, where the tide is stronger and the feet lead by more.")
    verdict = "MIXED"
    truth = ("Real: your feet are deeper than your head, so they are pulled harder and reach r = 0 a "
             "moment first; Δa is the real tide where the music puts you. Made up: the melody game.")
    CHANNELS = ("head", "feet")
    domain = (0.0, 6.0)
    KNOBS = {**_BASE_K, "cycles": (1.5, "head cycles per beat at A3")}
    RANGES = {**_BASE_R, "cycles": (0.5, 4.0, 0.25)}

    def __init__(self, names=None):
        super().__init__(names)
        self.hist_t: list[float] = []
        self.hist_q: list[float] = []
        self.ph_head = self.ph_feet = 0.0
        self.gl = 0.0
        self.pdg, self.pr = 1e-4, 0.0

    def react(self, f, dt):
        self.pdg, self.pr = self.probe(self.level)
        self.gl = max(0.0, 2.0 * math.log10(self.pdg / 0.01))
        q = f.pitch_hz / 220.0 if f.pitch_hz > 0 else 1.0
        self.hist_t.append(self.beat)
        self.hist_q.append(float(min(4.0, max(0.25, q))))
        while self.hist_t and self.hist_t[0] < self.beat - 12:
            self.hist_t.pop(0)
            self.hist_q.pop(0)

    def _q(self, beat):
        if not self.hist_t:
            return 1.0
        return float(np.interp(beat, self.hist_t, self.hist_q))

    def sweep(self, xs, f):
        n = len(xs)
        step = (self.domain[1] - self.domain[0]) / (self.N - 1)
        base = self.beat - (xs[-1] - xs)  # beat time of each new sample
        lead = self.gl / 4.0
        head, feet = np.empty(n), np.empty(n)
        for j in range(n):
            self.ph_feet += step * self.k["cycles"] * self._q(base[j]) / 2.0          # an octave down, now
            self.ph_head += step * self.k["cycles"] * self._q(base[j] - lead)         # the same tune, later
            head[j] = 0.5 + 0.3 * math.sin(TAU * self.ph_head)
            feet[j] = -0.5 + 0.3 * math.sin(TAU * self.ph_feet)
        return {"head": head, "feet": feet}

    def guides(self, x):
        return [("hline", 0.5, "aux"), ("hline", -0.5, "aux")]

    def overlay(self):
        x, y = self.you()
        a = math.atan2(float(x), float(y))
        r = float(math.hypot(x, y))
        sep = 0.03 + 0.012 * self.gl
        return [("iso", dot(a, r + sep, 0.02), "main"), ("iso", dot(a, max(0.0, r - sep), 0.02), "main"),
                ("iso", ring(np.array([a, a]), np.array([max(0.0, r - sep), r + sep])), "aux")]

    def live(self):
        return f"Δa={g_text(self.pdg)}  g={self.gl:.1f}  lead {self.gl / 4:.2f}"


# ---------------------------------------------------------------- 08 the squeeze


class Crush(_Hole):
    key = "crush"
    title = "squeeze · tide duck"
    formula = "y = (1 − d·e^(−5{x}))·sin 8πx"
    identity = "d ∝ log Δa: deeper, harder duck"
    default_drive = "max(kick, 0.5*rms)"
    sound = ("The tide crushes the sound on every beat, deeper and deeper: sidechain ducking whose depth d "
             "follows the log of the tidal pull where the music puts you.")
    verdict = "METAPHOR"
    truth = ("Real: tides stretch you along the fall and squeeze your sides with half that force, "
             "GM·L/r³. Made up: d is sidechain ducking on the beat.")

    def react(self, f, dt):
        self.pdg, self.pr = self.probe(self.level)
        lo = math.log10(max(1e-12, tidal_g(self.gm, self.rs)))
        self.d = min(0.95, max(0.05, (math.log10(self.pdg) - lo) / (4.0 - lo)))

    def sweep(self, xs, f):
        return {"main": 0.9 * (1 - self.d * np.exp(-5 * _frac(xs))) * np.sin(8 * math.pi * xs)}

    def overlay(self):
        x, y = self.you()
        a = math.atan2(float(x), float(y))
        duck = math.exp(-5 * float(_frac(np.array(self.pen))))
        e = 0.7 * self.d * (0.4 + 0.6 * duck)
        return [("iso", _circle(float(x), float(y), 0.09, sx=1 + e / 2, sy=1 - e, angle=math.pi / 2 - a), "aux")]

    def live(self):
        return f"Δa={g_text(self.pdg)}  d={self.d:.2f}  sides Δa/2"


# ---------------------------------------------------------------- 09 spaghetti


class Spaghetti(_Hole):
    key = "spaghetti"
    title = "spaghetti · stretch"
    formula = "y = sin(2π∫5/s(x) dx)"
    identity = "s = 1 + (3/2)·log₁₀(Δa / 1 g)"
    default_drive = "rms"
    sound = ("1 g, 10 g, 100 g: the same wave, stretched longer. The music sets how deep you are, "
             "and every tenfold rise in the tide stretches the wavelength by 1.5.")
    verdict = "MIXED"
    truth = ("Real: spaghettification. For this hole the tide on a 2 m body only reaches 1 g about "
             "615,000 km from the center, 0.3 s before the end. Made up: s as the log of the pull.")
    domain = (0.0, 6.0)

    def __init__(self, names=None):
        super().__init__(names)
        self.phase, self.s = 0.0, 1.0
        self.pdg, self.pr = 1e-4, 0.0

    def react(self, f, dt):
        self.pdg, self.pr = self.probe(self.level)
        self.s = 1.0 + 1.5 * max(0.0, math.log10(self.pdg))

    def sweep(self, xs, f):
        step = (self.domain[1] - self.domain[0]) / (self.N - 1)
        ph = self.phase + np.arange(1, len(xs) + 1) * step * 5.0 / self.s
        self.phase = float(ph[-1]) if len(ph) else self.phase
        return {"main": 0.9 * np.sin(TAU * ph)}

    def overlay(self):
        x, y = self.you()
        a = math.atan2(float(x), float(y))
        r = float(math.hypot(x, y))
        half = 0.025 * self.s
        return [("iso", ring(np.array([a, a]), np.array([max(0.0, r - half), r + half])), "main")]

    def live(self):
        return f"Δa={g_text(self.pdg)}  s={self.s:.1f}  r={km_text(self.pr)}"


# ---------------------------------------------------------------- 10 the singularity


class Singularity(_Hole):
    key = "singular"
    title = "singularity · r → 0"
    formula = "y = 3(1 − x/x₀)^(2/3)·sin 8πx → 0"
    identity = "r ∝ (τ₀ − τ)^(2/3) as r → 0"
    default_drive = "rms"
    sound = ("r → 0: the whole song squeezed into one point. The pen is your clock; the envelope is "
             "how your radius runs out.")
    verdict = "PHYSICS"
    truth = ("Real: near r = 0 a radial free fall obeys r ∝ (τ₀ − τ)^(2/3); the dim curve is the exact "
             "r(τ) from the cycloid and bends into the same power. Made up: the sine riding on it.")
    pen_on_fall = True
    domain = (0.0, 6.0)  # x₀ = 6 beats of the reel; the pen is the fraction of the fall done

    U = np.linspace(0.0, 1.0, 601)
    EXACT = r_of_eta(eta_of(U))

    def sweep(self, xs, f):
        env = np.clip(1.0 - xs / self.domain[1], 0.0, 1.0) ** (2.0 / 3.0)
        return {"main": 0.85 * (0.35 + 0.65 * self.level) * env * np.sin(8 * math.pi * xs)}

    def guides(self, x):
        r = np.interp(x / self.domain[1], self.U, self.EXACT) * 0.85
        return [("hline", 0.0, "aux"), ("curve", r, "aux"), ("curve", -r, "aux")]

    def overlay(self):
        return [("iso", _circle(0.0, 0.0, 0.9 * B_CRIT / SCALE * self.r_rs, n=121), "aux")]

    def live(self):
        return f"r={km_text(self.r_m)}  {self.tau_text()}"


HOLE = [Horizon, SOS, Light, Engines, NoWayOut, Tides, HeadFeet, Crush, Spaghetti, Singularity]
HOLE_ORDER = [cls.key for cls in HOLE]
BY_KEY.update({cls.key: cls for cls in HOLE})
GROUPS["hole"] = list(HOLE_ORDER)

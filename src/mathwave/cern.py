"""The CERN page: a proton's trip from Linac4 to the Higgs, one formula per stage.

Based on the "POV: you're a proton inside CERN" reel. Each panel prints the stage's
formula and computes it with real accelerator numbers (proton mass, LHC energies,
emittance, field), while the music moves it: the kick drum is the RF cavity, the
spectrum is the ring of magnets, the stereo channels are the two beams, and so on.

Ring-shaped stages use the renderer's "iso" layers: x and y in -1..1, drawn with the
same scale on both axes so circles stay round.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from .panels import BY_KEY, GROUPS, TAU, Panel, _lerp

C = 299_792_458.0     # m/s
M_P = 0.938272        # proton mass, GeV/c²
LAPS_PER_S = 11_245   # LHC revolution frequency


def gap_to_light(gamma: float) -> float:
    """c − v in m/s, computed without cancellation: 1 − β = (1/γ²)/(1 + β)."""
    inv = 1.0 / (gamma * gamma)
    return C * inv / (1.0 + math.sqrt(max(0.0, 1.0 - inv)))


def speed_text(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:,.0f} km/s"
    return f"{ms:.3g} m/s"


def gamma_text(g: float) -> str:
    return f"{g:.2f}" if g < 10 else f"{g:,.0f}"


def ring(theta: np.ndarray, r) -> tuple[np.ndarray, np.ndarray]:
    """Polar to the renderer's iso coordinates (0° at the top, clockwise like a dial)."""
    return r * np.sin(theta), r * np.cos(theta)


def dot(theta: float, r: float, size: float = 0.045) -> tuple[np.ndarray, np.ndarray]:
    """A small filled marker (the 'YOU' dot) as a short spiral."""
    t = np.linspace(0, TAU * 3, 60)
    rr = np.linspace(size, 0.0, 60)
    x0, y0 = ring(np.array(theta), r)
    return x0 + rr * np.cos(t), y0 + rr * np.sin(t)


def _wrapdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + math.pi) % TAU - math.pi


def resample(values: np.ndarray, n: int) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return np.zeros(n)
    edges = np.linspace(0, len(v), n + 1).astype(int)
    return np.array([v[a:max(a + 1, b)].mean() for a, b in zip(edges[:-1], edges[1:])])


class _Orbit:
    """Angle of the YOU dot: one lap every `beats` beats."""

    def __init__(self):
        self.theta = 0.0

    def step(self, dt: float, beats: float, beat_period: float, sign: float = 1.0) -> float:
        self.theta = (self.theta + sign * dt * TAU / max(0.05, beats * beat_period)) % TAU
        return self.theta


# ---------------------------------------------------------------- 01 the RF kick


class RFKick(Panel):
    key = "rf"
    title = "RF kick · cos φ"
    formula = "ΔE = qV₀·cos φ"
    identity = "one kick per gap, on the beat"
    default_drive = "max(kick, 0.6*sub)"
    sound = "Linac4: radio-frequency cavities push you through every gap. The kick drum is the RF wave."
    yrange = (-1.1, 1.1)
    KNOBS = {
        "v_min": (0.1, "RF voltage V₀ at level 0, in MV"),
        "v_max": (1.0, "RF voltage V₀ at level 1, in MV"),
        "phi": (-30.0, "synchronous phase φ, in degrees (crest is 0)"),
        "gaps": (12.0, "accelerating gaps across the panel"),
        "per_beat": (2.0, "gaps you cross per beat"),
    }
    INT_KNOBS = {"gaps"}
    RANGES = {"v_min": (0.0, 2.0, 0.05), "v_max": (0.1, 3.0, 0.05), "phi": (-90.0, 90.0, 1.0),
              "gaps": (4.0, 32.0, 1.0), "per_beat": (0.5, 8.0, 0.5)}

    def __init__(self, names=None):
        super().__init__(names)
        self.steps: deque[float] = deque([0.0] * 64, maxlen=64)
        self.pos = 0.0
        self.p = {"V₀": 0.1, "φ": -30.0, "ΔE": 0.0}

    def update(self, f, dt):
        super().update(f, dt)
        k = self.k
        v0 = _lerp(k["v_min"], k["v_max"], self.level)
        de = v0 * math.cos(math.radians(k["phi"]))
        self.pos += dt * k["per_beat"] / max(0.05, f.beat_period)
        while self.pos >= 1.0:
            self.pos -= 1.0
            self.steps.append(de)
        self.p = {"V₀": v0, "φ": k["phi"], "ΔE": de}

    def layers(self, x):
        g = int(self.k["gaps"])
        steps = np.array(list(self.steps)[-g:])
        energy = np.cumsum(steps)
        top = max(1e-6, g * max(self.k["v_max"], 1e-3))
        gx = (x - x[0]) / max(1e-9, x[-1] - x[0]) * g + self.pos
        idx = np.clip(np.floor(gx - self.pos).astype(int), 0, g - 1)
        y = -1.0 + 2.0 * energy[idx] / top
        rf = 0.45 * (self.p["V₀"] / max(1e-3, self.k["v_max"])) * np.cos(TAU * gx + math.radians(self.k["phi"]))
        return [("curve", rf, "aux"), ("hline", -1.0, "aux"), ("curve", y, "main")]

    def live(self):
        p = self.p
        return f"V₀={p['V₀']:.2f} MV  φ={p['φ']:+.0f}°  ΔE={p['ΔE']:.2f} MeV"


# ---------------------------------------------------------------- 02 the bend


class Bend(Panel):
    key = "bend"
    title = "bend · r = p/qB"
    formula = "r = p / (qB)"
    identity = "every magnet is a note: bar = band"
    default_drive = "rms"
    sound = "The PS Booster: dipole magnets hold you on a 25 m circle. Each magnet here plays one slice of the spectrum."
    KNOBS = {
        "magnets": (16.0, "dipole magnets around the ring"),
        "height": (0.35, "how far a loud magnet's bar reaches out"),
        "radius": (25.0, "ring radius r in m (PS Booster: 25)"),
        "beats": (4.0, "beats per lap of the YOU dot"),
    }
    INT_KNOBS = {"magnets"}
    RANGES = {"magnets": (4.0, 48.0, 1.0), "height": (0.05, 0.4, 0.01), "radius": (5.0, 4300.0, 5.0),
              "beats": (0.5, 16.0, 0.5)}

    def __init__(self, names=None):
        super().__init__(names)
        self.orbit = _Orbit()
        self.bars = np.zeros(16)
        self.p = {"p": 0.6, "B": 0.08}

    def update(self, f, dt):
        super().update(f, dt)
        k = self.k
        self.orbit.step(dt, k["beats"], f.beat_period)
        self.bars = resample(f.spectrum, int(k["magnets"]))
        kinetic = _lerp(0.16, 2.0, self.level)  # Booster: 160 MeV in, 2 GeV out
        p = math.sqrt(kinetic * kinetic + 2 * kinetic * M_P)
        self.p = {"p": p, "B": p / (0.299792458 * k["radius"])}

    def layers(self, x):
        n = int(self.k["magnets"])
        r0 = 0.6
        th = np.linspace(0, TAU, 361)
        base = ring(th, r0)
        xs, ys = [], []
        half = 0.32 * TAU / n
        for i in range(n):
            c = TAU * i / n
            h = 0.03 + self.k["height"] * float(self.bars[i]) * (0.35 + 0.65 * self.level)
            arc = np.linspace(c - half, c + half, 9)
            t = np.r_[c - half, arc, c + half]
            r = np.r_[r0, np.full(9, r0 + h), r0]
            px, py = ring(t, r)
            xs += [px, [np.nan]]
            ys += [py, [np.nan]]
        mx, my = np.concatenate(xs), np.concatenate(ys)
        dx, dy = dot(self.orbit.theta, r0)
        return [("iso", base, "aux"), ("iso", (mx, my), "main"), ("iso", (dx, dy), "main")]

    def live(self):
        return f"p={self.p['p']:.2f} GeV/c  B={self.p['B']:.3f} T  r={self.k['radius']:g} m"


# ---------------------------------------------------------------- 03 bunch splitting


class Bunches(Panel):
    key = "bunch"
    title = "bunches · Σ gauss"
    formula = "ρ(s) = Σₖ e^(−(s−sₖ)²/2σ²)"
    identity = "every split doubles the rhythm"
    default_drive = "mid"
    sound = "The Proton Synchrotron chops the beam into bunches. Each split doubles them, like quarter notes becoming eighths."
    KNOBS = {
        "splits": (5.0, "splits at level 1 (5 makes 2⁵ = 32 bunches)"),
        "sigma": (0.18, "bunch width σ, as a fraction of the spacing"),
        "height": (0.3, "bunch height around the ring"),
        "beats": (2.0, "beats per lap"),
    }
    INT_KNOBS = {"splits"}
    RANGES = {"splits": (1.0, 7.0, 1.0), "sigma": (0.03, 0.5, 0.01), "height": (0.05, 0.4, 0.01),
              "beats": (0.5, 16.0, 0.5)}

    def __init__(self, names=None):
        super().__init__(names)
        self.orbit = _Orbit()
        self.p = {"N": 1}

    def update(self, f, dt):
        super().update(f, dt)
        self.orbit.step(dt, self.k["beats"], f.beat_period)
        self.p = {"N": 2 ** int(round(self.k["splits"] * self.level))}

    def density(self, th: np.ndarray) -> np.ndarray:
        n = self.p["N"]
        centers = self.orbit.theta + TAU * np.arange(n) / n
        sig = self.k["sigma"] * TAU / n
        d = _wrapdist(th[:, None], centers[None, :])
        return np.exp(-(d * d) / (2 * sig * sig)).sum(axis=1)

    def layers(self, x):
        th = np.linspace(0, TAU, 721)
        r0 = 0.55
        r = r0 + self.k["height"] * (0.35 + 0.65 * self.level) * np.minimum(1.0, self.density(th))
        return [("iso", ring(th, np.full_like(th, r0)), "aux"), ("iso", ring(th, r), "main"),
                ("iso", dot(self.orbit.theta, r0 + self.k["height"] * (0.35 + 0.65 * self.level)), "main")]

    def live(self):
        return f"N={self.p['N']} bunches  σ={self.k['sigma']:.2f}·spacing"


# ---------------------------------------------------------------- 04 relativity


class Relativity(Panel):
    key = "gamma"
    title = "relativity · γ"
    formula = "v = c·√(1 − 1/γ²)"
    identity = "γ = E/mc²: more energy, same c"
    default_drive = "rms"
    sound = "The SPS: 17 times more energy buys a sliver more speed. The curve flattens against c and never crosses it."
    yrange = (-0.05, 1.1)
    KNOBS = {
        "e_min": (1.4, "total energy E at level 0, in GeV (a proton at rest is 0.938)"),
        "e_max": (6800.0, "total energy E at level 1, in GeV (LHC: 6800)"),
    }
    RANGES = {"e_min": (0.94, 100.0, 0.01), "e_max": (10.0, 7000.0, 10.0)}

    def update(self, f, dt):
        super().update(f, dt)
        lo, hi = max(M_P * 1.0001, self.k["e_min"]), max(self.k["e_min"] * 1.01, self.k["e_max"])
        e = math.exp(_lerp(math.log(lo), math.log(hi), self.level))
        self.p = {"E": e, "γ": e / M_P, "max": hi / M_P}

    def layers(self, x):
        gmax = self.p.get("max", 7247.0)
        u = (x - x[0]) / max(1e-9, x[-1] - x[0])
        g = np.exp(u * math.log(gmax))
        beta = np.sqrt(np.maximum(0.0, 1.0 - 1.0 / (g * g)))
        now = math.log(self.p.get("γ", 1.5)) / math.log(gmax)
        main = np.where(u <= now, beta, np.nan)
        xn = x[0] + now * (x[-1] - x[0])
        return [("curve", beta, "aux"), ("hline", 1.0, "aux"), ("vline", xn, "aux"), ("curve", main, "main")]

    def live(self):
        g = self.p.get("γ", 1.5)
        return f"γ={gamma_text(g)}  c−v={speed_text(gap_to_light(g))}"


# ---------------------------------------------------------------- 05 colder than space


class Cold(Panel):
    key = "cold"
    title = "cold · R = 0"
    formula = "R = 0   (T < T_c)"
    identity = "NbTi: T_c = 9.2 K, runs at 1.9 K"
    default_drive = "flatness"
    sound = "The LHC's superconducting magnets sit at 1.9 K, colder than space. Noisy, hissy passages heat them toward a quench."
    domain = (0.0, 15.0)
    yrange = (-0.1, 1.1)
    KNOBS = {
        "t_min": (1.9, "temperature T at level 0, in K"),
        "t_max": (12.0, "temperature T at level 1, in K"),
        "tc": (9.2, "critical temperature T_c, in K (NbTi: 9.2)"),
    }
    RANGES = {"t_min": (0.0, 8.0, 0.1), "t_max": (2.0, 15.0, 0.1), "tc": (1.0, 14.0, 0.1)}

    def update(self, f, dt):
        super().update(f, dt)
        self.p = {"T": _lerp(self.k["t_min"], self.k["t_max"], self.level)}

    def resistance(self, t: np.ndarray) -> np.ndarray:
        tc = self.k["tc"]
        on = 1.0 / (1.0 + np.exp(-(t - tc) / 0.08))
        return on * (0.6 + 0.03 * np.maximum(0.0, t - tc))

    def layers(self, x):
        r = self.resistance(x)
        main = np.where(x <= self.p.get("T", 1.9), r, np.nan)
        return [("curve", r, "aux"), ("hline", 0.0, "aux"), ("vline", self.p.get("T", 1.9), "aux"),
                ("curve", main, "main")]

    def live(self):
        t = self.p.get("T", 1.9)
        state = "R=0  superconducting" if t < self.k["tc"] else "R>0  QUENCH"
        return f"T={t:.2f} K  {state}"


# ---------------------------------------------------------------- 06 the ramp


class Ramp(Panel):
    key = "ramp"
    title = "ramp · E = γmc²"
    formula = "E = γmc²"
    identity = "11,245 laps a second, a bit per lap"
    default_drive = "rms"
    sound = "The LHC ramp: 450 GeV to 6.8 TeV, a few hundred keV per lap. At the top the beam is dumped and the ramp starts over."
    yrange = (-1.1, 1.1)
    HISTORY = 1200  # frames shown across the panel (20 s at 60 fps)
    KNOBS = {
        "ramp_s": (20.0, "seconds to ramp at full level (the real LHC takes about 20 minutes)"),
        "e_inj": (450.0, "injection energy, in GeV"),
        "e_top": (6800.0, "top energy, in GeV"),
    }
    RANGES = {"ramp_s": (2.0, 120.0, 1.0), "e_inj": (26.0, 3000.0, 2.0), "e_top": (500.0, 7000.0, 50.0)}

    def __init__(self, names=None):
        super().__init__(names)
        self.e = 450.0
        self.hold = 0.0
        self.rate = 0.0
        self.hist = np.full(self.HISTORY, np.nan)

    def update(self, f, dt):
        super().update(f, dt)
        k = self.k
        lo, hi = k["e_inj"], max(k["e_inj"] + 1.0, k["e_top"])
        self.e = min(max(self.e, lo), hi)
        self.rate = (hi - lo) / max(0.5, k["ramp_s"]) * self.level
        if self.e >= hi:
            self.hold += dt
            if self.hold > 1.0:  # beam dump, then a fresh fill at injection
                self.e, self.hold = lo, 0.0
        else:
            self.e = min(hi, self.e + self.rate * dt)
        self.hist = np.r_[self.hist[1:], self.e]

    def layers(self, x):
        top = max(self.k["e_top"], self.k["e_inj"] + 1.0)
        h = self.hist[np.linspace(0, len(self.hist) - 1, len(x)).astype(int)]
        smooth = -1.0 + 2.0 * h / top
        steps = -1.0 + 2.0 * np.floor(h / top * 24.0) / 24.0
        return [("hline", 1.0, "aux"), ("curve", smooth, "aux"), ("curve", steps, "main")]

    def live(self):
        g = self.e / M_P
        per_lap = self.rate / LAPS_PER_S * 1e6  # keV
        gain = f"+{per_lap / 1000:.1f} MeV/lap" if per_lap >= 1000 else f"+{per_lap:.0f} keV/lap"
        return f"E={self.e / 1000:.2f} TeV  γ={gamma_text(g)}  {gain}"


# ---------------------------------------------------------------- 07 the other beam


class Beams(Panel):
    key = "beams"
    title = "two beams · E_cm"
    formula = "E_cm = 2E"
    identity = "= 2√(E₁E₂) for head-on beams"
    default_drive = "rms"
    sound = "Beam 2 races the other way round the ring. The left channel is beam 1, the right is beam 2; they meet at the detectors."
    KNOBS = {
        "e_top": (6800.0, "energy per beam at level 1, in GeV"),
        "bunches": (8.0, "bunches drawn per beam"),
        "beats": (4.0, "beats per lap"),
    }
    INT_KNOBS = {"bunches"}
    RANGES = {"e_top": (450.0, 7000.0, 50.0), "bunches": (1.0, 32.0, 1.0), "beats": (0.5, 16.0, 0.5)}

    def __init__(self, names=None):
        super().__init__(names)
        self.o1, self.o2 = _Orbit(), _Orbit()
        self.p = {"E₁": 450.0, "E₂": 450.0, "l": 0.0, "r": 0.0}

    def update(self, f, dt):
        super().update(f, dt)
        k = self.k
        self.o1.step(dt, k["beats"], f.beat_period, 1.0)
        self.o2.step(dt, k["beats"], f.beat_period, -1.0)
        left, right = f.levels.get("left", 0.0), f.levels.get("right", 0.0)
        top = max(left, right, 1e-6)
        e = lambda ch: _lerp(450.0, max(450.0, k["e_top"]), self.level * ch / top)  # noqa: E731
        self.p = {"E₁": e(left), "E₂": e(right), "l": left / top, "r": right / top}

    def _beam(self, theta0: float, r0: float, amp: float):
        n = int(self.k["bunches"])
        th = np.linspace(0, TAU, 721)
        centers = theta0 + TAU * np.arange(n) / n
        d = _wrapdist(th[:, None], centers[None, :])
        sig = 0.12 * TAU / n
        bumps = np.exp(-(d * d) / (2 * sig * sig)).sum(axis=1)
        return ring(th, r0 + amp * np.minimum(1.0, bumps))

    def layers(self, x):
        a1 = 0.06 + 0.22 * self.level * self.p["l"]
        a2 = 0.06 + 0.22 * self.level * self.p["r"]
        b1 = self._beam(self.o1.theta, 0.62, a1)
        b2 = self._beam(self.o2.theta, 0.5, -a2 * 0.6)
        ticks_x, ticks_y = [], []
        for ip in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2):  # the four experiments
            tx, ty = ring(np.array([ip, ip]), np.array([0.3, 0.42]))
            ticks_x += [tx, [np.nan]]
            ticks_y += [ty, [np.nan]]
        return [("iso", b2, "aux"), ("iso", (np.concatenate(ticks_x), np.concatenate(ticks_y)), "aux"),
                ("iso", b1, "main"), ("iso", dot(self.o1.theta, 0.62), "main")]

    def live(self):
        e1, e2 = self.p["E₁"], self.p["E₂"]
        return f"E₁={e1 / 1000:.2f} E₂={e2 / 1000:.2f} → E_cm={2 * math.sqrt(e1 * e2) / 1000:.1f} TeV"


# ---------------------------------------------------------------- 08 the squeeze


class Squeeze(Panel):
    key = "squeeze"
    title = "squeeze · √εβ*"
    formula = "σ* = √(εβ*)"
    identity = "σ(s) = √(ε(β* + s²/β*))"
    default_drive = "bass"
    sound = "Quadrupoles squeeze the beam at the collision point: β* drops from 11 m to 30 cm and the beam narrows to about 10 µm."
    domain = (-2.0, 2.0)
    yrange = (-1.1, 1.1)
    KNOBS = {
        "beta_max": (11.0, "β* at level 0, in m (injection optics)"),
        "beta_min": (0.3, "β* at level 1, in m (collision optics)"),
        "eps_n": (2.5, "normalized emittance, in µm"),
        "energy": (6800.0, "beam energy, in GeV"),
    }
    RANGES = {"beta_max": (0.5, 20.0, 0.1), "beta_min": (0.1, 5.0, 0.05), "eps_n": (0.5, 10.0, 0.1),
              "energy": (450.0, 7000.0, 50.0)}

    def update(self, f, dt):
        super().update(f, dt)
        k = self.k
        eps = k["eps_n"] * 1e-6 / (k["energy"] / M_P)
        lo, hi = sorted((max(0.01, k["beta_min"]), max(0.01, k["beta_max"])))
        beta = math.exp(_lerp(math.log(hi), math.log(lo), self.level))
        self.p = {"ε": eps, "β*": beta, "σ*": math.sqrt(eps * beta), "ref": math.sqrt(eps * (lo + 4.0 / lo))}

    def layers(self, x):
        p = self.p
        if not p:
            return []
        sig = np.sqrt(p["ε"] * (p["β*"] + x * x / p["β*"])) / (1.1 * p["ref"])
        return [("hline", 0.0, "aux"), ("vline", 0.0, "aux"), ("curve", sig, "main"), ("curve", -sig, "main")]

    def live(self):
        p = self.p
        if not p:
            return "β* waiting for the beam"
        return f"β*={p['β*']:.2f} m  σ*={p['σ*'] * 1e6:.1f} µm  ε={p['ε'] * 1e9:.2f} nm"


# ---------------------------------------------------------------- 09 overlap


class Overlap(Panel):
    key = "overlap"
    title = "overlap · luminosity"
    formula = "L ∝ e^(−d²/4σ²)"
    identity = "two Gaussian bunches cross"
    default_drive = "kick"
    sound = "Two bunches of 100 billion protons cross head-on and only about 60 protons hit. Every kick pulls them together."
    domain = (-5.0, 5.0)
    yrange = (-0.05, 1.1)
    KNOBS = {
        "d_max": (4.0, "separation d at level 0, in units of σ"),
        "hits": (60.0, "collisions per crossing when the bunches line up"),
    }
    RANGES = {"d_max": (0.5, 8.0, 0.1), "hits": (1.0, 200.0, 1.0)}

    def update(self, f, dt):
        super().update(f, dt)
        d = self.k["d_max"] * (1.0 - self.level)
        self.p = {"d": d, "L": math.exp(-d * d / 4.0)}

    def layers(self, x):
        d = self.p.get("d", 4.0)
        a = np.exp(-((x + d / 2) ** 2) / 2.0)
        b = np.exp(-((x - d / 2) ** 2) / 2.0)
        return [("curve", a * b, "aux"), ("hline", 0.0, "aux"), ("curve", a, "main"), ("curve", b, "main")]

    def live(self):
        d, lum = self.p.get("d", 4.0), self.p.get("L", 0.0)
        return f"d={d:.2f}σ  L/L₀={lum:.2f}  hits≈{self.k['hits'] * lum:.0f}"


# ---------------------------------------------------------------- 10 collision


class Collision(Panel):
    key = "collide"
    title = "collision · tracks"
    formula = "E = mc²"
    identity = "tracks curl: r = p_T/(qB)"
    default_drive = "max(kick, snare)"
    sound = "13.6 TeV turns into new particles. Charged ones curl in the detector's magnet; every hit drum is a new event."
    KNOBS = {
        "tracks_min": (6.0, "tracks in a quiet event"),
        "tracks_max": (36.0, "tracks in a loud event"),
        "field": (1.0, "how hard tracks curl (1 = CMS's 3.8 T)"),
        "reveal": (0.25, "seconds for tracks to grow out from the center"),
    }
    INT_KNOBS = {"tracks_min", "tracks_max"}
    RANGES = {"tracks_min": (1.0, 40.0, 1.0), "tracks_max": (2.0, 150.0, 1.0), "field": (0.0, 4.0, 0.05),
              "reveal": (0.0, 1.0, 0.05)}

    def __init__(self, names=None):
        super().__init__(names)
        self.rng = np.random.default_rng(67)
        self.age = 0.0
        self.last_event = -1e9
        self.tracks: list[tuple[float, float, float]] = []  # (angle, curvature, energy)
        self.photons: list[float] = []
        self.events = 0
        self._new_event()

    def _new_event(self) -> None:
        k = self.k
        n = int(round(_lerp(k["tracks_min"], max(k["tracks_min"], k["tracks_max"]), self.level)))
        phi = self.rng.uniform(0, TAU, n)
        pt = self.rng.exponential(1.0, n) + 0.15
        q = self.rng.choice([-1.0, 1.0], n)
        self.tracks = [(float(a), float(c * 0.9 * k["field"] / p), float(p)) for a, c, p in zip(phi, q, pt)]
        self.photons = []
        if self.rng.random() < 0.3:  # a diphoton event: two straight, back-to-back lines
            a = float(self.rng.uniform(0, TAU))
            self.photons = [a, a + math.pi + float(self.rng.normal(0, 0.25))]
        self.age = 0.0
        self.events += 1

    def update(self, f, dt):
        super().update(f, dt)
        self.age += dt
        hit = f.onsets.get("kick") or f.onsets.get("snare")
        if (hit and self.level > 0.15) or self.age > 2 * f.beat_period:
            self._new_event()

    def layers(self, x):
        reach = 1.0 if self.k["reveal"] <= 0 else min(1.0, self.age / self.k["reveal"])
        s = np.linspace(0.0, 0.9 * reach, 40)
        xs, ys, tx, ty = [], [], [], []
        for a, kap, p in self.tracks:
            # Direction ψ(s) = a + κs, measured like ring(): x = sin ψ, y = cos ψ.
            if abs(kap) < 1e-6:
                px, py = s * np.sin(a), s * np.cos(a)
            else:
                px = (np.cos(a) - np.cos(a + kap * s)) / kap
                py = (np.sin(a + kap * s) - np.sin(a)) / kap
            r = np.hypot(px, py)
            px, py = np.where(r <= 0.9, px, np.nan), np.where(r <= 0.9, py, np.nan)
            xs += [px, [np.nan]]
            ys += [py, [np.nan]]
            if reach >= 1.0 and p > 1.2 and np.isfinite(px[-1]):
                end = math.atan2(px[-1], py[-1])
                cx, cy = ring(np.array([end, end]), np.array([0.92, 0.92 + min(0.07, 0.025 * p)]))
                tx += [cx, [np.nan]]
                ty += [cy, [np.nan]]
        for a in self.photons:
            px, py = ring(np.array([a, a]), np.array([0.0, 0.92 * reach]))
            xs += [px, [np.nan]]
            ys += [py, [np.nan]]
        th = np.linspace(0, TAU, 241)
        out = [("iso", ring(th, np.full_like(th, 0.92)), "aux"), ("iso", ring(th, np.full_like(th, 0.3)), "aux")]
        if xs:
            out.append(("iso", (np.concatenate(xs), np.concatenate(ys)), "main"))
        if tx:
            out.append(("iso", (np.concatenate(tx), np.concatenate(ty)), "main"))
        return out

    def live(self):
        kind = "  γγ" if self.photons else ""
        return f"{len(self.tracks)} tracks  E=13.6 TeV  B={3.8 * self.k['field']:.1f} T{kind}"


# ---------------------------------------------------------------- 11 the Higgs


class Higgs(Panel):
    key = "higgs"
    title = "Higgs · 125 GeV"
    formula = "m_H = 125 GeV"
    identity = "m² = 2E₁E₂(1 − cos θ)"
    default_drive = "rms"
    sound = "Billions of crashes later a bump rises at 125 GeV: the diphoton mass of the Higgs boson, found in 2012."
    domain = (100.0, 160.0)
    yrange = (0.0, 1.05)
    MASS, WIDTH = 125.0, 1.7
    KNOBS = {
        "rate": (1500.0, "events per second at level 1"),
        "signal": (0.04, "fraction of events that are Higgs bosons"),
        "memory": (40.0, "seconds before old events fade (0 keeps them all)"),
        "bins": (30.0, "histogram bins between 100 and 160 GeV"),
    }
    INT_KNOBS = {"bins"}
    RANGES = {"rate": (100.0, 50000.0, 100.0), "signal": (0.0, 0.3, 0.005), "memory": (0.0, 300.0, 5.0),
              "bins": (10.0, 60.0, 1.0)}

    def __init__(self, names=None):
        super().__init__(names)
        self.rng = np.random.default_rng(2012)
        self.counts = np.zeros(30)
        self.seen = 0.0

    def _pdf(self, m: np.ndarray, sig: float) -> np.ndarray:
        bkg = np.exp(-(m - 100.0) / 22.0)
        bkg /= np.exp(-(np.linspace(100, 160, 601) - 100.0) / 22.0).mean() * 60.0
        peak = np.exp(-((m - self.MASS) ** 2) / (2 * self.WIDTH**2)) / (self.WIDTH * math.sqrt(TAU))
        return (1 - sig) * bkg + sig * peak

    def update(self, f, dt):
        super().update(f, dt)
        k = self.k
        nb = int(k["bins"])
        if len(self.counts) != nb:
            self.counts = np.zeros(nb)
        if k["memory"] > 0:
            self.counts *= math.exp(-dt / k["memory"])
        centers = 100.0 + (np.arange(nb) + 0.5) * 60.0 / nb
        expected = k["rate"] * self.level * dt * self._pdf(centers, k["signal"]) * (60.0 / nb)
        self.counts += self.rng.poisson(np.maximum(0.0, expected))
        self.seen += k["rate"] * self.level * dt

    def significance(self) -> float:
        nb = len(self.counts)
        total = self.counts.sum()
        if total < 10:
            return 0.0
        centers = 100.0 + (np.arange(nb) + 0.5) * 60.0 / nb
        win = np.abs(centers - self.MASS) <= 2 * self.WIDTH + 30.0 / nb
        bkg_only = self._pdf(centers, 0.0) * (60.0 / nb)
        b = total * (1 - self.k["signal"]) * bkg_only[win].sum()
        return max(0.0, (self.counts[win].sum() - b) / math.sqrt(max(b, 1.0)))

    def layers(self, x):
        top = max(1.0, float(self.counts.max()))
        total = self.counts.sum()
        nb = len(self.counts)
        model = total * self._pdf(x, self.k["signal"]) * (60.0 / nb) / top
        return [("bars", self.counts / top, "main"), ("curve", model if total > 0 else np.full_like(x, np.nan), "aux")]

    def live(self):
        return f"N={self.counts.sum():,.0f}  bump at 125  Z={self.significance():.1f}σ"


CERN = [RFKick, Bend, Bunches, Relativity, Cold, Ramp, Beams, Squeeze, Overlap, Collision, Higgs]
CERN_ORDER = [cls.key for cls in CERN]
BY_KEY.update({cls.key: cls for cls in CERN})
GROUPS["cern"] = list(CERN_ORDER)

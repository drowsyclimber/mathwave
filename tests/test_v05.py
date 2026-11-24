"""Tests for 0.5: the CERN page (eleven accelerator panels), pages and n, the iso layer."""

import math
import os
import subprocess
import sys

import numpy as np
import pytest

from mathwave import cern, render, synth
from mathwave.cli import page_list
from mathwave.config import load
from mathwave.dsp import Analyzer
from mathwave.panels import BY_KEY, GROUPS, expand, make_panels

from test_v04 import SLIDERS, cli_env, make_app, plain, run, write_cfg

RATE = 44100
SHIPPED = os.path.join(os.path.dirname(__file__), "..", "config", "config")


def features(t=37.0):
    an = Analyzer(RATE)
    return an.analyze(synth.render(t - np.arange(an.need)[::-1] / RATE), t)


def at_level(p, level, f=None, frames=1, dt=1 / 60, **levels):
    """Run a panel with its drive level pinned, so the physics can be checked exactly."""
    f = f or features()
    if levels:
        f.levels = {**f.levels, **levels}
    p.drive_level = lambda f, dt: level
    for _ in range(frames):
        p.update(f, dt)
    return p


# ---------------------------------------------------------------- registry and text


def test_cern_page_has_all_eleven_stages_in_reel_order():
    assert GROUPS["cern"] == ["rf", "bend", "bunch", "gamma", "cold", "ramp", "beams", "squeeze", "overlap",
                              "collide", "higgs"]
    assert all(BY_KEY[k] is cls for k, cls in zip(cern.CERN_ORDER, cern.CERN))
    assert not set(GROUPS["cern"]) & set(GROUPS["math"])


def test_cern_text_fits_the_narrowest_panel():
    f = features()
    panels = make_panels(GROUPS["cern"], {})
    for i, p in enumerate(panels):
        assert len(f" {i + 1} {p.title} ") <= 25, p.title
        for line in (p.formula, p.identity, p.live()):  # before the first frame too
            assert len(line) <= 36, line
        for _ in range(30):
            p.update(f, 1 / 60)
        for line in (p.formula, p.identity, p.live()):
            assert len(line) <= 36, line


@pytest.mark.parametrize("cls", cern.CERN, ids=lambda c: c.key)
def test_cern_text_fits_at_the_slider_extremes(cls):
    f = features()
    for end in (0, 1):
        p = cls()
        for name, (lo, hi, _) in cls.RANGES.items():
            p.k[name] = (lo, hi)[end]
        for level in (0.0, 1.0):
            p.drive_level = lambda f, dt, v=level: v
            p.refresh_knobs = lambda f: None
            for _ in range(5):
                p.update(f, 1 / 60)
            assert len(p.live()) <= 36, p.live()
            for kind, *rest in p.layers(np.linspace(*p.domain, 120)):
                data = rest[0]
                arr = np.concatenate([np.ravel(a) for a in data]) if kind.startswith("iso") else np.ravel(data)
                assert not np.isinf(arr).any(), (cls.key, kind)


def test_every_cern_knob_has_a_slider_range():
    for cls in cern.CERN:
        assert cls.sound and cls.formula and cls.identity
        for name, (default, desc) in cls.KNOBS.items():
            assert desc and name in cls.RANGES, f"{cls.key}.{name}"
            lo, hi, step = cls.RANGES[name]
            assert lo <= default <= hi and 0 < step < hi - lo, f"{cls.key}.{name}"
            if name in cls.INT_KNOBS:
                assert step == int(step)


@pytest.mark.parametrize("key", cern.CERN_ORDER)
def test_tune_mode_works_on_every_cern_panel(tmp_path, key):
    app, _ = make_app(tmp_path, SLIDERS.replace("808, sinc, logistic, bell, controls", key))
    app.key("k", 0)
    s = app.tune
    assert s is not None and s.panel.key == key
    assert set(p.name for p in s.params[9:]) == set(s.panel.KNOBS)
    for i in range(len(s.params)):
        s.selected = i
        for k in ("end", "left", "home", "right", "d"):
            app.key(k, 0)
            run(app, 2)
            assert np.isfinite(s.panel.level)
    lines = plain(app.draw(150, 50, "demo"))
    assert len(lines) == 50 and all(len(line) <= 150 for line in lines)
    assert any(f"tune · {key}" in line for line in lines)


# ---------------------------------------------------------------- the physics


def test_gap_to_light_at_lhc_top_energy():
    # 6.8 TeV protons: γ ≈ 7247, so c − v ≈ c/(2γ²) ≈ 2.85 m/s.
    g = 6800 / cern.M_P
    assert g == pytest.approx(7247, rel=1e-3)
    assert cern.gap_to_light(g) == pytest.approx(2.854, rel=1e-3)
    assert cern.gap_to_light(1.0) == pytest.approx(cern.C)


def test_relativity_never_reaches_c():
    p = make_panels(["gamma"], {})[0]
    last = 0.0
    for lv in np.linspace(0, 1, 11):
        at_level(p, lv)
        v = cern.C - cern.gap_to_light(p.p["γ"])
        assert last <= v < cern.C
        last = v
    assert "c−v=" in p.live()


def test_rf_kick_follows_q_v_cos_phi():
    p = at_level(make_panels(["rf"], {})[0], 1.0)
    v0, phi = p.p["V₀"], math.radians(p.p["φ"])
    assert p.p["ΔE"] == pytest.approx(v0 * math.cos(phi), rel=1e-6)


def test_bend_holds_the_booster_radius():
    p = at_level(make_panels(["bend"], {})[0], 0.7)
    # r = p/(qB) with p in GeV/c and B in T: r = p / (0.2998 B)
    assert p.p["p"] / (0.299792458 * p.p["B"]) == pytest.approx(p.k["radius"])


def test_bunches_double():
    p = make_panels(["bunch"], {})[0]
    counts = [at_level(p, lv).p["N"] for lv in (0.0, 0.5, 1.0)]
    assert counts[0] == 1 and counts[-1] == 2 ** int(round(p.k["splits"]))
    assert all(math.log2(n).is_integer() for n in counts)


def test_cold_quenches_above_tc():
    p = make_panels(["cold"], {})[0]
    assert "superconducting" in at_level(p, 0.0).live()
    assert "QUENCH" in at_level(p, 1.0).live()
    r = p.resistance(np.array([1.9, 12.0]))
    assert r[0] < 1e-6 and r[1] > 0.5


def test_ramp_climbs_then_dumps_and_refills():
    p = make_panels(["ramp"], {})[0]
    p.refresh_knobs = lambda f: None
    p.k["ramp_s"] = 2.0
    at_level(p, 1.0, frames=60)
    assert 450 < p.e < p.k["e_top"]
    at_level(p, 1.0, frames=120)
    seen_top = p.e == p.k["e_top"] or np.nanmax(p.hist) == p.k["e_top"]
    assert seen_top
    at_level(p, 1.0, frames=90)
    assert p.e < p.k["e_top"]  # dumped and refilled from injection
    assert "/lap" in p.live() and len(p.live()) <= 36


def test_two_equal_beams_make_13_6_tev():
    p = at_level(make_panels(["beams"], {})[0], 1.0, left=0.8, right=0.8)
    assert p.p["E₁"] == pytest.approx(6800) and p.p["E₂"] == pytest.approx(6800)
    assert "E_cm=13.6 TeV" in p.live()
    q = at_level(make_panels(["beams"], {})[0], 1.0, left=0.8, right=0.2)
    assert q.p["E₁"] > q.p["E₂"]


def test_squeeze_reaches_ten_microns():
    p = at_level(make_panels(["squeeze"], {})[0], 1.0)
    assert p.p["β*"] == pytest.approx(0.3)
    assert p.p["σ*"] * 1e6 == pytest.approx(10.2, abs=0.1)
    assert at_level(p, 0.0).p["β*"] == pytest.approx(11.0)


def test_overlap_is_gaussian_in_separation():
    p = make_panels(["overlap"], {})[0]
    for lv in (0.0, 0.25, 0.5, 1.0):
        at_level(p, lv)
        d = p.k["d_max"] * (1 - lv)
        assert p.p["L"] == pytest.approx(math.exp(-d * d / 4))


def test_collision_tracks_stay_inside_the_detector():
    p = at_level(make_panels(["collide"], {})[0], 1.0)
    p.age = 10.0
    for kind, (xs, ys), style in p.layers(np.zeros(10)):
        r = np.hypot(xs, ys)
        assert np.nanmax(r) <= 1.0, style
    n0 = p.events
    p._new_event()
    assert p.events == n0 + 1 and len(p.tracks) == int(p.k["tracks_max"])


def test_higgs_significance_grows_with_events():
    p = make_panels(["higgs"], {})[0]
    p.k["memory"] = 0.0
    at_level(p, 1.0, frames=60)
    z1 = p.significance()
    at_level(p, 1.0, frames=600)
    z2 = p.significance()
    assert z2 > z1 > 0
    centers = 100 + (np.arange(len(p.counts)) + 0.5) * 2
    assert abs(centers[np.argmax(p.counts - p._pdf(centers, 0) * p.counts.sum() * 2)] - 125) <= 3


# ---------------------------------------------------------------- pages


def test_groups_expand():
    assert expand(["math"]) == GROUPS["math"]
    assert expand(["808", "cern", "higgs"]) == ["808"] + GROUPS["cern"] + ["higgs"]


def test_default_pages_are_math_then_cern():
    cfg = load(SHIPPED)
    assert page_list(cfg) == [cfg.panels, GROUPS["cern"]]
    cfg.panels = list(GROUPS["cern"])
    assert page_list(cfg)[1] == GROUPS["math"]


def test_pages_config_line(tmp_path):
    cfg = load(write_cfg(tmp_path, "[general]\npages = 808, sinc | cern | higgs, bell\n"
                                   "[panel.bell]\nexpr = exp(-x^2)\n"))
    assert cfg.pages == [["808", "sinc"], ["cern"], ["higgs", "bell"]]


def test_n_flips_pages_and_status_shows_it(tmp_path):
    app, _ = make_app(tmp_path)
    assert len(app.pages) == 2 and app.page == 0
    assert "page 1/2" in plain(app.draw(168, 50, "demo"))[0]
    app.key("2", 0)
    app.key("n", 0)
    assert app.page == 1 and app.focus is None
    assert [p.key for p in app.panels] == GROUPS["cern"]
    run(app, 5)
    text = "\n".join(plain(app.draw(168, 50, "demo")))
    assert "page 2/2" in text and "11 Higgs" in text
    app.key("N", 0)
    assert app.page == 0


def test_dot_and_comma_reach_panel_eleven(tmp_path):
    app, _ = make_app(tmp_path)
    app.key("n", 0)
    for _ in range(11):
        app.key(".", 0)
    assert app.focus == 10 and app.panels[app.focus].key == "higgs"
    app.key(".", 0)
    assert app.focus == 0
    app.key(",", 0)
    assert app.focus == 10
    app.key("k", 0)
    app.key(",", 0)
    assert app.tune is not None and app.tune.panel.key == "collide"


def test_single_page_explains_how_to_add_one(tmp_path):
    app, _ = make_app(tmp_path, "[general]\npages = 808, sinc\n")
    app.key("n", 0)
    assert app.page == 0 and "pages" in app.note


def test_page_flag_and_panels_flag(tmp_path):
    path = write_cfg(tmp_path, SLIDERS)
    env = cli_env(tmp_path)
    out = subprocess.run([sys.executable, "-m", "mathwave", "-p", path, "--page", "2", "--dump", "20",
                          "--size", "168x50", "--no-color"], capture_output=True, text=True, env=env,
                         check=True).stdout
    assert "page 2/2" in out and "squeeze" in out
    out = subprocess.run([sys.executable, "-m", "mathwave", "-p", path, "--panels", "cern", "--dump", "5",
                          "--size", "168x50", "--no-color"], capture_output=True, text=True, env=env,
                         check=True).stdout
    assert "page 1/2" in out and "1 RF kick" in out
    out = subprocess.run([sys.executable, "-m", "mathwave", "-p", path, "--page", "2", "--formulas"],
                         capture_output=True, text=True, env=env, check=True).stdout
    assert "σ* = √(εβ*)" in out and "m_H = 125 GeV" in out
    out = subprocess.run([sys.executable, "-m", "mathwave", "--list-panels"], capture_output=True, text=True,
                         env=env, check=True).stdout
    assert "CERN page" in out and "higgs" in out
    out = subprocess.run([sys.executable, "-m", "mathwave", "--keys"], capture_output=True, text=True,
                         env=env, check=True).stdout
    assert "CERN page" in out


@pytest.mark.parametrize("size", [(168, 50), (120, 40), (100, 40), (80, 30)])
def test_cern_page_fits_any_terminal(tmp_path, size):
    cols, rows = size
    app, _ = make_app(tmp_path)
    app.key("n", 0)
    run(app, 5)
    for keys in ([], ["."], ["k"], ["k", "down", "down", "down"], ["m"]):
        app.key("g", 0)
        for k in keys:
            app.key(k, 0)
        lines = plain(app.draw(cols, rows, "demo"))
        assert len(lines) == rows and all(len(line) <= cols for line in lines), keys


def test_hint_line_fits_with_the_status():
    assert len(render.HINTS) <= 100 and "n page" in render.HINTS


# ---------------------------------------------------------------- iso layer


def test_plot_iso_draws_a_round_circle():
    grid = np.zeros((80, 160), dtype=bool)
    th = np.linspace(0, 2 * np.pi, 400)
    render.plot_iso(grid, *cern.ring(th, np.full_like(th, 1.0)))
    rows, cols = np.nonzero(grid)
    assert np.ptp(cols) == pytest.approx(np.ptp(rows), abs=2)  # as wide as tall, in dots
    assert abs(cols.mean() - 79.5) < 1.5 and abs(rows.mean() - 39.5) < 1.5


def test_plot_iso_nan_lifts_the_pen():
    grid = np.zeros((40, 40), dtype=bool)
    render.plot_iso(grid, np.array([-0.9, -0.5, np.nan, 0.5, 0.9]), np.zeros(5))
    row = grid[grid.any(axis=1)][0]
    assert not row[19:21].any() and row.sum() > 6


def test_ring_puts_zero_degrees_at_the_top():
    x, y = cern.ring(np.array([0.0, np.pi / 2]), 1.0)
    assert x[0] == pytest.approx(0) and y[0] == pytest.approx(1)
    assert x[1] == pytest.approx(1) and y[1] == pytest.approx(0, abs=1e-12)

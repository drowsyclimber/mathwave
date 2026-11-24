"""Tests for 0.6: the black hole page. A lensed disk over every trace, the real fall clock,
and a PHYSICS / MIXED / METAPHOR label on every panel."""

import math
import subprocess
import sys

import numpy as np
import pytest

from mathwave import blackhole as bh
from mathwave import synth
from mathwave.dsp import Analyzer
from mathwave.panels import BY_KEY, GROUPS, make_panels

from test_v04 import SLIDERS, cli_env, make_app, plain, run, write_cfg

RATE = 44100


def features(t=37.0):
    an = Analyzer(RATE)
    return an.analyze(synth.render(t - np.arange(an.need)[::-1] / RATE), t)


# ---------------------------------------------------------------- registry, labels, text


def test_hole_page_follows_the_reel():
    assert GROUPS["hole"] == ["horizon", "sos", "light", "engines", "noway", "tides", "headfeet", "crush",
                              "spaghetti", "singular"]
    assert all(BY_KEY[k] is cls for k, cls in zip(bh.HOLE_ORDER, bh.HOLE))
    assert not set(GROUPS["hole"]) & (set(GROUPS["math"]) | set(GROUPS["cern"]))


def test_every_panel_is_labeled():
    verdicts = {cls.key: cls.verdict for cls in bh.HOLE}
    assert set(verdicts.values()) == {"PHYSICS", "MIXED", "METAPHOR"}
    assert verdicts["singular"] == "PHYSICS"
    assert {k for k, v in verdicts.items() if v == "METAPHOR"} == {"sos", "engines", "noway", "crush"}
    for cls in bh.HOLE:
        assert cls.truth.startswith("Real:") and "Made up:" in cls.truth, cls.key
        assert cls.sound and "—" not in cls.truth + cls.sound


@pytest.mark.parametrize("cls", bh.HOLE, ids=lambda c: c.key)
def test_text_fits_everywhere(cls):
    f = features()
    p = cls()
    i = bh.HOLE.index(cls)
    assert len(f" {i + 1} {p.title} ") <= 25
    for lines in ([p.formula, p.identity, p.live()],):
        assert all(len(s) <= 36 for s in lines), lines
    for end in (0, 1):
        for name, (lo, hi, _) in cls.RANGES.items():
            p.k[name] = (lo, hi)[end]
        p.refresh_knobs = lambda f: None
        for level, frac in ((0.0, 0.0), (1.0, 0.5), (1.0, 0.99999)):
            p.drive_level = lambda f, dt, v=level: v
            p.frac = frac
            for _ in range(3):
                p.update(f, 1 / 60)
            assert len(p.live()) <= 36, p.live()
            x = np.linspace(*p.domain, 160)
            for kind, *rest in p.layers(x) + p.scene():
                data = rest[0]
                arr = np.concatenate([np.ravel(a) for a in data]) if kind.startswith("iso") else np.ravel(data)
                assert not np.isinf(arr).any(), (cls.key, kind)


def test_every_hole_knob_has_a_slider_range():
    for cls in bh.HOLE:
        assert {"mass", "tilt"} <= set(cls.KNOBS)
        for name, (default, desc) in cls.KNOBS.items():
            assert desc and name in cls.RANGES, f"{cls.key}.{name}"
            lo, hi, step = cls.RANGES[name]
            assert lo <= default <= hi and 0 < step < hi - lo, f"{cls.key}.{name}"


@pytest.mark.parametrize("key", bh.HOLE_ORDER)
def test_tune_mode_works_on_every_hole_panel(tmp_path, key):
    app, _ = make_app(tmp_path, SLIDERS.replace("808, sinc, logistic, bell, controls", key))
    app.key("k", 0)
    s = app.tune
    assert s is not None and s.panel.key == key
    for i in range(len(s.params)):
        s.selected = i
        for k in ("end", "left", "home", "right", "d"):
            app.key(k, 0)
            run(app, 2)
            assert np.isfinite(s.panel.level)
    lines = plain(app.draw(150, 50, "demo"))
    assert len(lines) == 50 and all(len(line) <= 150 for line in lines)


# ---------------------------------------------------------------- the fall


def test_sgr_a_star_gives_the_reels_67_seconds():
    assert bh.fall_time(bh.SGR_A) == pytest.approx(66.49, abs=0.05)
    gm, rs, t = bh.hole(bh.SGR_A)
    assert rs / 1000 == pytest.approx(1.269e7, rel=1e-3)
    assert bh.fall_time(bh.SGR_A) == pytest.approx(math.pi * t)


def test_cycloid_ends():
    assert bh.eta_of(0.0) == pytest.approx(0.0, abs=1e-9)
    assert bh.eta_of(1.0) == pytest.approx(math.pi, abs=1e-9)
    eta = bh.eta_of(np.linspace(0, 1, 50))
    assert np.all(np.diff(eta) > 0)
    assert np.allclose((eta + np.sin(eta)) / math.pi, np.linspace(0, 1, 50), atol=1e-9)
    assert float(bh.r_of_eta(0.0)) == 1.0 and float(bh.r_of_eta(math.pi)) == pytest.approx(0.0, abs=1e-12)


def test_radius_runs_out_as_two_thirds_power():
    # Near r = 0: r/r_s ≈ (6π·(1 − frac))^(2/3) / 4, so r ∝ (time left)^(2/3).
    for left in (1e-3, 1e-4, 1e-5):
        r = float(bh.r_of_eta(bh.eta_of(1 - left)))
        assert r == pytest.approx((6 * math.pi * left) ** (2 / 3) / 4, rel=0.02)
    a = float(bh.r_of_eta(bh.eta_of(1 - 1e-4)))
    b = float(bh.r_of_eta(bh.eta_of(1 - 1e-5)))
    assert a / b == pytest.approx(10 ** (2 / 3), rel=1e-2)


def test_tides_in_the_truth_texts():
    gm, rs, _ = bh.hole(bh.SGR_A)
    assert bh.tidal_g(gm, rs) == pytest.approx(1.14e-4, rel=0.02)   # "about 1e-4 g" at the horizon
    r1 = bh.r_at_tide(gm, 1.0)
    assert r1 / 1000 == pytest.approx(615_000, rel=0.01)             # "about 615,000 km"
    eta = 2 * math.acos(math.sqrt(r1 / rs))
    left = bh.fall_time(bh.SGR_A) - bh.hole(bh.SGR_A)[2] * (eta + math.sin(eta))
    assert left == pytest.approx(0.3, abs=0.02)                       # "0.3 s before the end"
    assert bh.tidal_g(gm, 0.0) == math.inf


def test_cot_half_eta_is_a_radius_ratio():
    for r in (0.9, 0.5, 0.1):
        eta = 2 * math.acos(math.sqrt(r))
        assert 1 / math.tan(eta / 2) == pytest.approx(math.sqrt(r / (1 - r)))


def test_panels_share_one_clock_and_refill():
    f = features()
    ps = make_panels(GROUPS["hole"], {})
    for _ in range(120):
        for p in ps:
            p.update(f, 1 / 60)
    assert len({round(p.tau_left, 6) for p in ps}) == 1
    assert ps[0].tau_left == pytest.approx(bh.fall_time(bh.SGR_A) - 2.0, abs=0.05)
    p = ps[0]
    p.frac = 1.0
    for _ in range(130):
        p.update(f, 1 / 60)
    assert p.frac < 0.1 and p.r_rs > 0.99  # the next one falls in


def test_mass_knob_changes_the_clock():
    p = make_panels(["singular"], {}, {"singular": {"mass": "1"}})[0]
    p.update(features(), 1 / 60)
    assert p.tau_max == pytest.approx(66.49 / 4.297, rel=1e-3)


# ---------------------------------------------------------------- the image


def test_far_side_is_lifted_over_the_shadow():
    for (X, Y, shift), r in zip(bh.disk_image(80.0), bh.RADII):
        b = np.hypot(X, Y)
        far = np.sin(np.linspace(0, 2 * np.pi, len(X))) < -0.5
        assert np.all(b[far] > bh.B_CRIT)           # far side light lands outside the shadow
        assert Y[far].max() == pytest.approx(2 * math.sqrt(r), rel=0.05)  # b = 2√(r·r_s) behind the hole
        assert Y[~far].min() < 0                     # the near side crosses in front, below center


def test_doppler_side_and_isco_speed():
    (X, Y, shift), = bh.disk_image(80.0, (3.0,))
    assert shift[X > 0].max() > 1.2 and shift[X < 0].min() < 0.6
    u = 1 / 3
    assert math.sqrt(u / (2 * (1 - u))) == pytest.approx(0.5)  # c/2 at the innermost stable orbit


def test_secondary_image_sits_under_the_shadow():
    x, y = bh.secondary_image(80.0)
    ok = np.isfinite(y)
    assert ok.any() and np.all(y[ok] < 0)
    assert np.all(np.hypot(x[ok], y[ok]) >= bh.B_CRIT - 1e-9)
    assert bh.B_CRIT == pytest.approx(2.598, abs=1e-3)


def test_sos_gate_spells_sos():
    u = np.arange(0, 34) + 0.5
    on = bh._sos_gate(u)
    runs, cur, n = [], on[0], 0
    for v in on:
        if v == cur:
            n += 1
        else:
            runs.append((cur, n))
            cur, n = v, 1
    runs.append((cur, n))
    marks = [n for v, n in runs if v]
    assert marks == [1, 1, 1, 3, 3, 3, 1, 1, 1]
    assert on.sum() == 15


def test_tides_wave_ends_at_r_zero():
    p = make_panels(["tides"], {})[0]
    p.frac = 1.0 - 1.0 / p.tau_max  # one second left
    p._fall()
    y = p._wave(np.linspace(-6, 2, 801))
    x = np.linspace(-6, 2, 801)
    assert np.isfinite(y[x <= 0.99]).all() and np.isnan(y[x > 1.01]).all()


# ---------------------------------------------------------------- on screen


def test_scene_badge_and_truth_on_screen(tmp_path):
    app, _ = make_app(tmp_path)
    app.key("n", 0)
    app.key("n", 0)
    run(app, 20)
    lines = plain(app.draw(168, 50, "demo"))
    text = "\n".join(lines)
    assert "page 3/3" in text and "PHYSICS" in text and "METAPHOR" in text and "MIXED" in text
    top = "\n".join(lines[:12])
    assert any(ch in top for ch in "⣿⠿⡇⢸")  # the disk is drawn above the traces
    app.key("0", 0)
    text = "\n".join(plain(app.draw(140, 50, "demo")))
    assert "physics check, PHYSICS: Real: near r = 0" in text


@pytest.mark.parametrize("size", [(168, 50), (120, 40), (100, 40), (80, 30)])
def test_hole_page_fits_any_terminal(tmp_path, size):
    cols, rows = size
    app, _ = make_app(tmp_path, "[general]\npanels = hole\n")
    run(app, 5)
    for keys in ([], ["."], [",", ","], ["k"], ["m"]):
        app.key("g", 0)
        for k in keys:
            app.key(k, 0)
        lines = plain(app.draw(cols, rows, "demo"))
        assert len(lines) == rows and all(len(line) <= cols for line in lines), keys


def test_cli_lists_and_prints_the_hole_page(tmp_path):
    env = cli_env(tmp_path)
    out = subprocess.run([sys.executable, "-m", "mathwave", "--list-panels"], capture_output=True, text=True,
                         env=env, check=True).stdout
    assert "Black hole page" in out and "singular" in out and "[PHYSICS]" in out
    path = write_cfg(tmp_path, SLIDERS)
    out = subprocess.run([sys.executable, "-m", "mathwave", "-p", path, "--page", "3", "--formulas"],
                         capture_output=True, text=True, env=env, check=True).stdout
    assert out.count("physics check:") == 10 and "physics check: METAPHOR" in out
    out = subprocess.run([sys.executable, "-m", "mathwave", "-p", path, "--panels", "hole", "--dump", "10",
                          "--size", "168x50", "--no-color"], capture_output=True, text=True, env=env,
                         check=True).stdout
    assert "1 horizon" in out and "10 singularity" in out


def test_no_knob_shadows_a_shared_tuning_key():
    from mathwave.cern import CERN
    from mathwave.panels import BUILTIN, COMMON_KEYS
    for cls in BUILTIN + CERN + bh.HOLE:
        assert not set(cls.KNOBS) & set(COMMON_KEYS), cls.key


# ---------------------------------------------------------------- 0.7: a bigger, busier black hole


def test_lens_images_solve_the_lens_equation():
    bx, by = np.array([0.5, 4.0, -9.0]), np.array([1.0, -2.0, 0.3])
    (ox, oy), (ix, iy) = bh.lens(bx, by, 3.3)
    beta = np.hypot(bx, by)
    for x, y in ((ox, oy), (ix, iy)):
        th = np.hypot(x, y) * np.sign(x * bx + y * by)
        assert np.allclose(th - 3.3**2 / th, beta)
    assert np.all(np.hypot(ox, oy) > 3.3) and np.all(np.hypot(ix, iy) < 3.3)


def test_project_matches_the_ring_images():
    phi = np.linspace(0, 2 * np.pi, 181)
    X, Y, g = bh.project(np.full_like(phi, 4.9), phi, 80.0)
    ring = bh.disk_image(80.0, (4.9,), 181)[0]
    assert np.allclose(X, ring[0]) and np.allclose(Y, ring[1]) and np.allclose(g, ring[2])


def _bounds(layers):
    xs = np.concatenate([np.ravel(v[0]) for _, v, _ in layers])
    ys = np.concatenate([np.ravel(v[1]) for _, v, _ in layers])
    ok = np.isfinite(xs) & np.isfinite(ys) & (np.abs(xs) < 1.2)
    return ys[ok].min(), ys[ok].max()


def test_scene_fills_the_height_and_is_batched():
    p = make_panels(["horizon"], {})[0]
    p.update(features(), 1 / 60)
    ys = np.concatenate([(Y - bh.OFFSET) / bh.SCALE for _, Y, _ in bh.disk_image(80.0, bh.FINE_RADII)])
    ring_lo = (-bh.B_CRIT - bh.OFFSET) / bh.SCALE
    hi, lo = ys.max(), min(ys.min(), ring_lo)
    assert 0.8 < hi <= 1.0 and -1.0 <= lo < -0.75  # centered, using the whole height
    lo2, hi2 = _bounds([layer for layer in p.scene(96, 300) if layer[0] == "iso"])
    assert lo2 >= -1.1 and hi2 <= 1.1
    assert len(p.scene(96, 300)) <= 6  # a few long polylines, not hundreds of short ones


def test_big_views_get_more_detail():
    p = make_panels(["horizon"], {})[0]
    p.update(features(), 1 / 60)
    count = lambda layers: sum(np.isfinite(np.ravel(v[0])).sum() for _, v, _ in layers)  # noqa: E731
    assert count(p.scene(120, 300)) > 1.4 * count(p.scene(24, 90))


def test_the_disk_moves():
    f = features()
    p = make_panels(["horizon"], {})[0]
    p.update(f, 1 / 60)
    c0, g0, s0 = p.clump_phi.copy(), p.gas_r.copy(), p.star_x.copy()
    for _ in range(30):
        p.update(f, 1 / 60)
    turned = (p.clump_phi - c0) % (2 * np.pi)
    assert np.all(turned > 0)
    inner, outer = np.argmin(p.clump_r), np.argmax(p.clump_r)
    assert turned[inner] > turned[outer]  # Kepler: inner clumps orbit faster
    assert not np.allclose(p.gas_r, g0) and not np.allclose(p.star_x, s0)


def test_gas_plunges_inside_the_innermost_orbit():
    f = features()
    p = make_panels(["horizon"], {})[0]
    p.update(f, 1 / 60)
    p.gas_r[:] = np.linspace(2.0, 6.0, len(p.gas_r))
    before = p.gas_r.copy()
    p.update(f, 1 / 60)
    drop = before - p.gas_r
    assert drop[before < 3].min() > 5 * drop[before > 3].max()
    p.gas_r[:] = bh.horizons(0.0)[0] / 2 + 0.041  # just outside the horizon, at 1 r_s
    p.update(f, 1 / 60)
    assert np.all(p.gas_r == bh.R_OUT)  # swallowed gas is replaced at the outer edge


def test_kick_flares_the_photon_ring():
    f = features()
    p = make_panels(["horizon"], {})[0]
    p.update(f, 1 / 60)
    p.level, p.onset, p.flare = 1.0, True, 0.0
    p._animate(1 / 60)
    assert p.flare == 1.0
    p.onset = False
    for _ in range(60):
        p._animate(1 / 60)
    assert p.flare < 0.1


def test_focus_gives_the_black_hole_more_rows(tmp_path):
    app, _ = make_app(tmp_path, "[general]\npanels = hole\n")
    run(app, 10)
    grid = plain(app.draw(168, 56, "demo"))
    app.key("1", 0)
    focus = plain(app.draw(168, 56, "demo"))
    dots = lambda lines: sum(1 for line in lines for ch in line if 0x2800 < ord(ch) <= 0x28FF)  # noqa: E731
    assert dots(focus[:30]) > dots(grid[:12])

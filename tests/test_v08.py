"""Tests for 0.8: a spinning (Kerr) black hole on the black hole page, set by the spin knob."""

import math

import numpy as np
import pytest

from mathwave import blackhole as bh
from mathwave import synth
from mathwave.dsp import Analyzer
from mathwave.panels import GROUPS, make_panels

from test_v04 import make_app, plain, run

RATE = 44100


def features(t=37.0):
    an = Analyzer(RATE)
    return an.analyze(synth.render(t - np.arange(an.need)[::-1] / RATE), t)


def spun(key, spin, frames=2):
    p = make_panels([key], {}, {key: {"spin": str(spin)}})[0]
    f = features()
    for _ in range(frames):
        p.update(f, 1 / 60)
    return p


# ---------------------------------------------------------------- the Kerr formulas


def test_horizons():
    assert bh.horizons(0.0) == (2.0, 0.0)
    rp, rm = bh.horizons(0.6)
    assert rp == pytest.approx(1.8) and rm == pytest.approx(0.2)
    assert bh.horizons(0.998)[0] == pytest.approx(1.0632, abs=1e-3)


@pytest.mark.parametrize("a, r", [(0.0, 6.0), (0.5, 4.233), (0.9, 2.321), (0.998, 1.237), (-0.998, 8.994),
                                  (-1e-9, 6.0)])
def test_innermost_stable_orbit(a, r):
    assert bh.isco(a) == pytest.approx(r, abs=2e-3)


def test_orbits_and_frame_dragging():
    r = np.array([3.0, 6.0, 10.0])
    assert np.allclose(bh.kepler_omega(r, 0.0), r**-1.5)
    assert np.allclose(bh.frame_drag(r, 0.0), 0.0)
    for a in (0.3, 0.9):
        rp = bh.horizons(a)[0]
        assert float(bh.frame_drag(rp, a)) == pytest.approx(a / (2 * rp))   # the horizon's own spin, Ω_H
        assert float(bh.frame_drag(rp, -a)) == pytest.approx(-a / (2 * rp))


def test_orbit_shift_reduces_to_schwarzschild():
    r, lam = 8.0, np.array([-5.0, 0.0, 5.0])
    ut = 1 / math.sqrt(1 - 3 / r)
    assert np.allclose(bh.orbit_shift(r, lam, 0.0), 1 / (ut * (1 - r**-1.5 * lam)))
    assert bh.orbit_shift(r, 5.0, 0.0) > 1 > bh.orbit_shift(r, -5.0, 0.0)


def test_no_spin_shadow_is_the_old_circle():
    x, y = bh.critical_curve(0.0, 80.0)
    assert np.allclose(np.hypot(x, y), bh.B_CRIT)


def test_spinning_shadow_is_flattened_and_shifted():
    x, y = bh.critical_curve(0.99, 89.5, 2001)
    # Bardeen 1973, edge-on, a = 0.99: from −6.98 M to +2.25 M across, 2√27 M tall (in r_s, half).
    assert x.min() == pytest.approx(-6.98 / 2, abs=0.03) and x.max() == pytest.approx(2.25 / 2, abs=0.05)
    assert y.max() == pytest.approx(bh.B_CRIT, abs=0.01) and y.min() == pytest.approx(-bh.B_CRIT, abs=0.01)
    xm, _ = bh.critical_curve(-0.99, 89.5, 2001)
    assert xm.max() == pytest.approx(-x.min(), abs=0.03)  # the other spin is the mirror image
    # The flat side faces the side of the disk moving toward you, which is the bright side.
    (X, _, g), = bh.disk_image(80.0, (bh.isco(0.9) / 2 + 0.5,), 181, spin=0.9)
    assert g[X > 0].max() > 1.0 > g[X < 0].min()
    assert abs(bh.critical_curve(0.9, 80.0)[0].max()) < abs(bh.critical_curve(0.9, 80.0)[0].min())


def test_crit_radius_follows_the_curve():
    assert np.allclose(bh.crit_radius(0.0, 80.0, np.linspace(-3, 3, 9)), bh.B_CRIT)
    assert bh.crit_radius(0.95, 85.0, 0.0) < bh.crit_radius(0.95, 85.0, math.pi)


# ---------------------------------------------------------------- the fall with spin


def test_fall_time_with_spin():
    assert bh.fall_table(0.0)[2][-1] == pytest.approx(math.pi, abs=1e-6)  # same as the cycloid
    assert bh.fall_table(0.998)[2][-1] == pytest.approx(4.4396, abs=1e-3)
    assert bh.fall_time(bh.SGR_A, 0.0) == pytest.approx(66.49, abs=0.05)
    assert bh.fall_time(bh.SGR_A, 0.9) == pytest.approx(90.54, abs=0.05)
    assert bh.fall_time(bh.SGR_A, -0.9) == pytest.approx(bh.fall_time(bh.SGR_A, 0.9))


def test_fall_ends_at_the_inner_horizon():
    a = 0.6
    _, r0 = bh.fall_state(0.0, a)
    _, r1 = bh.fall_state(1.0, a)
    assert r0 == pytest.approx(bh.horizons(a)[0]) and r1 == pytest.approx(bh.horizons(a)[1])
    t, r, tau = bh.fall_table(a)
    rm = bh.horizons(a)[1]
    left = tau[-1] - tau
    k = (r - rm) / np.maximum(left, 1e-12) ** 2   # r − r₋ ∝ (τ₀ − τ)² near the end
    near = (left > 1e-3) & (left < 0.03)
    assert np.ptp(k[near]) / k[near].mean() < 0.02


def test_no_spin_fall_matches_the_closed_form():
    for frac in (0.1, 0.5, 0.9):
        eta, r = bh.fall_state(frac, 0.0)
        assert r == pytest.approx(2 * math.cos(eta / 2) ** 2)


def test_axis_tide():
    r = np.array([1.0, 2.0, 5.0])
    assert np.allclose(bh.tide_axis(r, 0.0), 2 / r**3)
    a = 0.5
    assert bh.tide_axis(math.sqrt(3) * a, a) == pytest.approx(0.0, abs=1e-12)  # flips at r = √3·a
    assert bh.tide_axis(0.5, a) < 0 < bh.tide_axis(1.5, a)


def test_spinning_sgr_a_star_never_spaghettifies_you():
    for a in (0.5, 0.9):
        p = spun("spaghetti", a)
        stretch, squeeze = p.tide_extremes()
        assert max(stretch, squeeze) < 0.01  # thousandths of a g
        assert "never spaghettifies" in p.truth
    assert "only squeezes" in spun("tides", 0.9).truth          # r₊ < √3·a: the whole path squeezes
    assert "stretches you at most" in spun("tides", 0.5).truth


# ---------------------------------------------------------------- the panels


def test_every_panel_has_the_spin_knob():
    from mathwave.blackhole import HOLE
    for cls in HOLE:
        assert "spin" in cls.KNOBS and cls.KNOBS["spin"][0] == 0.0
        lo, hi, step = cls.RANGES["spin"]
        assert lo < 0 < hi and step == pytest.approx(0.01)


def test_spin_moves_the_disk_edge():
    assert spun("horizon", 0.0).disk_radii(True)[0] == pytest.approx(3.0)
    assert spun("horizon", 0.9).disk_radii(True)[0] == pytest.approx(bh.isco(0.9) / 2)
    assert spun("horizon", -0.9).disk_radii(True)[0] == pytest.approx(bh.isco(-0.9) / 2)
    radii = spun("horizon", -0.9).disk_radii(True)
    assert min(np.diff(radii)) > 0.45  # the narrower disk keeps its rings apart


def test_counter_spinning_hole_turns_the_gas_around():
    p = spun("horizon", -0.9)
    near = p.r_plus / 2 + 0.02
    assert float(p.gas_omega(near)) < 0 < float(p.gas_omega(5.0))
    q = spun("horizon", 0.9)
    assert float(q.gas_omega(q.r_plus / 2 + 0.02)) > 0


def test_singularity_label_follows_the_spin():
    assert spun("singular", 0.0).verdict == "PHYSICS"
    p = spun("singular", 0.7)
    assert p.verdict == "MIXED" and "inner horizon" in p.truth
    assert spun("horizon", 0.7).truth != spun("horizon", 0.0).truth


@pytest.mark.parametrize("spin", [-0.99, -0.5, 0.5, 0.99])
def test_spinning_scene_draws_cleanly(spin):
    p = spun("horizon", spin, 20)
    for gh, gw in ((24, 90), (120, 300)):
        for kind, (x, y), style in p.scene(gh, gw):
            assert not np.isinf(x).any() and not np.isinf(y).any()
            ok = np.isfinite(x) & np.isfinite(y)
            assert np.abs(y[ok]).max() < 2.0


def test_spin_on_screen(tmp_path):
    text = "[general]\npanels = hole\n" + "".join(f"[tune.{k}]\nspin = 0.9\n" for k in GROUPS["hole"])
    app, _ = make_app(tmp_path, text)
    run(app, 10)
    lines = plain(app.draw(168, 56, "demo"))
    assert len(lines) == 56 and all(len(line) <= 168 for line in lines)
    app.key("0", 0)
    focus = "\n".join(plain(app.draw(140, 52, "demo")))
    assert "inner horizon" in focus and "spin=0.9" in focus


def test_spinning_cycloid_angle():
    a = 0.7
    rp, rm = bh.horizons(a)
    for frac in (0.2, 0.5, 0.8):
        eta, r = bh.fall_state(frac, a)
        assert 1 / math.tan(eta / 2) == pytest.approx(math.sqrt((r - rm) / (rp - r)), rel=1e-6)
    p = spun("light", a)
    assert "r₋" in p.truth and "cycloid angle" not in p.sound
    assert "inner one" in spun("horizon", a).sound and "cycloid" in spun("horizon", 0.0).sound


def test_no_way_out_points_at_the_inner_horizon_when_spinning():
    p = spun("noway", 0.8)
    assert "r₋" in p.live() and "inner horizon" in p.truth and p.verdict == "METAPHOR"
    assert "r=0" in spun("noway", 0.0).live()


def test_identities_follow_the_spin():
    assert "r₊ to r₋" in spun("horizon", 0.5).identity and "1 to 0" in spun("horizon", 0.0).identity
    assert "r₋" in spun("noway", -0.5).identity
    for key in GROUPS["hole"]:
        p = spun(key, 0.99)
        assert len(p.identity) <= 36 and len(p.live()) <= 36

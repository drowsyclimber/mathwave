"""Tests for 0.4 manual controls: [slider.*], the controls panel and view, tune mode on
every panel, writing tuning back to the config, custom-panel knobs, keys and mouse."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mathwave.audio import OfflineDemoSource
from mathwave.cli import App, drive_choices
from mathwave.config import load
from mathwave.controls import (Mouse, SliderBank, SliderSpec, fmt, parse_input, parse_slider,
                               slider_usage, snap, update_section)
from mathwave.panels import BUILTIN, Controls, make_panels

RATE = 44100
SHIPPED = os.path.join(os.path.dirname(__file__), "..", "config", "config")


def cli_env(tmp_path):
    return dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(__file__), "..", "src"),
                XDG_CONFIG_HOME=str(tmp_path / "xdg"), HOME=str(tmp_path / "home"),
                XDG_STATE_HOME=str(tmp_path / "state"))


def write_cfg(tmp_path, text, name="config"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


SLIDERS = """
[general]
panels = 808, sinc, logistic, bell, controls

[slider.punch]
range = 0-10
default = 5
label = kick weight

[slider.chaos]
range = 0-10
default = 3

[panel.bell]
expr = A*exp(-x^2/(2*s^2))
params = A = 0.15 + height*level, s = 0.06 + spread*level
knobs = height = 0.85 (0..1), spread = 0.3 (0.05..1)
drive = bass

[tune.808]
gain = 0.5 + punch/5

[tune.logistic]
r_min = 2.5 + chaos/10
"""


def args_for(path, **kw):
    base = dict(config=path, profile=None, method=None, demo=False, source=None, fifo=None, panels=None,
                layout=None, theme=None, fps=None, sensitivity=None, fft_size=None, bpm=None, no_color=True,
                no_mouse=False, set=None)
    base.update(kw)
    return SimpleNamespace(**base)


def make_app(tmp_path, text=SLIDERS, frames=10, **kw):
    path = write_cfg(tmp_path, text)
    a = args_for(path)
    from mathwave.cli import load_config
    app = App(load_config(a), a, **kw)
    run(app, frames)
    return app, path


def run(app, frames, t0=30.0):
    src = OfflineDemoSource(RATE)
    for k in range(frames):
        t = t0 + k / 60
        app.step(src.latest_stereo(app.an.need, t), t, 1 / 60)


def plain(lines):
    import re
    return [re.sub(r"\x1b\[[0-9;]*m", "", line) for line in lines]


# ---------------------------------------------------------------- sliders in the config


def test_parse_slider_defaults_and_values():
    s = parse_slider("punch", {"range": "0-10", "default": "5", "label": "kick weight"}, set())
    assert (s.lo, s.hi, s.step, s.default, s.label) == (0, 10, 0.1, 5, "kick weight")
    s = parse_slider("pan", {"range": "-1..1", "step": "0.05"}, set())
    assert (s.lo, s.hi, s.step, s.default) == (-1, 1, 0.05, -1)
    assert parse_slider("x2", {}, set()).hi == 10


@pytest.mark.parametrize("name,sec,msg", [
    ("bass", {}, "already a drive"),
    ("width", {}, "already a drive"),
    ("Punch", {}, "lowercase"),
    ("p", {"range": "10-0"}, "LO must be below"),
    ("p", {"range": "ten"}, "use LO-HI"),
    ("p", {"range": "0-10", "default": "11"}, "outside"),
    ("p", {"step": "0"}, "step"),
    ("p", {"rnage": "0-1"}, "unknown key 'rnage'"),
])
def test_bad_sliders(name, sec, msg):
    from mathwave.config import _RESERVED
    from mathwave.dsp import BANDS
    with pytest.raises(ValueError, match=msg):
        parse_slider(name, sec, _RESERVED | set(BANDS))


def test_sliders_become_drive_names_and_shape_expressions(tmp_path):
    app, _ = make_app(tmp_path)
    assert {"punch", "chaos"} <= app.cfg.names
    p808 = app.panels[0]
    assert p808.tune.gain == pytest.approx(1.5)  # 0.5 + 5/5
    app.bank.set("punch", 10)
    run(app, 1)
    assert p808.tune.gain == pytest.approx(2.5)
    logistic = app.panels[2]
    assert logistic.k["r_min"] == pytest.approx(2.8)
    assert "gain=0.5+punch/5" in p808.tuning_summary()


def test_slider_usage_lists_every_reader(tmp_path):
    app, _ = make_app(tmp_path)
    assert app.usage == {"punch": ["808 gain"], "chaos": ["logistic r_min"]}
    assert slider_usage(app.panels, []) == {}


def test_shape_expression_is_clamped(tmp_path):
    app, _ = make_app(tmp_path, SLIDERS.replace("gain = 0.5 + punch/5", "threshold = punch"))
    assert app.panels[0].tune.threshold == pytest.approx(0.99)


def test_controls_key_and_custom_panel_named_controls_are_rejected(tmp_path):
    with pytest.raises(SystemExit, match="does not match"):
        make_panels(["808"], {}, {"controls": {"gain": "2"}})
    with pytest.raises(SystemExit, match="built-in slider panel"):
        load(write_cfg(tmp_path, "[panel.controls]\nexpr = x\n"))


# ---------------------------------------------------------------- the slider bank


def test_snap_and_fmt():
    assert snap(5.04, 0, 10, 0.1) == pytest.approx(5.0)
    assert snap(12, 0, 10, 0.1) == 10
    assert fmt(0.30000000000000004, 0.01) == "0.3"
    assert fmt(3.0, 1) == "3"
    assert fmt(-0.0, 0.1) == "0"


def test_bank_moves_resets_and_normalizes():
    b = SliderBank({"p": SliderSpec("p", 0, 10, 0.1, 5)})
    b.nudge("p", 10)
    assert b.values["p"] == pytest.approx(6.0) and b.dirty
    b.nudge("p", 100)
    assert b.values["p"] == 10 and b.norm("p") == 1
    b.reset("p")
    assert b.values["p"] == 5


def test_bank_persists_per_profile(tmp_path):
    state = str(tmp_path / "state" / "sliders.json")
    specs = {"p": SliderSpec("p", 0, 10, 0.1, 5)}
    b = SliderBank(specs, None, state)
    b.set("p", 7.3)
    b.save()
    c = SliderBank(specs, "phonk", state)
    c.set("p", 1.0)
    c.save()
    assert SliderBank(specs, None, state).values["p"] == pytest.approx(7.3)
    assert SliderBank(specs, "phonk", state).values["p"] == pytest.approx(1.0)
    data = json.load(open(state))
    assert set(data["profiles"]) == {"default", "phonk"}
    # a slider whose range shrank is clamped, an unknown one ignored
    data["profiles"]["default"] = {"p": 99, "gone": 3}
    json.dump(data, open(state, "w"))
    assert SliderBank(specs, None, state).values["p"] == 10


def test_set_flag_and_unknown_slider(tmp_path):
    path = write_cfg(tmp_path, SLIDERS)
    out = subprocess.run([sys.executable, "-m", "mathwave", "-p", path, "--dump", "5", "--set", "punch=9",
                          "--controls", "--size", "120x30", "--no-color"],
                         capture_output=True, text=True, env=cli_env(tmp_path), check=True).stdout
    assert "punch" in out and " 9" in out and "used by 808 gain" in out
    bad = subprocess.run([sys.executable, "-m", "mathwave", "-p", path, "--dump", "5", "--set", "nope=1"],
                         capture_output=True, text=True, env=cli_env(tmp_path))
    assert bad.returncode != 0 and "no slider 'nope'" in bad.stderr


# ---------------------------------------------------------------- tune mode on every panel


def test_every_builtin_knob_has_a_slider_range():
    for cls in BUILTIN:
        for name, (default, _) in cls.KNOBS.items():
            assert name in cls.RANGES, f"{cls.key}.{name}"
            lo, hi, step = cls.RANGES[name]
            assert lo <= default <= hi and 0 < step < hi - lo, f"{cls.key}.{name}"
            if name in cls.INT_KNOBS:
                assert step == int(step)


ALL_PANELS = [cls.key for cls in BUILTIN] + ["bell"]


@pytest.mark.parametrize("key", ALL_PANELS)
def test_tune_mode_moves_every_param_on_every_panel(tmp_path, key):
    app, _ = make_app(tmp_path, SLIDERS.replace("808, sinc, logistic, bell, controls", key))
    app.key("k", 0)
    s = app.tune
    assert s is not None and s.panel.key == key
    names = [p.name for p in s.params]
    assert names[:9] == ["drive", "gain", "threshold", "curve", "min", "max", "attack_ms", "decay_ms", "hold_ms"]
    assert set(names[9:]) == set(s.panel.KNOBS)
    x = np.linspace(*s.panel.domain, 200)
    for i, p in enumerate(s.params):
        s.selected = i
        for k in ("end", "left", "home", "S-right", "right", "d"):
            app.key(k, 0)
            run(app, 2)
            s.panel.layers(x)
            assert np.isfinite(s.panel.level)
        assert s.panel.drive == s.panel.default_drive or p.name != "drive"
    lines = plain(app.draw(150, 50, "demo"))
    assert len(lines) == 50 and all(len(line) <= 150 for line in lines)
    assert any(f"tune · {key}" in line for line in lines)


def test_tune_changes_are_live_and_undo_restores(tmp_path):
    app, _ = make_app(tmp_path)
    app.key("2", 0)
    app.key("k", 0)
    s = app.tune
    sinc = app.panels[1]
    s.selected = [p.name for p in s.params].index("width_max")
    app.key("S-right", 0)
    assert sinc.k["width_max"] == pytest.approx(23)
    assert s.dirty and s.section == {"width_max": "23"}
    app.key("u", 0)
    assert sinc.k["width_max"] == 18 and not s.dirty


def test_drive_row_cycles_choices(tmp_path):
    app, _ = make_app(tmp_path)
    app.key("2", 0)
    app.key("k", 0)
    s = app.tune
    assert s.current().name == "drive"
    app.key("right", 0)
    assert app.panels[1].drive == s.drive_choices[1] == "sub"
    app.key("left", 0)
    app.key("left", 0)  # wraps to the last choice: a slider scaled to 0..1
    assert app.panels[1].drive == "chaos/10"
    run(app, 2)
    app.key("d", 0)
    assert app.panels[1].drive == "air" and "drive" not in s.section
    assert "punch/10" in drive_choices(app.cfg)


def test_expression_values_show_as_text_and_become_numbers(tmp_path):
    app, _ = make_app(tmp_path)
    app.key("k", 0)  # panel 1: the 808, whose gain is an expression
    s = app.tune
    gain = next(p for p in s.params if p.name == "gain")
    assert s.expression(gain) == "0.5 + punch/5" and s.text(gain) == "0.5 + punch/5"
    s.selected = s.params.index(gain)
    app.key("right", 0)
    assert s.section["gain"] == "1.55" and not app.panels[0].t_src


def test_ms_rows_show_the_global_value_until_set(tmp_path):
    app, _ = make_app(tmp_path)
    app.key("k", 0)
    s = app.tune
    p = next(q for q in s.params if q.name == "decay_ms")
    assert s.text(p).startswith("global ")
    s.selected = s.params.index(p)
    app.key("right", 0)
    assert "decay_ms" in s.section and not s.text(p).startswith("global")


def test_retune_keeps_the_smoothing_state(tmp_path):
    app, _ = make_app(tmp_path, SLIDERS.replace("[tune.808]\n", "[tune.808]\ndecay_ms = 400\n"), frames=30)
    p = app.panels[0]
    follow, before = p._follow, p.level
    p.retune(dict(p.section, decay_ms="500"))
    assert p._follow is follow and p._follow.s.decay_ms == 500
    assert p.level == before


def test_controls_panel_and_view_take_the_arrows(tmp_path):
    app, _ = make_app(tmp_path)
    assert app.slider_mode()  # a controls panel is in the grid
    app.key("right", 0)
    assert app.bank.values["punch"] == pytest.approx(5.1)
    app.key("down", 0)
    app.key("]", 0)
    assert app.bank.values["chaos"] == pytest.approx(4.0)
    app.key("1", 0)
    assert not app.slider_mode()  # focused on the 808
    app.key("c", 0)
    assert app.slider_mode() and app.controls
    app.key("home", 0)
    assert app.bank.values["chaos"] == 0
    app.key("esc", 0)
    assert not app.controls
    app.key("5", 0)
    app.key("k", 0)
    assert app.tune is None and "no tuning" in app.note


def test_digits_switch_the_tuned_panel_and_keep_sessions(tmp_path):
    app, _ = make_app(tmp_path)
    app.key("k", 0)
    app.tune.selected = 1
    app.key("right", 0)
    first = app.tune
    app.key("2", 0)
    assert app.tune.panel.key == "sinc"
    app.key("1", 0)
    assert app.tune is first and first.dirty
    assert app.key("esc", 0) and app.tune is None
    assert app.key("esc", 0) is False  # nothing left to close: quit


# ---------------------------------------------------------------- writing to the config


def test_update_section_keeps_comments_and_other_sections(tmp_path):
    path = write_cfg(tmp_path, "# top\n[general]\nfps = 60\n\n[tune.sinc]\n# a note\n"
                               "gain = 1.5   # louder\ncurve = 2\n\n[color]\ntheme = ocean\n")
    update_section(path, "tune.sinc", {"gain": "2", "curve": None, "width_max": "24", "drive": None})
    text = open(path).read()
    assert "gain = 2   # louder" in text and "curve" not in text and "# a note" in text
    assert text.index("width_max = 24") < text.index("[color]")
    assert "# top" in text and "theme = ocean" in text
    assert os.path.exists(path + ".bak")
    update_section(path, "tune.808@phonk", {"tau": "0.3"})
    assert open(path).read().rstrip().endswith("[tune.808@phonk]\ntau = 0.3")
    assert load(path).tune["sinc"] == {"gain": "2", "width_max": "24"}


def test_update_section_creates_a_missing_file(tmp_path):
    path = str(tmp_path / "new" / "config")
    update_section(path, "tune.saw", {"n_max": "40"})
    assert load(path).tune["saw"] == {"n_max": "40"}


def test_w_writes_only_what_differs_from_the_defaults(tmp_path):
    app, path = make_app(tmp_path)
    app.key("2", 0)
    app.key("k", 0)
    s = app.tune
    s.selected = [p.name for p in s.params].index("width_max")
    app.key("right", 0)
    s.selected = [p.name for p in s.params].index("curve")
    app.key("right", 0)
    app.key("left", 0)  # back to the default: not written
    app.key("w", 0)
    assert "wrote [tune.sinc]" in app.note and not s.dirty
    assert load(path).tune["sinc"] == {"width_max": "18.5"}
    # the config reloads to exactly what was on screen
    app.key("r", 0)
    assert app.panels[1].k["width_max"] == pytest.approx(18.5)


def test_w_with_a_profile_writes_only_the_difference(tmp_path):
    text = SLIDERS + "\n[tune.sinc]\nwidth_max = 20\n\n[tune.sinc@live]\ngain = 2\n"
    path = write_cfg(tmp_path, text)
    a = args_for(path, profile="live")
    from mathwave.cli import load_config
    app = App(load_config(a), a)
    run(app, 3)
    app.key("2", 0)
    app.key("k", 0)
    s = app.tune
    assert s.section_name == "tune.sinc@live"
    s.selected = [p.name for p in s.params].index("gain")
    app.key("d", 0)  # gain back to default inside the profile
    s.selected = [p.name for p in s.params].index("amp_min")
    app.key("right", 0)
    app.key("w", 0)
    live, base = load(path, "live").tune["sinc"], load(path).tune["sinc"]
    # gain is back at the default, which the plain section already gives: no key needed
    assert live == {"width_max": "20", "amp_min": "0.16"}
    assert load(path, "live").tune["sinc"].get("gain") is None
    assert base == {"width_max": "20"}


def test_unwritable_config_reports_instead_of_crashing(tmp_path):
    app, path = make_app(tmp_path)
    os.chmod(tmp_path, 0o500)
    try:
        app.key("k", 0)
        app.tune.selected = 1
        app.key("right", 0)
        app.key("w", 0)
        assert app.note.startswith("could not write") or os.geteuid() == 0
    finally:
        os.chmod(tmp_path, 0o700)


def test_reload_keeps_slider_positions_and_reopens_tune(tmp_path):
    app, _ = make_app(tmp_path)
    app.bank.set("punch", 8.0)
    app.key("k", 0)
    app.tune.selected = 1
    app.key("right", 0)
    app.key("r", 0)
    assert app.bank.values["punch"] == 8.0
    assert app.tune is not None and not app.tune.dirty
    assert "unsaved tuning discarded" in app.note


# ---------------------------------------------------------------- custom panel knobs


def test_custom_knobs_parse_and_tune(tmp_path):
    app, _ = make_app(tmp_path, SLIDERS + "\n[tune.bell]\nheight = 0.5\n")
    bell = app.panels[3]
    assert bell.KNOBS["height"][0] == 0.85 and bell.RANGES["spread"] == (0.05, 1.0, pytest.approx(0.0095))
    assert bell.k["height"] == 0.5
    assert bell.p["A"] == pytest.approx(0.15 + 0.5 * bell.level)


@pytest.mark.parametrize("knobs,msg", [
    ("width = 0.3", "already a drive"),
    ("s = 0.3", "already a drive, param"),
    ("A = 0.3", "lowercase"),
    ("height 0.3", "use name = default"),
    ("height = 2 (0..1)", "default inside"),
])
def test_bad_custom_knobs(tmp_path, knobs, msg):
    text = SLIDERS.replace("knobs = height = 0.85 (0..1), spread = 0.3 (0.05..1)", f"knobs = {knobs}")
    with pytest.raises(SystemExit, match=msg):
        load(write_cfg(tmp_path, text))


# ---------------------------------------------------------------- input and mouse


def test_parse_input():
    data = "a\x1b[A\x1b[B\x1b[1;2C\x1b[1;2D\x1bOC\x1b[5~\x1b[6~\x1b[H\x1b[4~\x1b[<0;10;5M\x1b[<32;12;5M\x1b[<0;12;5m\x1b[<64;3;4M\x1b"
    ev = parse_input(data)
    assert ev[:10] == ["a", "up", "down", "S-right", "S-left", "right", "pgup", "pgdn", "home", "end"]
    assert ev[10] == Mouse(0, 9, 4, True, False)
    assert ev[11] == Mouse(0, 11, 4, True, True)
    assert ev[12] == Mouse(0, 11, 4, False, False)
    assert ev[13] == Mouse(64, 2, 3, True, False)
    assert ev[14] == "esc"
    assert parse_input("q\r\t\x7f") == ["q", "enter", "tab", "backspace"]


def test_mouse_click_drag_and_wheel_on_the_controls_view(tmp_path):
    app, _ = make_app(tmp_path)
    app.key("c", 0)
    app.draw(140, 40, "demo")
    hit = next(h for h in app.rend.hits if h.name == "chaos")
    app.key(Mouse(0, hit.c1, hit.row, True), 0)
    assert app.bank.values["chaos"] == 10 and app.bank.current() == "chaos"
    app.key(Mouse(0, hit.c0, hit.row, True, motion=True), 0)
    assert app.bank.values["chaos"] == 0
    mid = (hit.c0 + hit.c1) // 2
    app.key(Mouse(0, mid, hit.row, True, motion=True), 0)
    assert app.bank.values["chaos"] == pytest.approx(5.0, abs=0.2)
    app.key(Mouse(0, mid, hit.row, False), 0)
    app.key(Mouse(0, 0, 0, True, motion=True), 0)  # no drag after release
    assert app.bank.values["chaos"] == pytest.approx(5.0, abs=0.2)
    v = app.bank.values["chaos"]
    app.key(Mouse(64, hit.span[0], hit.row, True), 0)
    assert app.bank.values["chaos"] == pytest.approx(v + 0.1)


def test_mouse_in_tune_mode(tmp_path):
    app, _ = make_app(tmp_path)
    app.key("2", 0)
    app.key("k", 0)
    app.draw(160, 50, "demo")
    hit = next(h for h in app.rend.hits if h.kind == "tune" and h.name == "width_max")
    app.key(Mouse(0, hit.c1, hit.row, True), 0)
    assert app.panels[1].k["width_max"] == 40
    assert app.tune.current().name == "width_max"
    drive = next(h for h in app.rend.hits if h.name == "drive")
    app.key(Mouse(0, drive.c1, drive.row, True), 0)
    assert app.panels[1].drive == "sub"


# ---------------------------------------------------------------- layout


@pytest.mark.parametrize("size", [(168, 50), (100, 40), (90, 44), (80, 30)])
def test_tune_and_controls_fit_any_terminal(tmp_path, size):
    cols, rows = size
    app, _ = make_app(tmp_path)
    for keys in (["k"], ["k", "down"] + ["down"] * 15, ["c"], ["m"]):
        app.key("g", 0)
        for k in keys:
            app.key(k, 0)
        lines = plain(app.draw(cols, rows, "demo"))
        assert len(lines) == rows and all(len(line) <= cols for line in lines), keys
    app.key("g", 0)
    app.key("m", 0)
    assert any("SLIDERS" in line for line in plain(app.draw(140, 60, "demo")))


def test_controls_panel_without_sliders_explains_itself(tmp_path):
    app, _ = make_app(tmp_path, "[general]\npanels = 808, controls\n")
    text = "\n".join(plain(app.draw(120, 30, "demo")))
    assert "No sliders yet" in text
    assert isinstance(app.panels[1], Controls) and app.panels[1].live().startswith("no sliders")


def test_dump_with_tune_and_keys(tmp_path):
    path = write_cfg(tmp_path, SLIDERS)
    out = subprocess.run([sys.executable, "-m", "mathwave", "-p", path, "--dump", "5", "--focus", "2", "--tune",
                          "--keys-in", r"\x1b[B\x1b[1;2C", "--size", "150x45", "--no-color"],
                         capture_output=True, text=True, env=cli_env(tmp_path), check=True).stdout
    assert "tune · sinc · unsaved" in out and "gain=1.5" in out


def test_keys_list_mentions_the_controls(tmp_path):
    out = subprocess.run([sys.executable, "-m", "mathwave", "--keys"], capture_output=True, text=True,
                         env=cli_env(tmp_path), check=True).stdout
    for word in ["tune mode", "controls", "drag", "write tuning"]:
        assert word in out


def test_shipped_config_slider_examples_load(tmp_path):
    from test_v03 import _uncommented_config
    cfg = load(_uncommented_config(tmp_path))
    assert set(cfg.sliders) == {"punch", "chaos"}
    assert cfg.sliders["punch"].label == "kick weight"


def test_tune_values_are_never_truncated(tmp_path):
    app, _ = make_app(tmp_path)
    app.key("2", 0)
    app.key("k", 0)
    text = "\n".join(plain(app.draw(168, 50, "demo")))
    s = app.tune
    for p in s.params:
        if p.kind != "drive":
            assert s.text(p) in text, p.name

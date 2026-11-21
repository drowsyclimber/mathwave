import os
import subprocess
import sys
import wave

import numpy as np
import pytest

from mathwave import synth
from mathwave.audio import RingBuffer, pcm_to_mono
from mathwave.config import load
from mathwave.dsp import Analyzer
from mathwave.panels import DEFAULT_ORDER, ExprPanel, Logistic, compile_expr, make_panels
from mathwave.render import Renderer, braille_codes, grid_layout, plot_curve

RATE = 44100


def cli_env(tmp_path):
    """Environment for running the CLI in tests: never read the user's real config."""
    return dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(__file__), "..", "src"),
                XDG_CONFIG_HOME=str(tmp_path / "xdg"), HOME=str(tmp_path / "home"))


def tone(freq, seconds=0.2, amp=0.5):
    t = np.arange(int(RATE * seconds)) / RATE
    return amp * np.sin(2 * np.pi * freq * t)


def settle(an, x, frames=30):
    f = None
    for k in range(frames):
        f = an.analyze(x, k / 60)
    return f


@pytest.mark.parametrize("freq,band", [(40, "sub"), (120, "bass"), (1000, "mid"), (5000, "presence"), (9000, "air")])
def test_tone_lands_in_its_band(freq, band):
    an = Analyzer(RATE, fft_size=4096)
    f = settle(an, tone(freq))
    assert max(f.levels, key=lambda k: f.levels[k] if k not in ("rms", "centroid") else -1) == band


def test_silence_reads_as_zero():
    an = Analyzer(RATE)
    f = settle(an, np.zeros(2048))
    assert all(v < 0.05 for v in f.levels.values())
    assert f.spectrum.max() < 0.05


def test_onsets_and_tempo_from_demo():
    an = Analyzer(RATE)
    onsets = 0
    for k in range(60 * 6):
        t = k / 60
        x = synth.render(np.maximum(t - np.arange(2048)[::-1] / RATE, 0))
        f = an.analyze(x, t)
        onsets += f.onset
    beats = 6 / synth.BEAT
    assert 0.7 * beats <= onsets <= 1.2 * beats
    assert abs(f.bpm - synth.BPM) < 10


def test_ring_buffer_wraps():
    rb = RingBuffer(8)
    rb.write(np.arange(5))
    rb.write(np.arange(5, 11))
    assert rb.latest(4).tolist() == [7, 8, 9, 10]


def test_pcm_to_mono_averages_channels():
    pcm = np.array([16384, -16384, 32767, 32767], dtype="<i2").tobytes()
    mono = pcm_to_mono(pcm, 2)
    assert mono.shape == (2,)
    assert abs(mono[0]) < 1e-6 and mono[1] > 0.99


def test_braille_encoding():
    g = np.zeros((4, 2), dtype=bool)
    g[0, 0] = True  # dot 1
    g[3, 1] = True  # dot 8
    assert braille_codes(g)[0, 0] == 0x01 | 0x80


def test_plot_curve_breaks_on_nan():
    g = np.zeros((8, 10), dtype=bool)
    y = np.linspace(-1, 1, 10)
    y[5] = np.nan
    plot_curve(g, y, (-1, 1))
    assert not g[:, 5].any()
    assert g[:, 4].any() and g[:, 6].any()


def test_layout_fits_and_covers():
    rects = grid_layout(9, 180, 54)
    assert len(rects) == 9
    assert max(r.c + r.w for r in rects) == 180
    assert max(r.r + r.h for r in rects) == 54
    assert grid_layout(9, 40, 12) is None


def test_every_panel_renders_and_prints_its_formula():
    panels = make_panels(DEFAULT_ORDER)
    an = Analyzer(RATE)
    t = 37.0
    f = an.analyze(synth.render(t - np.arange(2048)[::-1] / RATE), t)
    for p in panels:
        p.update(f, 1 / 60)
    lines = Renderer(color_mode="none").render(panels, f, 170, 50)
    text = "\n".join(lines)
    assert len(lines) == 50 and all(len(l) == 170 for l in lines)
    for p in panels:
        assert p.formula in text
        assert any(ord(c) > 0x2800 and ord(c) <= 0x28FF for c in text)


def test_logistic_regimes():
    assert Logistic.regime(2.9) == "fixed point"
    assert Logistic.regime(3.2) == "period 2"
    assert Logistic.regime(3.99) == "chaos"


def test_expr_panel_and_sandbox():
    p = ExprPanel("bell", "A*exp(-(x/s)^2)", {"A": "0.5 + 0.5*bass", "s": "0.3"})
    an = Analyzer(RATE)
    p.update(an.analyze(np.zeros(2048), 0.0), 1 / 60)
    kind, y, _ = p.layers(np.linspace(-1, 1, 5))[1]
    assert kind == "curve" and np.isclose(y[2], 0.5)
    for bad in ["__import__('os')", "x.real", "open('f')", "[x]", "lambda: 1"]:
        with pytest.raises((ValueError, SyntaxError)):
            compile_expr(bad, {"x"})


def test_config_custom_panel(tmp_path):
    cfg_file = tmp_path / "config"
    cfg_file.write_text(
        "[general]\npanels = 808, bell\nformulas = 1\n"
        "[panel.bell]\nexpr = A*exp(-(x/s)^2)\nparams = A = bass, s = 0.2\n"
    )
    cfg = load(str(cfg_file))
    assert cfg.panels == ["808", "bell"] and cfg.formulas == 1
    panels = make_panels(cfg.panels, cfg.custom)
    assert panels[1].formula == "y = A·exp(-(x/s)^2)"


def test_wav_render(tmp_path):
    out = tmp_path / "demo.wav"
    synth.write_wav(str(out), seconds=1.0)
    with wave.open(str(out)) as w:
        assert w.getnframes() == RATE and w.getframerate() == RATE


def test_cli_dump_and_formulas(tmp_path):
    env = cli_env(tmp_path)
    out = subprocess.run([sys.executable, "-m", "mathwave", "--dump", "30", "--size", "120x40", "--no-color"],
                         capture_output=True, text=True, env=env, check=True).stdout
    assert "y = clip(A·tan(ωx + φ), −1, 1)" in out
    out = subprocess.run([sys.executable, "-m", "mathwave", "--formulas"], capture_output=True, text=True,
                         env=env, check=True).stdout
    assert out.count("driven by:") == len(DEFAULT_ORDER)


def test_shipped_config_bell_panel():
    cfg = load(os.path.join(os.path.dirname(__file__), "..", "config", "config"))
    assert cfg.panels[-1] == "bell"
    bell = cfg.custom["bell"]
    assert bell.drive == "bass" and bell.formula == "y = A·exp(-x^2/(2·s^2))"
    an = Analyzer(RATE)
    quiet = settle(an, np.zeros(2048))
    bell.update(quiet, 1 / 60)
    a_quiet, s_quiet = bell.p["A"], bell.p["s"]
    loud = settle(an, tone(120, amp=0.8))
    bell.update(loud, 1 / 60)
    assert bell.p["A"] > a_quiet + 0.5 and bell.p["s"] > s_quiet
    y = bell.layers(np.linspace(-1, 1, 201))[1][1]
    assert np.isclose(y[100], bell.p["A"]) and y[0] < 0.1 * y[100]


def test_panel_text_fits_the_narrowest_panel():
    """Panels are at least 40 columns wide, so 36 columns of text and a 24-char title."""
    cfg = load(os.path.join(os.path.dirname(__file__), "..", "config", "config"))
    panels = make_panels(cfg.panels, cfg.custom)
    an = Analyzer(RATE)
    t = 37.0
    f = an.analyze(synth.render(t - np.arange(2048)[::-1] / RATE), t)
    for i, p in enumerate(panels):
        p.update(f, 1 / 60)
        assert len(f" {i + 1} {p.title} ") <= 25, p.title
        for line in (p.formula, p.identity, p.live()):
            assert len(line) <= 36, line


def test_params_with_multi_argument_functions(tmp_path):
    cfg_file = tmp_path / "config"
    cfg_file.write_text(
        "[general]\npanels = bell\n"
        "[panel.bell]\nexpr = A*exp(-x^2/(2*s^2))\n"
        "params = A = 0.15 + 0.85*max(sub, bass), s = clip(0.05 + 0.4*bass, 0.05, 0.3)\n"
    )
    bell = load(str(cfg_file)).custom["bell"]
    assert set(bell.param_code) == {"A", "s"}
    an = Analyzer(RATE)
    bell.update(settle(an, tone(50, amp=0.8)), 1 / 60)
    assert bell.p["A"] > 0.9 and bell.p["s"] == pytest.approx(0.3)


def test_fft_size_config_and_sub_separation(tmp_path):
    cfg_file = tmp_path / "config"
    cfg_file.write_text("[smoothing]\nfft_size = 4096\n")
    assert load(str(cfg_file)).fft_size == 4096
    cfg_file.write_text("[smoothing]\nfft_size = 3000\n")
    with pytest.raises(SystemExit):
        load(str(cfg_file))

    # Kicks on even beats, an 80 Hz bass line on odd beats: 4096 keeps the line out of "sub".
    def sub_on_line(n):
        an = Analyzer(RATE, fft_size=n)
        beat = 60 / 140
        vals = []
        for k in range(60 * 8):
            tt = k / 60
            times = np.maximum(tt - np.arange(n)[::-1] / RATE, 0)
            pos = times / beat
            tb = (pos % 1) * beat
            kick = np.exp(-tb / 0.1) * np.sin(2 * np.pi * (46 * tb + 114 * 0.035 * (1 - np.exp(-tb / 0.035))))
            x = np.where(np.floor(pos) % 2 == 0, 0.8 * kick, 0.35 * np.sin(2 * np.pi * 80 * times))
            f = an.analyze(np.tanh(x), tt)
            p = (tt / beat) % 2
            if tt > 2 and 1.4 < p < 1.9:
                vals.append(f["sub"])
        return float(np.mean(vals))

    assert sub_on_line(4096) < 0.7 * sub_on_line(2048)


# ---------------------------------------------------------------- per-panel tuning

from mathwave.dsp import Features
from mathwave.panels import BUILTIN, Exp808, Spectrum, Tuning, make_panels as _mk


def feats(**levels):
    base = {k: 0.0 for k in ("sub", "bass", "lowmid", "mid", "highmid", "presence", "air", "rms", "centroid")}
    base.update(levels)
    return Features(levels=dict(base), raw=dict(base))


def test_tuning_shape_math():
    t = Tuning(gain=2.0, threshold=0.5, curve=2.0, lo=0.2, hi=0.8)
    # 0.3*2=0.6 -> (0.6-0.5)/0.5=0.2 -> 0.04 -> 0.2+0.6*0.04
    assert t.shape(0.3) == pytest.approx(0.224)
    assert t.shape(0.1) == pytest.approx(0.2)  # under threshold rests at min
    assert t.shape(1.0) == pytest.approx(0.8)
    inv = Tuning(lo=1.0, hi=0.0)
    assert inv.shape(0.25) == pytest.approx(0.75)


def test_drive_override_changes_what_the_panel_hears():
    default, sub_only = _mk(["808"]), _mk(["808"], tune={"808": {"drive": "sub"}})
    f = feats(sub=0.1, bass=1.0)
    default[0].update(f, 1 / 60)
    sub_only[0].update(f, 1 / 60)
    assert default[0].level == pytest.approx(0.7)  # max(sub, 0.7*bass)
    assert sub_only[0].level == pytest.approx(0.1)


def test_knobs_override_formula_constants():
    p = _mk(["808"], tune={"808": {"tau": "0.5", "f1": "1"}})[0]
    p.update(feats(sub=1.0), 1 / 60)
    assert p.p["τ"] == 0.5 and p.p["f₁"] == 1.0
    assert "tau=0.5" in p.tuning_summary() and "f1=1" in p.tuning_summary()
    saw = _mk(["saw"], tune={"saw": {"n_max": "7.6"}})[0]
    saw.update(feats(centroid=1.0), 1 / 60)
    assert saw.p["N"] == 8


def test_own_decay_falls_faster_than_global_smoothing():
    fast = _mk(["sinc"], tune={"sinc": {"decay": "0.5"}})[0]
    slow = _mk(["sinc"])[0]
    # global smoothing would hold f.levels high; the panel's own decay reads raw levels.
    held = Features(levels={**feats().levels, "air": 0.9}, raw={**feats().raw, "air": 1.0})
    fast.update(held, 1 / 60)
    for _ in range(3):
        f = Features(levels={**feats().levels, "air": 0.8}, raw=feats().raw)
        fast.update(f, 1 / 60)
        slow.update(f, 1 / 60)
    assert fast.level == pytest.approx(0.125) and slow.level == pytest.approx(0.8)


def test_bad_tune_sections_are_reported():
    with pytest.raises(SystemExit, match="unknown key 'tua'"):
        _mk(["808"], tune={"808": {"tua": "1"}})
    with pytest.raises(SystemExit, match="does not match any panel"):
        _mk(["808"], tune={"80": {"gain": "2"}})
    with pytest.raises(SystemExit, match="unknown name 'subb'"):
        _mk(["808"], tune={"808": {"drive": "subb"}})
    with pytest.raises(SystemExit, match="must be"):
        _mk(["808"], tune={"808": {"decay": "1.5"}})


def test_every_builtin_has_documented_knobs_and_accepts_common_keys():
    for cls in BUILTIN:
        p = _mk([cls.key], tune={cls.key: {"gain": "1.5", "curve": "2", "min": "0.1", "max": "0.9",
                                         "threshold": "0.1", "attack": "0.5", "decay": "0.8"}})[0]
        for _ in range(3):
            p.update(feats(sub=0.7, bass=0.7, lowmid=0.7, mid=0.7, highmid=0.7, presence=0.7, air=0.7,
                           rms=0.7, centroid=0.7), 1 / 60)
        assert 0.1 <= p.level <= 0.9
        p.layers(np.linspace(*p.domain, 80))
        for name, (default, desc) in cls.KNOBS.items():
            assert name == name.lower() and desc


def test_spectrum_tuning_shapes_bars():
    p = _mk(["spectrum"], tune={"spectrum": {"threshold": "0.5"}})[0]
    f = Features(spectrum=np.array([0.2, 0.75, 1.0], dtype=np.float32))
    p.update(f, 1 / 60)
    assert p.bars.tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_custom_panel_level_uses_its_tuning(tmp_path):
    cfg_file = tmp_path / "config"
    cfg_file.write_text(
        "[general]\npanels = bell\n"
        "[panel.bell]\nexpr = A*exp(-x^2/(2*s^2))\nparams = A = level, s = 0.2\ndrive = bass\n"
        "[tune.bell]\ndrive = sub\ncurve = 2\n"
    )
    cfg = load(str(cfg_file))
    bell = _mk(cfg.panels, cfg.custom, cfg.tune)[0]
    bell.update(feats(sub=0.5, bass=1.0), 1 / 60)
    assert bell.p["A"] == pytest.approx(0.25)


def test_shipped_config_tune_examples_parse():
    """Every commented [tune.*] example in the shipped config must be valid when uncommented."""
    path = os.path.join(os.path.dirname(__file__), "..", "config", "config")
    lines, block = open(path).read().splitlines(), []
    text = []
    for line in lines:
        if line.startswith("# [tune.") or (block and line.startswith("# ") and "=" in line and not line.startswith("# #")):
            block.append(line[2:])
        elif block:
            text += block + [""]
            block = []
    assert any(l.startswith("[tune.") for l in text)
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".ini", delete=False) as fh:
        fh.write("\n".join(text))
    cfg = load(fh.name)
    _mk(DEFAULT_ORDER, cfg.custom, cfg.tune)
    assert set(cfg.tune) >= set(DEFAULT_ORDER)


def test_cli_list_knobs(tmp_path):
    env = cli_env(tmp_path)
    out = subprocess.run([sys.executable, "-m", "mathwave", "--list-knobs"], capture_output=True, text=True,
                         env=env, check=True).stdout
    for cls in BUILTIN:
        assert f"[tune.{cls.key}]" in out
    out = subprocess.run([sys.executable, "-m", "mathwave", "--list-knobs", "808"], capture_output=True, text=True,
                         env=env, check=True).stdout
    assert "tau" in out and "[tune.dive]" not in out


def test_cli_tests_ignore_the_users_config(tmp_path):
    """A personal config with extra panels or tuning must not change test results."""
    env = cli_env(tmp_path)
    cfg_dir = tmp_path / "xdg" / "mathwave"
    cfg_dir.mkdir(parents=True)
    out = subprocess.run([sys.executable, "-m", "mathwave", "--formulas"], capture_output=True, text=True,
                         env=env, check=True).stdout
    assert out.count("driven by:") == len(DEFAULT_ORDER)
    (cfg_dir / "config").write_text("[general]\npanels = 808, bell\n"
                                    "[panel.bell]\nexpr = A*exp(-x^2)\nparams = A = level\n")
    out = subprocess.run([sys.executable, "-m", "mathwave", "--formulas"], capture_output=True, text=True,
                         env=env, check=True).stdout
    assert out.count("driven by:") == 2  # proves the env really controls which config is read

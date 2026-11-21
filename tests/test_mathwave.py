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


def test_cli_dump_and_formulas():
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(__file__), "..", "src"))
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

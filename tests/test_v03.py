"""Tests for the 0.3 tuning features: bands, weighting, multi-resolution FFT,
onsets, tempo, pitch, spectral features, chroma, stereo, ms smoothing, hold,
meters and profiles."""

import os
import subprocess
import sys

import numpy as np
import pytest

from mathwave import synth
from mathwave.audio import OfflineDemoSource, RingBuffer, pcm_to_stereo
from mathwave.cli import build
from mathwave.config import load, parse_band, parse_ranges
from mathwave.dsp import BANDS, AnalysisSettings, Analyzer, BandSpec, Follower, Smoothing
from mathwave.panels import compile_expr, inline_bands, make_panels
from mathwave.render import Renderer, chroma_palette

RATE = 44100


def cli_env(tmp_path):
    return dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(__file__), "..", "src"),
                XDG_CONFIG_HOME=str(tmp_path / "xdg"), HOME=str(tmp_path / "home"))


def write_cfg(tmp_path, text, name="config"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def harmonic(freq, seconds=0.5, amp=0.4, n=5, phase=0.0):
    t = np.arange(int(RATE * seconds)) / RATE
    y = sum(np.sin(2 * np.pi * freq * k * t + phase) / k for k in range(1, n + 1))
    return amp * y / np.max(np.abs(y))


def tone(freq, seconds=0.5, amp=0.4):
    t = np.arange(int(RATE * seconds)) / RATE
    return amp * np.sin(2 * np.pi * freq * t)


def settle(an, x, frames=40, dt=1 / 60):
    f = None
    for k in range(frames):
        f = an.analyze(x, (k + 1) * dt)
    return f


# ---------------------------------------------------------------- 1, 3, 5: custom bands


def test_custom_band_from_config(tmp_path):
    path = write_cfg(tmp_path, "[bands]\nkickband = 40-90 peak range=25\nbass = 60-200\n")
    cfg = load(path)
    assert cfg.bands["kickband"] == BandSpec(40, 90, mode="peak", range_db=25.0)
    assert cfg.bands["bass"].hi == 200
    assert "kickband" in cfg.names
    an = Analyzer(settings=cfg.analysis())
    assert an.bands["bass"].hi == 200
    f = settle(an, tone(60, 0.5))
    assert f.levels["kickband"] > 0.8
    assert f.band_ranges["kickband"][:2] == (40, 90)


def test_band_override_without_range_keeps_builtin_limits():
    spec = parse_band("sub", "peak floor=-60")
    assert (spec.lo, spec.hi) == BANDS["sub"]
    assert spec.mode == "peak" and spec.floor_db == -60


@pytest.mark.parametrize("text,msg", [
    ("90-40", "LO < HI"),
    ("40-90 loud", "unknown option"),
    ("40-90 fft=3000", "bad value"),
    ("40-90 norm=both", "bad value"),
])
def test_bad_band_specs(text, msg):
    with pytest.raises(ValueError, match=msg):
        parse_band("thing", text)


@pytest.mark.parametrize("name", ["rms", "band_1_2", "t", "Bad"])
def test_reserved_or_bad_band_names(name):
    with pytest.raises(ValueError):
        parse_band(name, "40-90")


def test_peak_mode_beats_mean_for_a_narrow_tone():
    """A single sine inside a wide band: peak sees its full power, mean spreads it out."""
    s = AnalysisSettings(bands={"wm": BandSpec(100, 2000), "wp": BandSpec(100, 2000, mode="peak")})
    f = settle(Analyzer(settings=s), tone(500))
    assert f.band_db["wp"] > f.band_db["wm"] + 10


def test_per_band_range_changes_sensitivity():
    quiet = tone(120, amp=0.4)
    loud = tone(120, amp=0.4 * 10 ** (-12 / 20))  # 12 dB down
    def level(range_db):
        s = AnalysisSettings(bands={"bass": BandSpec(60, 250, range_db=range_db)}, fall_db_per_s=0)
        an = Analyzer(settings=s)
        settle(an, quiet)
        return settle(an, loud, frames=60).levels["bass"]
    # a narrow 15 dB window drops much further for the same 12 dB change than a 60 dB one
    assert level(15) < level(60) - 0.2


def test_shared_normalization_keeps_relative_levels():
    s = AnalysisSettings(bands={"lo": BandSpec(60, 250, norm="shared"), "hi": BandSpec(2000, 4000, norm="shared")})
    an = Analyzer(settings=s)
    x = tone(120, amp=0.4) + tone(3000, amp=0.4 * 10 ** (-20 / 20))
    f = settle(an, x, frames=90)
    # shared: the quieter band stays clearly below the louder one
    assert f.levels["hi"] < f.levels["lo"] - 0.2
    own = Analyzer(settings=AnalysisSettings(bands={"lo": BandSpec(60, 250), "hi": BandSpec(2000, 4000)}))
    g = settle(own, x, frames=90)
    # own: each band is scaled against its own history, so the quiet band reads much higher
    assert g.levels["hi"] > f.levels["hi"] + 0.2 and g.levels["lo"] > 0.8


# ---------------------------------------------------------------- 2: inline band()


def test_inline_band_compiles_and_registers():
    inline = {}
    code = compile_expr("band(40, 90) + 0.5*peak(2000, 4000)", {"bass"}, inline)
    assert inline == {"band_40_90": (40.0, 90.0, "mean"), "peak_2000_4000": (2000.0, 4000.0, "peak")}
    assert code is not None


@pytest.mark.parametrize("src", ["band(90, 40)", "band(lo, 90)", "band(40)"])
def test_inline_band_errors(src):
    with pytest.raises(ValueError):
        compile_expr(src, {"bass"}, {})


def test_inline_band_drives_a_panel(tmp_path):
    path = write_cfg(tmp_path, "[tune.808]\ndrive = band(40, 90)\n")
    cfg = load(path)
    an, panels = build(cfg)
    assert "band_40_90" in an.bands
    assert inline_bands(panels)["band_40_90"][:2] == (40.0, 90.0)
    f = settle(an, tone(60))
    p = panels[0]
    for _ in range(30):
        p.update(f, 1 / 60)
    assert p.level > 0.7


def test_inline_band_in_custom_panel_params(tmp_path):
    path = write_cfg(tmp_path, "[panel.blip]\nexpr = A*sin(x)\nparams = A = peak(40, 90)\n"
                               "[general]\npanels = blip\n")
    an, panels = build(load(path))
    assert "peak_40_90" in an.bands


# ---------------------------------------------------------------- 4: tilt and A-weighting


def test_tilt_and_a_weighting_lift_the_highs_relative_to_bass():
    x = tone(80, amp=0.4) + tone(8000, amp=0.4)
    shares = {}
    for name, kw in {"flat": {}, "tilt": {"tilt_db_per_oct": 4.5}, "a": {"weighting": "a"}}.items():
        an = Analyzer(settings=AnalysisSettings(fft_size=4096, **kw))
        db = settle(an, x).band_db
        shares[name] = db["air"] - db["bass"]
    assert shares["tilt"] > shares["flat"] + 20
    assert shares["a"] > shares["flat"] + 20


def test_weighting_validation(tmp_path):
    with pytest.raises(SystemExit, match="weighting"):
        load(write_cfg(tmp_path, "[analysis]\nweighting = c\n"))


# ---------------------------------------------------------------- 6: multi-resolution FFT


def test_multires_picks_fft_sizes_per_band():
    an = Analyzer(settings=AnalysisSettings(multires=True, low_fft=8192, high_fft=1024, split_hz=250))
    assert an.band_fft(an.bands["sub"]) == 8192
    assert an.band_fft(an.bands["air"]) == 1024
    assert an.band_fft(an.bands["lowmid"]) == 1024  # 250-500 sits above the split
    an.add_band("mix", 200, 300)
    assert an.band_fft(an.bands["mix"]) == 2048  # straddles the split: base size
    assert an.need >= 8192


def test_multires_separates_close_sub_tones():
    """30 Hz and 50 Hz custom bands: a 50 Hz tone should leak far less into 25-35 Hz at 8192."""
    bands = {"a": BandSpec(25, 35, norm="shared"), "b": BandSpec(45, 55, norm="shared")}
    leak = {}
    for multi in (False, True):
        an = Analyzer(settings=AnalysisSettings(fft_size=1024, multires=multi, bands=dict(bands)))
        db = settle(an, tone(50, seconds=0.5)).band_db
        leak[multi] = db["b"] - db["a"]
    assert leak[True] > leak[False] + 10


def test_per_band_fft_option():
    an = Analyzer(settings=AnalysisSettings(bands={"deep": BandSpec(20, 40, fft=16384)}))
    assert an.band_fft(an.bands["deep"]) == 16384
    assert an.need == 16384


# ---------------------------------------------------------------- 7: kick, snare, hat triggers


def _drum_track(seconds=4.0):
    t = np.arange(int(RATE * seconds)) / RATE
    beat = 0.5
    tb = t % beat
    kick = np.exp(-tb / 0.12) * np.sin(2 * np.pi * (50 * tb + 110 * 0.03 * (1 - np.exp(-tb / 0.03))))
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(len(t))
    tsn = (t + beat) % (2 * beat)  # snare on beats 2 and 4
    snare = np.where(tsn < 0.15, np.exp(-tsn / 0.05), 0.0) * (np.sin(2 * np.pi * 200 * t) * 0.5 + 0.5 * noise)
    th = (t + beat / 2) % beat  # hats on the offbeats
    hp = noise - np.convolve(noise, np.ones(4) / 4, mode="same")
    hat = np.where(th < 0.04, np.exp(-th / 0.01), 0.0) * hp
    return t, 0.6 * kick, 0.4 * snare, 0.3 * hat


@pytest.mark.parametrize("which", ["kick", "snare", "hat"])
def test_each_trigger_fires_on_its_own_drum(which):
    t, kick, snare, hat = _drum_track()
    sig = {"kick": kick, "snare": snare, "hat": hat}[which]
    an = Analyzer(RATE)
    counts = {"kick": 0, "snare": 0, "hat": 0}
    for k in range(int(4.0 * 60)):
        now = k / 60
        end = int(now * RATE)
        x = sig[max(0, end - an.need):end]
        if len(x) < 64:
            continue
        an.analyze(x, now)
    counts = {name: det.count for name, det in an.onset_det.items()}
    assert counts[which] >= (3 if which == "snare" else 6)  # snares land twice per 2 s bar
    for other in counts:
        if other != which:
            assert counts[other] <= counts[which] // 2


def test_trigger_envelope_decays():
    an = Analyzer(settings=AnalysisSettings(envelope_ms=200))
    an.last_hit["kick"] = 0.0
    x = np.zeros(2048)
    f = an.analyze(x, 0.2)
    assert f.levels["kick"] == pytest.approx(0.1, abs=0.02)


# ---------------------------------------------------------------- 8: tempo


def test_tempo_fold_keeps_bpm_in_range():
    an = Analyzer(settings=AnalysisSettings(bpm_range=(70, 140)))
    assert 60 / an._fold(60 / 168) == pytest.approx(84)
    assert 60 / an._fold(60 / 50) == pytest.approx(100)
    an = Analyzer(settings=AnalysisSettings(bpm_range=(70, 180)))
    assert 60 / an._fold(60 / 168) == pytest.approx(168)


def test_fixed_bpm_and_multiplier():
    an = Analyzer(settings=AnalysisSettings(bpm=120, bpm_multiplier=2))
    f = an.analyze(np.zeros(2048), 0.0)
    assert f.tempo_source == "fixed"
    assert f.bpm == pytest.approx(240)
    an.last_hit["kick"] = 0.0
    assert an.analyze(np.zeros(2048), 0.125).levels["phase"] == pytest.approx(0.5, abs=0.01)


def test_tap_tempo_wins_and_resets():
    an = Analyzer(settings=AnalysisSettings(bpm=120))
    for k in range(4):
        an.tap(k * 0.4)  # 150 BPM
    f = an.analyze(np.zeros(2048), 2.0)
    assert f.tempo_source == "tap" and f.bpm == pytest.approx(150)
    an.reset_tempo()
    f = an.analyze(np.zeros(2048), 2.1)
    assert f.tempo_source == "fixed" and f.bpm == pytest.approx(120)


def test_tap_series_restarts_after_a_pause():
    an = Analyzer()
    an.tap(0.0)
    an.tap(0.5)
    an.tap(10.0)  # long gap starts a new series
    assert an.taps == [10.0]


def test_demo_tempo_converges_to_140():
    an = Analyzer(RATE)
    src = OfflineDemoSource(RATE)
    f = None
    for k in range(60 * 12):
        t = (k + 1) / 60
        f = an.analyze(src.latest_stereo(an.need, t), t)
    assert f.bpm == pytest.approx(synth.BPM, rel=0.03)


def test_tempo_config(tmp_path):
    cfg = load(write_cfg(tmp_path, "[tempo]\nbpm = 96\nmultiplier = 0.5\nrange = 60-120\n"))
    assert (cfg.bpm, cfg.bpm_multiplier, cfg.bpm_range) == (96, 0.5, (60, 120))
    with pytest.raises(SystemExit, match="bpm"):
        load(write_cfg(tmp_path, "[tempo]\nbpm = 900\n"))


# ---------------------------------------------------------------- 9: pitch


@pytest.mark.parametrize("freq", [41.2, 55.0, 82.4, 110.0, 146.8])
def test_pitch_tracks_bass_notes(freq):
    an = Analyzer(RATE)
    f = settle(an, harmonic(freq, seconds=0.5), frames=20)
    assert f.pitch_hz == pytest.approx(freq, rel=0.02)


def test_pitch_ignores_a_weak_fundamental():
    """Strong 2nd harmonic, weak fundamental: still report the fundamental, not the octave."""
    t = np.arange(int(RATE * 0.5)) / RATE
    x = 0.15 * np.sin(2 * np.pi * 55 * t) + 0.4 * np.sin(2 * np.pi * 110 * t) + 0.2 * np.sin(2 * np.pi * 165 * t)
    f = settle(Analyzer(RATE), x, frames=20)
    assert f.pitch_hz == pytest.approx(55, rel=0.02)


def test_pitch_names():
    f = settle(Analyzer(RATE), harmonic(55.0), frames=20)
    assert f.pitch_name.startswith("A1")


def test_pitch_range_config(tmp_path):
    cfg = load(write_cfg(tmp_path, "[analysis]\npitch_range = 40-160\n"))
    assert cfg.pitch_range == (40, 160)


# ---------------------------------------------------------------- 10: flatness, rolloff, flux


def test_flatness_noise_vs_tone():
    noise = np.random.default_rng(2).standard_normal(RATE // 2) * 0.2
    fn = settle(Analyzer(RATE), noise)
    ft = settle(Analyzer(RATE), tone(440))
    assert fn.levels["flatness"] > 0.8
    assert ft.levels["flatness"] < 0.3


def test_rolloff_follows_brightness():
    dark = settle(Analyzer(RATE), harmonic(110, n=3), frames=60)
    bright = settle(Analyzer(RATE), harmonic(110, n=3) + 0.3 * tone(9000), frames=60)
    assert bright.rolloff_hz > 4 * dark.rolloff_hz
    assert bright.levels["rolloff"] > dark.levels["rolloff"] + 0.3


def test_flux_reacts_to_change_not_level():
    an = Analyzer(RATE)
    steady = settle(an, tone(440), frames=60)
    assert steady.levels["flux"] < 0.2
    rng = np.random.default_rng(3)
    vals = []
    for k in range(60):
        x = tone(rng.uniform(200, 4000)) * rng.uniform(0.2, 0.5)
        vals.append(an.analyze(x, 2 + k / 60).levels["flux"])
    assert max(vals) > 0.6


# ---------------------------------------------------------------- 11: chroma


@pytest.mark.parametrize("freq,pc", [(440.0, 9), (261.63, 0), (329.63, 4), (196.0, 7)])
def test_chroma_finds_the_note(freq, pc):
    f = settle(Analyzer(RATE), harmonic(freq, n=3), frames=30)
    assert f.chroma_pc == pc
    assert f.levels["note"] == pytest.approx(pc / 11)
    assert f.levels["tonality"] > 0.3


def test_chroma_theme_follows_the_note():
    a = chroma_palette(9, 0.8, 4)
    c = chroma_palette(0, 0.8, 4)
    assert a != c and len(a) == 4
    r = Renderer("chroma")
    assert r.theme == "chroma"


# ---------------------------------------------------------------- 12: stereo


def test_balance_and_width():
    x = tone(440)
    an = Analyzer(RATE)
    f = settle(an, np.stack([x, x], axis=1))
    assert f.levels["balance"] == pytest.approx(0.5, abs=0.02)
    assert f.levels["width"] < 0.02
    an = Analyzer(RATE)
    f = settle(an, np.stack([x, 0.2 * x], axis=1))
    assert f.levels["balance"] < 0.4
    an = Analyzer(RATE)
    f = settle(an, np.stack([0.2 * x, x], axis=1))
    assert f.levels["balance"] > 0.6
    an = Analyzer(RATE)
    f = settle(an, np.stack([x, -x], axis=1))  # fully out of phase: all side
    assert f.levels["width"] > 0.9


def test_ring_buffer_keeps_stereo():
    rb = RingBuffer(4096)
    pcm = (np.array([[1000, -1000]] * 100, dtype="<i2")).tobytes()
    rb.write(pcm_to_stereo(pcm, 2))
    st = rb.latest_stereo(100)
    assert st.shape == (100, 2)
    assert st[0, 0] > 0 > st[0, 1]
    assert np.allclose(rb.latest(100), 0.0)


def test_demo_is_stereo_and_mid_matches_mono():
    t = np.arange(4096) / RATE + 40.0  # all layers on
    st = synth.render(t, stereo=True)
    mono = synth.render(t)
    assert st.shape == (4096, 2)
    assert np.allclose(st.mean(axis=1), mono, atol=1e-3)
    assert np.abs(st[:, 0] - st[:, 1]).max() > 0.01


# ---------------------------------------------------------------- 13, 14: ms smoothing and hold


def _response(smoothing, dt, seconds=0.3):
    fol = Follower(smoothing)
    fol(0.0, dt)
    v = 0.0
    for _ in range(int(round(seconds / dt))):
        v = fol(1.0, dt)
    return float(v)


def test_ms_smoothing_is_frame_rate_independent():
    s = Smoothing(attack_ms=100, decay_ms=400)
    at30 = _response(s, 1 / 30, 0.1)
    at144 = _response(s, 1 / 144, 0.1)
    assert at30 == pytest.approx(0.9, abs=0.02)
    assert at144 == pytest.approx(0.9, abs=0.02)


def test_hold_freezes_the_peak():
    fol = Follower(Smoothing(attack_ms=0, decay_ms=50, hold_ms=200))
    fol(1.0, 1 / 60)
    for _ in range(9):  # 150 ms later, still held
        v = fol(0.0, 1 / 60)
    assert v == pytest.approx(1.0)
    for _ in range(12):  # past the hold, it falls
        v = fol(0.0, 1 / 60)
    assert v < 0.2


def test_ms_smoothing_in_config_and_per_panel(tmp_path):
    path = write_cfg(tmp_path, "[smoothing]\nattack_ms = 20\ndecay_ms = 300\nhold_ms = 50\n"
                               "[tune.808]\ndecay_ms = 900\n")
    cfg = load(path)
    assert (cfg.attack_ms, cfg.decay_ms, cfg.hold_ms) == (20, 300, 50)
    panels = make_panels(["808"], {}, cfg.tune, names=cfg.names, smoothing=cfg.smoothing)
    sm = panels[0].tune.smoothing(cfg.smoothing)
    assert sm.decay_ms == 900 and sm.attack_ms == 20


# ---------------------------------------------------------------- knob expressions


def test_knob_expression_follows_a_feature(tmp_path):
    path = write_cfg(tmp_path, "[tune.reese]\na = 8 + 16*pitch\n")
    cfg = load(path)
    an, panels = build(cfg)
    p = panels[[q.key for q in panels].index("reese")]
    lo = settle(an, harmonic(41.2), frames=20)
    p.update(lo, 1 / 60)
    a_lo = p.k["a"]
    an2, _ = build(cfg)
    hi = settle(an2, harmonic(146.8), frames=20)
    p.update(hi, 1 / 60)
    assert p.k["a"] > a_lo + 4


def test_bad_knob_expression_is_a_config_error(tmp_path):
    path = write_cfg(tmp_path, "[tune.reese]\na = 8 + wobble\n")
    with pytest.raises(SystemExit, match="wobble"):
        build(load(path))


# ---------------------------------------------------------------- 15: meters


def test_meter_view_lists_every_feature():
    an = Analyzer(RATE)
    f = settle(an, harmonic(110))
    lines = Renderer("phonk", color_mode="none").render([], f, 120, 50, meters=True)
    text = "\n".join(lines)
    import re
    for name in ["sub", "air", "rms", "rolloff", "flatness", "flux", "pitch", "kick", "hat", "balance", "width"]:
        assert re.search(rf"^ *{name} ", text, re.M), name
    assert "A2" in text  # pitch name of 110 Hz
    assert len(lines) == 50
    assert all(len(line) <= 120 for line in lines)


def test_meter_view_fits_small_terminals():
    f = settle(Analyzer(RATE), harmonic(110))
    lines = Renderer("phonk", color_mode="none").render([], f, 60, 12, meters=True)
    assert len(lines) == 12 and all(len(line) <= 60 for line in lines)


def test_dump_meters_cli(tmp_path):
    out = subprocess.run([sys.executable, "-m", "mathwave", "--dump", "30", "--meters", "--size", "100x40", "--no-color"],
                         capture_output=True, text=True, env=cli_env(tmp_path), check=True).stdout
    assert "BANDS" in out and "STEREO" in out


# ---------------------------------------------------------------- 16: profiles


PROFILE_CFG = """
[general]
sensitivity = 1.0
[tune.808]
gain = 1.0
[tune.808@phonk]
gain = 1.8
[tempo@phonk]
bpm = 140
[smoothing@chill]
decay_ms = 800
"""


def test_profiles_merge_over_base(tmp_path):
    path = write_cfg(tmp_path, PROFILE_CFG)
    base = load(path)
    assert base.profiles == ["chill", "phonk"] and base.profile is None
    assert base.tune["808"]["gain"] == "1.0" and base.bpm is None
    phonk = load(path, "phonk")
    assert phonk.profile == "phonk"
    assert phonk.tune["808"]["gain"] == "1.8" and phonk.bpm == 140
    chill = load(path, "chill")
    assert chill.decay_ms == 800 and chill.tune["808"]["gain"] == "1.0"


def test_profile_from_general_and_unknown_profile(tmp_path):
    path = write_cfg(tmp_path, PROFILE_CFG.replace("sensitivity = 1.0", "profile = chill"))
    assert load(path).profile == "chill"
    assert load(path, "").profile is None
    with pytest.raises(SystemExit, match="unknown profile 'loud'.*chill, phonk"):
        load(path, "loud")


def test_profile_cli_flag(tmp_path):
    path = write_cfg(tmp_path, PROFILE_CFG)
    out = subprocess.run([sys.executable, "-m", "mathwave", "-p", path, "--profile", "phonk", "--dump", "5",
                          "--size", "160x48", "--no-color"],
                         capture_output=True, text=True, env=cli_env(tmp_path), check=True).stdout
    assert "profile phonk" in out
    assert "BPM≈140 (fixed)" in out


# ---------------------------------------------------------------- onsets config, CLI lists


def test_onset_config(tmp_path):
    cfg = load(write_cfg(tmp_path, "[onsets]\nkick = 30-120\nsnare = 180-400 + 2000-5000\n"
                                   "sensitivity = 2\nrefractory_ms = 90\nenvelope_ms = 150\n"))
    assert cfg.kick_bands == ((30, 120),)
    assert cfg.snare_bands == ((180, 400), (2000, 5000))
    assert (cfg.onset_sensitivity, cfg.refractory_ms, cfg.envelope_ms) == (2, 90, 150)
    with pytest.raises(ValueError):
        parse_ranges("30-", "kick")


def test_list_features_cli(tmp_path):
    path = write_cfg(tmp_path, "[bands]\nkickband = 40-90 peak\n[tune.808]\ndrive = band(30, 60)\n")
    out = subprocess.run([sys.executable, "-m", "mathwave", "-p", path, "--list-features"],
                         capture_output=True, text=True, env=cli_env(tmp_path), check=True).stdout
    for word in ["kickband", "band_30_60", "inline", "flatness", "balance", "pitch"]:
        assert word in out


def test_keys_cli(tmp_path):
    out = subprocess.run([sys.executable, "-m", "mathwave", "--keys"],
                         capture_output=True, text=True, env=cli_env(tmp_path), check=True).stdout
    assert "tap tempo" in out and "meter" in out and "profiles" in out


# ---------------------------------------------------------------- shipped config


def _uncommented_config(tmp_path):
    """The shipped config with every commented example section switched on in place."""
    path = os.path.join(os.path.dirname(__file__), "..", "config", "config")
    out, block = [], False
    for line in open(path).read().splitlines():
        if line.startswith("# [") and line.rstrip().endswith("]"):
            block = True
            out.append(line[2:])
        elif block and line.startswith("# ") and "=" in line:
            out.append(line[2:])
        else:
            block = False
            out.append(line)
    return write_cfg(tmp_path, "\n".join(out) + "\n", "all-on")


def test_shipped_config_every_example_section_loads(tmp_path):
    path = _uncommented_config(tmp_path)
    cfg = load(path)
    assert {"kickband", "snarebody", "vocal"} <= set(cfg.bands)
    assert cfg.profiles == ["chill", "phonk"]
    from mathwave.panels import DEFAULT_ORDER
    assert set(cfg.tune) >= set(DEFAULT_ORDER) | {"bell"}  # an example for every panel
    for prof in [None] + cfg.profiles:
        c = load(path, prof)
        an, panels = build(c)
        f = an.analyze(OfflineDemoSource(RATE).latest_stereo(an.need, 30.0), 30.0)
        for p in panels:
            p.update(f, 1 / 60)
    assert load(path, "phonk").bpm == 140
    assert load(path, "chill").theme == "chroma"


def test_shipped_config_defaults_match_code(tmp_path):
    from mathwave.config import Config
    cfg, base = load(os.path.join(os.path.dirname(__file__), "..", "config", "config")), Config()
    for key in ["kick_bands", "snare_bands", "hat_bands", "onset_sensitivity", "refractory_ms", "envelope_ms",
                "bpm", "bpm_multiplier", "bpm_range", "fft_size", "fall_db_per_s", "attack", "decay"]:
        assert getattr(cfg, key) == getattr(base, key), key


def test_reload_error_note():
    from mathwave.cli import _error_note
    assert _error_note(SystemExit("mathwave: config error in /x/config: band name 'rms' is reserved")) == \
        "config error: band name 'rms' is reserved"
    assert _error_note(SystemExit("mathwave: [tune.tan] unknown key 'freqmax'; did you mean freq_max?")) == \
        "config error: [tune.tan] unknown key 'freqmax'"

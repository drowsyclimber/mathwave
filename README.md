# mathwave

A Cava-style terminal visualizer for Arch Linux. It listens to whatever your system is playing and drives a grid of math functions with it. Every panel plots one formula, the music moves the formula's parameters, and the formula is printed under the plot with its live values.

The idea comes from the "math functions that go hard" reels: an 808 from an exponential, a dive bomb from 1/x, a Reese from a trig identity, a screech from tan(x). mathwave runs that in reverse. Instead of turning formulas into sound, it turns sound back into the formulas.

![mathwave grid](docs/mathwave-grid.png)

## Panels

| # | Key | Formula plotted | Driven by |
|---|-----|-----------------|-----------|
| 1 | `808` | `y = A·e^(−kx)·sin(2π·f(x)·x)`, `f(x) = f₁ + (f₀ − f₁)·e^(−x/τ)` | sub 20–60 Hz, redrawn on every kick |
| 2 | `dive` | `y = a / (x − p)` | bass 60–250 Hz, asymptote sweeps on the beat |
| 3 | `reese` | `sin(ax) + sin(bx) = 2·sin((a+b)x/2)·cos((a−b)x/2)` | low mids 250–500 Hz set the detune |
| 4 | `tan` | `y = clip(A·tan(ωx + φ), −1, 1)` | presence 4–6 kHz |
| 5 | `fm` | `y = sin(ωc·x + I·sin(ωm·x + φ))` | high mids 2–4 kHz set the index, tempo sets φ |
| 6 | `saw` | `y = (2/π)·Σₖ₌₁ᴺ(−1)^(k+1)·sin(kx)/k` | brightness (spectral centroid) sets N |
| 7 | `sinc` | `y = A·sin(πωx)/(πωx)` | air 6–16 kHz |
| 8 | `logistic` | `xₙ₊₁ = r·xₙ·(1 − xₙ)` | loudness sets r, so loud passages go chaotic |
| 9 | `spectrum` | `X[k] = Σₙ x[n]·w[n]·e^(−2πikn/N)` | the classic Cava bars |

Each panel's title bar has a level meter for its drive signal. Dim curves are guides: envelopes, asymptotes, the ideal saw, the logistic fixed point.

## Install

```sh
# from this folder (the tarball sits next to the PKGBUILD)
makepkg -si
```

Runtime dependencies are `python` and `python-numpy`. For capture you need one of `libpulse` (provides `parec`, works with pipewire-pulse) or `pipewire` (provides `pw-record`). Most Arch desktops already have both.

`packaging/PKGBUILD.aur` builds from a tagged GitHub release and `packaging/PKGBUILD-git` builds from the repository head, for when you publish it.

## Use

```sh
mathwave                      # capture the default output's monitor, like cava
mathwave --demo --audible     # play the built-in track (every sound is a formula) and watch it
mathwave --panels 808,dive,reese,tan --layout 2x2
mathwave --focus 3            # start with the Reese panel full screen
mathwave --formulas           # print every formula, its drive and what it sounds like
mathwave --render-wav demo.wav
mathwave -i fifo --fifo /tmp/mpd.fifo
```

Keys: `1`–`9` full-screen a panel (shows the sound, drive and tuning too), `0` full-screens panel 10, the same key again or `g` returns to the grid, `f` cycle formula detail, `r` reload the config, `t` next theme, `+`/`-` gain, `p` pause, `q` quit.

Themes: `phonk`, `ocean`, `ember`, `matrix`, `mono`, and `wal`, which reads `~/.cache/wal/colors.json` so it follows a pywal-based Hyprland rice.

## Config

Copy `/usr/share/doc/mathwave/config.example` to `~/.config/mathwave/config`. The sections mirror Cava's: `[general]`, `[input]`, `[color]`, `[smoothing]`.

`fft_size` under `[smoothing]` (or `--fft-size`) sets the analysis window. The default 2048 is the snappiest; 4096 keeps bass notes around 80 Hz out of the sub band, which helps panels that listen to `sub`.

You can add your own function panels. The shipped config already enables one, a Gaussian bell driven by bass, as panel 10:

```ini
[general]
panels = 808, dive, reese, tan, fm, saw, sinc, logistic, spectrum, bell

[panel.bell]
title = bell · gaussian
expr = A*exp(-x^2/(2*s^2))
params = A = 0.15 + 0.85*level, s = 0.06 + 0.30*level
domain = -1, 1
yrange = -0.05, 1.05
drive = bass
```

Louder bass makes the bell taller (A) and wider (s). The panel prints `y = A·exp(-x^2/(2·s^2))` with the live A and s values. `level` is the panel's drive after any `[tune.bell]` shaping, so custom panels tune the same way as built-ins.

## Per-panel tuning

Every panel, built-in or custom, can have a `[tune.<panel>]` section. Press `r` while mathwave runs to reload the config and see the change immediately. Full-screen a panel (`1`–`9`, `0`) to see its live level and active tuning.

```ini
[tune.808]
drive = sub          # what it listens to: a band, rms, centroid, or max(sub, 0.7*bass)
curve = 2            # 1 linear, 2 punchier, 0.5 more sensitive
decay = 0.85         # this panel's own fall speed
tau = 0.3            # an 808 knob: the pitch-drop time in its formula
reveal = 0.25        # an 808 knob: how fast the curve redraws after each kick
```

Keys every panel accepts, applied in this order:

| Key | Effect | Default |
|---|---|---|
| `drive` | what the panel listens to | per panel (see `--list-knobs`) |
| `gain` | multiply the level | 1 |
| `threshold` | ignore anything below this, 0..1 | 0 |
| `curve` | exponent on the level | 1 |
| `min`, `max` | the range the panel moves within (min > max inverts) | 0, 1 |
| `attack`, `decay` | this panel's own smoothing | the `[smoothing]` values |

Each built-in also has knobs for the constants in its own formula, such as `index_max` and `beats` on the FM wobble or `r_min` and `r_max` on the logistic map. `mathwave --list-knobs` prints all of them with defaults and descriptions, and the shipped config has a commented example for every panel. A typo in a key is rejected with a suggestion, for example `unknown key 'freqmax' (did you mean freq_max?)`.

Expressions are checked before they run: only arithmetic, numeric constants, the drive names (`sub bass lowmid mid highmid presence air rms centroid t beat`) and a fixed list of math functions are allowed.

## How it works

1. Capture: `parec --device=@DEFAULT_MONITOR@` or `pw-record -P '{ stream.capture.sink = true }' -` streams raw 16-bit PCM into a ring buffer. `--raw` is passed to `pw-record` only on PipeWire versions that support it.
2. Analysis: a 2048-point Hann-windowed FFT every frame. Each band is converted to dB and normalized against its own recent peak with a 40 dB window, so quiet bands still move but silence reads as zero. Rises use an attack factor and falls use Cava-style gravity. Kicks are detected from spectral flux below 150 Hz, and the spacing between kicks gives the BPM.
3. Panels: each panel evaluates its drive expression, shapes it with its tuning (gain, threshold, curve, range, smoothing), maps the result onto its formula's parameters through its knobs, evaluates the formula with numpy across the panel width, and hands back curves.
4. Rendering: curves are drawn into a dot grid at 2×4 dots per character and encoded as Unicode braille, then written in one synchronized ANSI frame. A 200×60 terminal renders in about 7 ms per frame.

## Development

```sh
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m mathwave --dump 240 --at 38 --size 168x50 > frame.ans   # headless frame
```

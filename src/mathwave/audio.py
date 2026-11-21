"""Audio inputs: PulseAudio/PipeWire monitor capture, FIFO/stdin, and the demo synth."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

from . import synth


class RingBuffer:
    def __init__(self, size: int):
        self.buf = np.zeros(size, dtype=np.float32)
        self.pos = 0
        self.lock = threading.Lock()

    def write(self, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=np.float32)
        n = len(self.buf)
        if len(data) >= n:
            data = data[-n:]
        with self.lock:
            end = self.pos + len(data)
            if end <= n:
                self.buf[self.pos : end] = data
            else:
                k = n - self.pos
                self.buf[self.pos :] = data[:k]
                self.buf[: end - n] = data[k:]
            self.pos = end % n

    def latest(self, count: int) -> np.ndarray:
        with self.lock:
            return np.roll(self.buf, -self.pos)[-count:].copy()


def pcm_to_mono(raw: bytes, channels: int, fmt: str = "s16le") -> np.ndarray:
    if fmt == "f32le":
        a = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    else:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        usable = len(a) - (len(a) % channels)
        a = a[:usable].reshape(-1, channels).mean(axis=1)
    return a


class Source:
    name = "silence"

    def __init__(self, rate: int = 44100, channels: int = 2):
        self.rate = rate
        self.channels = channels
        self.ring = RingBuffer(rate * 2)
        self.error: str | None = None
        self.alive = True

    def start(self) -> "Source":
        return self

    def stop(self) -> None:
        self.alive = False

    def latest(self, count: int, t: float | None = None) -> np.ndarray:
        return self.ring.latest(count)


class StreamSource(Source):
    """Reads raw PCM from any binary stream on a background thread."""

    chunk_frames = 512

    def __init__(self, stream, rate=44100, channels=2, fmt="s16le", label="stream"):
        super().__init__(rate, channels)
        self.stream = stream
        self.fmt = fmt
        self.name = label
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        width = 4 if self.fmt == "f32le" else 2
        nbytes = self.chunk_frames * self.channels * width
        frame = self.channels * width
        leftover = b""
        try:
            while self.alive:
                data = self.stream.read1(nbytes) if hasattr(self.stream, "read1") else self.stream.read(nbytes)
                if not data:
                    self.error = self.error or f"{self.name}: input ended"
                    break
                data = leftover + data
                cut = len(data) - (len(data) % frame)
                leftover = data[cut:]
                if cut:
                    self.ring.write(pcm_to_mono(data[:cut], self.channels, self.fmt))
        except Exception as exc:  # pragma: no cover - depends on the host
            self.error = f"{self.name}: {exc}"
        self.alive = False


class ProcessSource(StreamSource):
    def __init__(self, cmd: list[str], rate=44100, channels=2, fmt="s16le", label="process"):
        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
        super().__init__(self.proc.stdout, rate, channels, fmt, label)

    def _run(self):
        super()._run()
        if self.proc.poll() is not None and self.proc.returncode not in (0, None):
            err = (self.proc.stderr.read() or b"").decode(errors="replace").strip().splitlines()
            tail = err[-1] if err else f"exit code {self.proc.returncode}"
            self.error = f"{self.name}: {tail}"

    def stop(self):
        super().stop()
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class DemoSource(Source):
    """Real-time playback of the formula-only demo track, optionally audible."""

    name = "demo"

    def __init__(self, rate=44100, audible=False):
        super().__init__(rate, 1)
        self.audible = audible
        self.player: subprocess.Popen | None = None
        self.t0 = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _open_player(self):
        candidates = [
            ["pw-play", "--rate", str(self.rate), "--channels", "1", "--format", "s16", "-"],
            ["paplay", "--raw", f"--rate={self.rate}", "--channels=1", "--format=s16le"],
            ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(self.rate), "-c", "1"],
        ]
        for cmd in candidates:
            if shutil.which(cmd[0]):
                return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.error = "demo: no player found (install pipewire, libpulse or alsa-utils)"
        return None

    def start(self):
        if self.audible:
            self.player = self._open_player()
        self.thread.start()
        return self

    def _run(self):
        block = 512
        n = 0
        self.t0 = time.monotonic()
        while self.alive:
            t = (n + np.arange(block)) / self.rate
            y = synth.render(t)
            self.ring.write(y)
            if self.player and self.player.stdin:
                try:
                    self.player.stdin.write((y * 32767).astype("<i2").tobytes())
                except (BrokenPipeError, ValueError):
                    self.player = None
            n += block
            ahead = n / self.rate - (time.monotonic() - self.t0)
            if ahead > 0.02:
                time.sleep(ahead - 0.02)

    def stop(self):
        super().stop()
        if self.player:
            try:
                self.player.stdin.close()
            except Exception:
                pass
            self.player.terminate()


class OfflineDemoSource(Source):
    """Deterministic demo for headless rendering and tests: samples come from t."""

    name = "demo (offline)"

    def latest(self, count, t=None):
        t = 0.0 if t is None else t
        times = t - (np.arange(count)[::-1] / self.rate)
        return synth.render(np.maximum(times, 0.0))


def _pw_record_cmd(rate: int, channels: int, target: str | None) -> list[str]:
    cmd = []
    if shutil.which("stdbuf"):
        cmd += ["stdbuf", "-o0"]
    cmd += ["pw-record"]
    try:
        helptext = subprocess.run(["pw-record", "--help"], capture_output=True, text=True, timeout=2).stdout
    except Exception:
        helptext = ""
    if "--raw" in helptext:
        cmd.append("--raw")  # PipeWire >= 1.4 writes a header to '-' unless --raw is given
    cmd += ["--rate", str(rate), "--channels", str(channels), "--format", "s16", "--latency", "20ms"]
    if target and target != "auto":
        cmd += ["--target", target]
    else:
        cmd += ["-P", "{ stream.capture.sink = true node.name = mathwave }"]
    cmd.append("-")
    return cmd


def _parec_cmd(rate: int, channels: int, device: str | None) -> list[str]:
    dev = device if device and device != "auto" else "@DEFAULT_MONITOR@"
    return [
        "parec", f"--device={dev}", "--raw", "--format=s16le", f"--rate={rate}",
        f"--channels={channels}", "--latency-msec=15", "--client-name=mathwave",
        "--stream-name=visualizer",
    ]


def build_source(method: str, device: str | None = None, rate: int = 44100, channels: int = 2,
                 fifo: str | None = None, audible: bool = False) -> tuple[Source, str | None]:
    """Returns (source, note). Note explains any fallback taken."""
    method = (method or "auto").lower()
    note = None
    if method == "auto":
        if shutil.which("parec"):
            method = "pulse"
        elif shutil.which("pw-record"):
            method = "pipewire"
        else:
            method = "demo"
            note = "no parec or pw-record found, playing the demo (install libpulse or pipewire)"

    if method == "pulse":
        src = ProcessSource(_parec_cmd(rate, channels, device), rate, channels, label="pulse")
    elif method == "pipewire":
        src = ProcessSource(_pw_record_cmd(rate, channels, device), rate, channels, label="pipewire")
    elif method == "fifo":
        path = fifo or "/tmp/mpd.fifo"
        if not os.path.exists(path):
            raise SystemExit(f"mathwave: fifo {path} does not exist")
        src = StreamSource(open(path, "rb", buffering=0), rate, channels, label=f"fifo {path}")
    elif method == "stdin":
        src = StreamSource(sys.stdin.buffer, rate, channels, label="stdin")
    elif method == "demo":
        src = DemoSource(rate, audible=audible)
    elif method == "offline":
        src = OfflineDemoSource(rate, 1)
    else:
        raise SystemExit(f"mathwave: unknown input method '{method}'")
    return src.start(), note

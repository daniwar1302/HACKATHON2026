"""Live beat tracking from call audio, numpy only.

Feed it mono PCM as it arrives; it keeps a few seconds of onset strength (spectral flux),
finds the tempo by autocorrelation (70-180 BPM, gently preferring ~120) and the beat phase
with a comb filter, and reports when the last beat landed, in wall-clock time, so other
processes on the machine can predict the next beats.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

HOP_S = 0.02          # onset-envelope rate: 50 Hz
WINDOW_S = 8.0        # history used for tempo
MIN_BPM, MAX_BPM = 70.0, 180.0


@dataclass
class Beat:
    bpm: float
    beat_t: float        # wall-clock time of a recent beat
    confidence: float    # 0..1, how periodic the onsets are
    loudness: float      # RMS of the last second, 0..1 (full scale = 1)


class BeatTracker:
    def __init__(self, sample_rate: int = 24000, clock=time.time):
        self.sr, self.clock = sample_rate, clock
        self.hop = int(sample_rate * HOP_S)
        self.n_fft = 2 * self.hop
        self.window = np.hanning(self.n_fft)
        self.buf = np.zeros(0, dtype=np.float32)
        self.prev_spec = None
        self.onsets: deque[float] = deque(maxlen=int(WINDOW_S / HOP_S))
        self.rms: deque[float] = deque(maxlen=int(1.0 / HOP_S))
        self.last_frame_t = 0.0

    def feed(self, pcm16: bytes) -> None:
        """Append little-endian 16-bit mono samples."""
        x = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        self.buf = np.concatenate([self.buf, x])
        while len(self.buf) >= self.n_fft:
            frame = self.buf[: self.n_fft]
            self.buf = self.buf[self.hop:]
            spec = np.log1p(np.abs(np.fft.rfft(frame * self.window)) * 10)
            flux = 0.0 if self.prev_spec is None else float(np.maximum(spec - self.prev_spec, 0).sum())
            self.prev_spec = spec
            self.onsets.append(flux)
            self.rms.append(float(np.sqrt(np.mean(frame ** 2))))
        self.last_frame_t = self.clock()

    def estimate(self) -> Beat | None:
        """Current tempo and beat phase, or None until there are ~4 s of audio."""
        n = len(self.onsets)
        if n < int(4.0 / HOP_S):
            return None
        env = np.asarray(self.onsets)
        env = env - env.mean()
        if not env.any():
            return None
        ac = np.correlate(env, env, mode="full")[n - 1:]
        fps = 1.0 / HOP_S
        lags = np.arange(int(fps * 60 / MAX_BPM), int(fps * 60 / MIN_BPM) + 1)
        bpms = 60 * fps / lags
        prior = np.exp(-0.5 * (np.log2(bpms / 120.0) / 0.9) ** 2)  # mild preference for ~120 BPM
        scores = ac[lags] * prior
        i = int(np.argmax(scores))
        lag = float(lags[i])
        if 0 < i < len(lags) - 1:  # parabolic refinement between lag samples
            a, b, c = scores[i - 1], scores[i], scores[i + 1]
            denom = a - 2 * b + c
            if denom != 0:
                lag += 0.5 * (a - c) / denom
        confidence = float(np.clip(ac[lags[i]] / ac[0], 0, 1)) if ac[0] > 0 else 0.0

        # Phase: which offset within one period collects the most onset energy (last 4 s).
        recent = np.asarray(self.onsets)[-int(4.0 / HOP_S):]
        period = int(round(lag))
        comb = [recent[p::period].sum() for p in range(period)]
        phase = int(np.argmax(comb))
        last = phase + period * ((len(recent) - 1 - phase) // period)  # newest frame index on the beat grid
        frames_ago = len(recent) - 1 - last
        beat_t = self.last_frame_t - frames_ago * HOP_S - self.n_fft / self.sr / 2
        loud = float(np.sqrt(np.mean(np.square(self.rms)))) if self.rms else 0.0
        return Beat(round(60 * fps / lag, 1), beat_t, round(confidence, 2), round(min(loud * 4, 1.0), 2))

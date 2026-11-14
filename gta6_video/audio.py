"""Генерує саундтрек: синтвейв-музика + звукові ефекти + озвучка з ducking."""
import os
import wave

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from timeline import SCENES, TOTAL, HERE, VOICE_DIR

SR = 44100
rng = np.random.default_rng(7)


def read_wav(path):
    with wave.open(path) as w:
        assert w.getsampwidth() == 2
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        if w.getnchannels() == 2:
            data = data.reshape(-1, 2).mean(1)
        assert w.getframerate() == SR
    return data


def lp(x, f, order=2):
    return sosfilt(butter(order, f / (SR / 2), "low", output="sos"), x)


def hp(x, f, order=2):
    return sosfilt(butter(order, f / (SR / 2), "high", output="sos"), x)


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo / (SR / 2), hi / (SR / 2)], "band", output="sos"), x)


def saw(freq, n, phase=0.0):
    t = np.arange(n) / SR
    return 2 * ((t * freq + phase) % 1.0) - 1


def note(n):  # MIDI -> Hz
    return 440.0 * 2 ** ((n - 69) / 12)


def music(total):
    N = int(total * SR) + SR
    L = np.zeros(N, np.float32)
    R = np.zeros(N, np.float32)
    bpm = 96
    beat = 60 / bpm
    bar = beat * 4
    # Am - F - C - G
    chords = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]
    roots = [45, 41, 48, 43]
    nbars = int(total / bar) + 2

    def add(sig, t0, gain=1.0, pan=0.0):
        i0 = int(t0 * SR)
        if i0 >= N:
            return
        sig = sig[: N - i0]
        L[i0:i0 + len(sig)] += sig * gain * (1 - max(0, pan))
        R[i0:i0 + len(sig)] += sig * gain * (1 + min(0, pan))

    # пад
    for b in range(nbars):
        c = chords[b % 4]
        n = int(bar * SR)
        env = np.minimum(1, np.arange(n) / (0.25 * SR)) * np.minimum(1, (n - np.arange(n)) / (0.2 * SR))
        for k, m in enumerate(c):
            for det, pan in ((-0.12, -0.6), (0.0, 0.0), (0.12, 0.6)):
                s = saw(note(m + 12) * 2 ** (det / 12), n, rng.random())
                add(lp(s, 1800).astype(np.float32) * env, b * bar, 0.035, pan)
    # бас (восьмі)
    for b in range(nbars):
        r = note(roots[b % 4])
        for e in range(8):
            n = int(beat / 2 * SR)
            t = np.arange(n) / SR
            env = np.exp(-t * 6) * np.minimum(1, t / 0.005)
            s = saw(r, n) * 0.6 + np.sin(2 * np.pi * r / 2 * t) * 0.6
            add(lp(s, 500).astype(np.float32) * env, b * bar + e * beat / 2, 0.22)
    # арпеджіо (шістнадцяті)
    for b in range(nbars):
        c = chords[b % 4]
        seq = [c[0] + 24, c[1] + 24, c[2] + 24, c[1] + 24]
        for s16 in range(16):
            n = int(beat / 4 * SR)
            t = np.arange(n) / SR
            f = note(seq[s16 % 4])
            sq = np.sign(np.sin(2 * np.pi * f * t)) * 0.5 + saw(f, n) * 0.5
            env = np.exp(-t * 18)
            add(lp(sq, 3000).astype(np.float32) * env, b * bar + s16 * beat / 4, 0.03, 0.4 if s16 % 2 else -0.4)
    # ударні
    kn = int(0.45 * SR)
    t = np.arange(kn) / SR
    kick = np.sin(2 * np.pi * (45 * t + (110 / 18) * (1 - np.exp(-t * 18)))) * np.exp(-t * 7)
    sn = int(0.3 * SR)
    t2 = np.arange(sn) / SR
    snare = bp(rng.standard_normal(sn), 900, 6000) * np.exp(-t2 * 14) * 0.8 + np.sin(2 * np.pi * 190 * t2) * np.exp(-t2 * 25) * 0.4
    hn = int(0.06 * SR)
    hat = hp(rng.standard_normal(hn), 7000) * np.exp(-np.arange(hn) / SR * 70)
    for b in range(nbars):
        bt = b * bar
        intro = bt < 2 * bar  # перші такти — лише пад
        for q in range(4):
            if not intro:
                add(kick.astype(np.float32), bt + q * beat, 0.55)
                if q in (1, 3):
                    add(snare.astype(np.float32), bt + q * beat, 0.30)
            for h in range(2):
                add(hat.astype(np.float32), bt + q * beat + h * beat / 2, 0.06 if not intro else 0.03, 0.3)
    # простий реверб для простору (дифузні затримки)
    for d, g in ((0.113, 0.25), (0.171, 0.2), (0.237, 0.15)):
        k = int(d * SR)
        L[k:] += R[:-k] * g
        R[k:] += L[:-k] * g * 0.9
    # фейд-ін / фейд-аут
    env = np.ones(N, np.float32)
    fi = int(1.0 * SR)
    env[:fi] = np.linspace(0, 1, fi)
    fo = int(3.0 * SR)
    end = int(total * SR)
    env[end - fo:end] = np.linspace(1, 0, fo)
    env[end:] = 0
    return np.stack([L * env, R * env], 1)[: int(total * SR)]


def whoosh(dur=0.7):
    n = int(dur * SR)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    out = np.zeros(n)
    # розгортка смуги частот
    seg = 32
    for k in range(seg):
        a, b = k * n // seg, (k + 1) * n // seg
        f = 300 + 5000 * (k / seg) ** 1.5
        out[a:b] = bp(noise, max(80, f * 0.6), min(19000, f * 1.6))[a:b]
    env = np.sin(np.pi * t / dur) ** 2
    return (out * env * 0.5).astype(np.float32)


def impact(dur=1.6):
    n = int(dur * SR)
    t = np.arange(n) / SR
    boom = np.sin(2 * np.pi * (38 * t + 60 / 8 * (1 - np.exp(-t * 8)))) * np.exp(-t * 3)
    crack = hp(rng.standard_normal(n), 2000) * np.exp(-t * 30) * 0.3
    return (boom + crack).astype(np.float32) * 0.8


def tick():
    n = int(0.03 * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 2400 * t) * np.exp(-t * 200)).astype(np.float32) * 0.25


def main():
    total = TOTAL
    N = int(total * SR)
    mus = music(total)

    voice = np.zeros(N, np.float32)
    for s in SCENES:
        v = read_wav(os.path.join(VOICE_DIR, f"s{s['i']}.wav"))
        i0 = int(s["voice_at"] * SR)
        voice[i0:i0 + len(v)] += v[: N - i0]

    sfx = np.zeros(N, np.float32)

    def put(sig, t0, g=1.0):
        i0 = int(t0 * SR)
        sig = sig[: max(0, N - i0)]
        sfx[i0:i0 + len(sig)] += sig * g

    w = whoosh()
    for s in SCENES[1:]:
        put(w, s["start"] - 0.4, 0.8)
    put(impact(), 0.35, 0.9)             # поява заголовка
    put(impact(), SCENES[7]["start"] + 0.4, 0.8)  # фінальна дата
    put(impact(), SCENES[6]["start"] + 3.0, 0.4)  # штамп "ПЕРЕНЕСЕНО"
    put(impact(), SCENES[6]["start"] + 5.4, 0.4)
    put(impact(), SCENES[3]["start"] + 2.8, 0.4)  # штамп "РЕКОРД"

    # ducking музики під голос
    env = np.abs(voice)
    win = int(0.25 * SR)
    env = np.convolve(env, np.ones(win) / win, mode="same")
    duck = 1 - np.clip(env / 0.05, 0, 1) * 0.55
    duck = lp(duck, 4).astype(np.float32)

    mix = mus * (duck[:, None] * 0.9)
    mix += voice[:, None] * 0.95
    mix += sfx[:, None] * 0.6
    peak = np.max(np.abs(mix))
    mix = np.tanh(mix / peak * 1.3) * 0.89
    out = (mix * 32767).astype(np.int16)
    path = os.path.join(HERE, "build", "soundtrack.wav")
    with wave.open(path, "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(out.tobytes())
    print("soundtrack", path, f"{total:.2f}s")


if __name__ == "__main__":
    main()

"""Таймлайн відео: тривалість кожної сцени підлаштовується під озвучку."""
import os
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR = os.path.join(HERE, "build", "voice")

FPS = 30
W, H = 1920, 1080

VOICE_LEAD = [1.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]   # пауза перед голосом у сцені
TAIL = [1.0, 1.1, 1.1, 1.1, 1.1, 1.1, 1.2, 3.2]         # пауза після голосу


def wav_duration(path):
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def build():
    scenes = []
    t = 0.0
    for i in range(8):
        vd = wav_duration(os.path.join(VOICE_DIR, f"s{i}.wav"))
        dur = VOICE_LEAD[i] + vd + TAIL[i]
        scenes.append({"i": i, "start": t, "dur": dur, "voice_at": t + VOICE_LEAD[i]})
        t += dur
    return scenes, t


SCENES, TOTAL = build()

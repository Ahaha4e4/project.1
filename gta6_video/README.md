# Відео «Чому GTA 6 — найочікуваніша гра у світі»

Motion-графіка генерується процедурно (Python + Pillow + numpy), озвучка — чоловічий голос RHVoice (Anatol, українська),
музика та звукові ефекти синтезуються в `audio.py`.

```bash
apt-get install -y rhvoice rhvoice-ukrainian ffmpeg
pip install numpy scipy pillow
./make_voice.sh          # озвучка по сценах (тексти в narration.py)
python3 audio.py         # музика + SFX + ducking -> build/soundtrack.wav
python3 render.py        # кадри -> build/video_noaudio.mp4
ffmpeg -i build/video_noaudio.mp4 -i build/soundtrack.wav -c:v libx264 -crf 25 -c:a aac -b:a 192k -shortest GTA6.mp4
```

`python3 render.py preview 2.5 12 39` — зберегти окремі кадри для перевірки.

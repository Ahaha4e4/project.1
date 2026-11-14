#!/bin/bash
# Генерує озвучку по сценах чоловічим голосом RHVoice (Anatol, українська)
set -e
cd "$(dirname "$0")"; mkdir -p build/voice
python3 -c "
from narration import SCENES
for i,s in enumerate(SCENES): open(f'build/voice/s{i}.txt','w').write(s)"
for f in build/voice/s*.txt; do
  b=${f%.txt}
  RHVoice-test -p anatol -r 108 -t 92 -o $b.raw.wav < $f
  # легка обробка: прибрати тишу, компресія, EQ для "дикторського" тембру
  ffmpeg -loglevel error -y -i $b.raw.wav -af "silenceremove=start_periods=1:start_threshold=-45dB,areverse,silenceremove=start_periods=1:start_threshold=-45dB,areverse,highpass=f=70,equalizer=f=180:t=q:w=1:g=3,equalizer=f=3500:t=q:w=1:g=2,acompressor=threshold=-18dB:ratio=3:attack=5:release=80,loudnorm=I=-16:TP=-1.5,aresample=44100" -ac 1 $b.wav
  rm $b.raw.wav
done
for f in build/voice/s?.wav; do echo "$f $(ffprobe -v error -show_entries format=duration -of csv=p=0 $f)"; done

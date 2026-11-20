"""List wordless vocal moments (screams, shouts, sighs, laughs) as candidates for expression tags
like *SCREAMS* or *SIGHS*. Claude reviews them and adds the right ones to clips.json "tags".

    python run.py sounds <project>                  every candidate in the video
    python run.py sounds <project> --clip 1         only inside clip 1's ranges

How it works (cheap, CPU-light): loudness of the voice-band audio in 0.1 s steps. A stretch with
no transcribed word that is much louder than the speaker's normal talking is a likely scream /
shout; a softer wordless stretch is a possible sigh / laugh / groan. Game sounds can trigger it
too, so check the facecam (mouth open, reaction) before tagging.
"""
import argparse
import wave

import numpy as np

from .common import fmt_ts, load_json, project_dir

HOP = 0.1


def loudness(wav):
    w = wave.open(str(wav))
    sr = w.getframerate()
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    n = int(sr * HOP)
    frames = a[:len(a) // n * n].reshape(-1, n)
    return 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)


def candidates(db, words, lo=None, hi=None, min_len=0.3):
    """[(start, end, peak_db, kind)] wordless stretches louder than the background."""
    spoken = np.zeros(len(db), bool)
    for w in words:
        spoken[int(w['s'] / HOP):int(w['e'] / HOP) + 1] = True
    talk = db[spoken] if spoken.any() else db
    loud = np.percentile(talk, 60)                       # speaker's normal talking level
    floor = np.percentile(db, 30)                        # background level
    active = (db > max(floor + 10, loud - 14)) & ~spoken
    out, i = [], 0
    while i < len(db):
        if not active[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(db) and (active[j + 1] or (j + 2 < len(db) and active[j + 2])):
            j += 1
        a, b = i * HOP, (j + 1) * HOP
        if b - a >= min_len and (lo is None or (b > lo and a < hi)):
            peak = float(db[i:j + 1].max())
            kind = 'SCREAMS / SHOUTS' if peak >= loud + 2 else 'soft: SIGHS / LAUGHS / GROANS?'
            out.append((round(a, 1), round(b, 1), round(peak, 1), kind))
        i = j + 1
    return out, float(loud)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('project')
    ap.add_argument('--clip', type=int, default=None, help='only inside this clip id')
    a = ap.parse_args()
    pdir = project_dir(a.project)
    wav = pdir / 'source' / 'audio16k_voice.wav'
    if not wav.exists():
        wav = pdir / 'source' / 'audio16k.wav'
    words = load_json(pdir / 'transcript' / 'words.json')
    db = loudness(wav)
    ranges = [(None, None)]
    if a.clip is not None:
        clip = next(c for c in load_json(pdir / 'clips.json')['clips'] if int(c['id']) == a.clip)
        ranges = clip['ranges']
    found = []
    for lo, hi in ranges:
        c, loud = candidates(db, words, lo, hi)
        found += c
    print(f'speaker talking level ~{loud:.0f} dB. Wordless vocal moments (source seconds):')
    for s, e, peak, kind in found:
        before = [w['w'] for w in words if w['e'] <= s][-4:]
        after = [w['w'] for w in words if w['s'] >= e][:4]
        print(f'  {s:7.1f}-{e:<7.1f} {fmt_ts(s):>8}  peak {peak:5.1f} dB  {kind:32s} '
              f'... {" ".join(before)} | {" ".join(after)} ...')
    if not found:
        print('  none found')
    print('\nAdd the real ones to the clip in clips.json:  "tags": [{"at": 10.2, "text": "SCREAMS"}]  '
          '(source seconds; optional "dur", default 1.2 s). Check the facecam first: game audio can trigger it.')


if __name__ == '__main__':
    main()

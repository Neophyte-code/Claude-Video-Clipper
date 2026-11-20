"""Sound-effect library: everything in assets/sfx/ is scanned automatically.

The role comes from the file name: whoosh / swoosh / swish -> whoosh,
hit / impact / boom -> hit, click / pop / tick -> click. Drop your own files in with those words
in the name and they are picked up. A file that holds several sounds (a "pack") is split at the
silences, and each sound becomes its own variant.
"""
import subprocess

import numpy as np

from .common import ASSETS, load_json, save_json

SFX_DIR = ASSETS / 'sfx'
CATALOG = SFX_DIR / 'catalog.json'
SR = 48000
ROLES = {'whoosh': ('whoosh', 'swoosh', 'swish', 'swipe'),
         'hit': ('hit', 'impact', 'boom'), 'click': ('click', 'pop', 'tick')}
EXTS = ('.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac')


def role_of(name):
    n = name.lower()
    for role, keys in ROLES.items():
        if any(k in n for k in keys):
            return role
    return None


def decode(path):
    r = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                       capture_output=True)
    return np.frombuffer(r.stdout, dtype=np.float32)


def segments(x, floor_db=-42, min_gap=0.25, min_len=0.15):
    """Split a file into separate sounds at silences; return [(start, end, peak)] in seconds."""
    hop = SR // 100
    n = len(x) // hop
    if n == 0:
        return []
    rms = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms / (rms.max() + 1e-12))
    on = db > floor_db
    segs, i = [], 0
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        quiet = 0
        while j < n and quiet < min_gap * 100:
            quiet = quiet + 1 if not on[j] else 0
            j += 1
        end = j - quiet
        if (end - i) / 100 >= min_len:
            pk = i + int(np.argmax(rms[i:end]))
            segs.append((i / 100, end / 100, pk / 100))
        i = j
    return segs


def scan():
    files = sorted(p for p in SFX_DIR.iterdir() if p.suffix.lower() in EXTS) if SFX_DIR.exists() else []
    sig = [[p.name, p.stat().st_size] for p in files]
    cat = load_json(CATALOG)
    if cat and cat.get('sig') == sig:
        return cat
    items = []
    for p in files:
        role = role_of(p.stem)
        if not role:
            continue
        x = decode(p)
        segs = segments(x)
        for k, (a, b, pk) in enumerate(segs[:12]):
            amp = float(np.abs(x[int(a * SR):int(b * SR)]).max() or 1e-6)
            items.append({'file': p.name, 'role': role, 'variant': k, 'start': round(a, 2),
                          'length': round(b - a, 2), 'peak': round(pk - a, 2),
                          'norm_db': round(-20 * np.log10(amp), 2)})
    cat = {'sig': sig, 'items': items}
    save_json(CATALOG, cat)
    return cat


# ---------- meme sounds (assets/sfx/meme/meme-<name>.mp3) ----------
# Short reaction sounds (vine boom, bone crack, rizz...). Claude picks them per clip in clips.json
# ("sfx": [{"word": "...", "sound": "vine-boom"}]); they are levelled by their loudest moment so
# every one sits at the same quiet level under the voice, whatever its source loudness.

MEME_DIR = SFX_DIR / 'meme'
MEME_CATALOG = MEME_DIR / 'catalog.json'


def _meme_info(path):
    x = decode(path)
    hop, win = SR // 100, SR // 10 * 3                     # 10 ms hop, 300 ms loudness window
    if len(x) < hop:
        return None
    n = len(x) // hop
    rms = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    on = np.nonzero(20 * np.log10(rms / rms.max()) > -40)[0]
    start = on[0] / 100 if len(on) else 0.0
    end = (on[-1] + 1) / 100 if len(on) else n / 100
    k = max(1, win // hop)
    loud = np.convolve(rms ** 2, np.ones(k) / k, mode='same') if n >= k else rms ** 2
    pk = int(np.argmax(rms))
    return {'start': round(start, 2), 'length': round(end - start, 2),
            'peak': round(max(0.0, pk / 100 - start), 2),
            'loud_db': round(10 * np.log10(loud.max() + 1e-12), 2)}


def memes():
    """{name: info} for every meme sound, cached until the folder changes."""
    files = sorted(p for p in MEME_DIR.iterdir() if p.suffix.lower() in EXTS) if MEME_DIR.exists() else []
    sig = [[p.name, p.stat().st_size] for p in files]
    cat = load_json(MEME_CATALOG)
    if cat and cat.get('sig') == sig:
        return cat['items']
    items = {}
    for p in files:
        info = _meme_info(p)
        if info:
            items[p.stem.removeprefix('meme-')] = {'file': f'meme/{p.name}', **info}
    save_json(MEME_CATALOG, {'sig': sig, 'items': items})
    return items


def pick(role, prefer_len=None, exclude=()):
    """Choose a sound of a role, preferring a length close to prefer_len (seconds)."""
    items = [i for i in scan()['items'] if i['role'] == role and (i['file'], i['variant']) not in exclude]
    if not items:
        return None
    if prefer_len:
        items.sort(key=lambda i: abs(i['length'] - prefer_len))
    return items[0]


if __name__ == '__main__':
    for it in scan()['items']:
        print(it)

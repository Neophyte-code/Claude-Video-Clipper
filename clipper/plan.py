"""Builds the edit plan for one clip: camera zooms, caption groups and sound-effect cues.

Both render engines (python and remotion) read the same plan, so they produce the same edit:
the camera is pre-computed per frame here, captions carry exact word times, and the audio mix
is shared. Only the drawing differs.

Zoom style (adapted from a tested short-form editing engine):
  * open on motion: 1.18 -> wide, a fast ease-out snap over 9 frames
  * every 3-5 s (on sentence ends) the framing alternates wide <-> punched-in, as a hard cut
  * camera cuts / spliced ranges always get a framing change (hides the jump)
  * one or two smooth "snap" zooms for emphasis
  * between changes a slow +0.03 push so the frame is never dead
  * scale is always around the speaker's face, so the face stays in place
"""
import re

from .common import FPS, H, W
from . import sfx as sfxlib

WIDE = [1.0, 1.04, 1.02]
PUNCH = [1.16, 1.22, 1.13, 1.2]
DRIFT = 0.03
OPEN_SCALE, OPEN_FRAMES = 1.18, 9
SNAP_FRAMES = 8

# ---------- easing (cubic-bezier, same curves as CSS / Remotion Easing.bezier) ----------


def bezier(x1, y1, x2, y2):
    def sample(a1, a2, t):
        return ((1 - 3 * a2 + 3 * a1) * t + (3 * a2 - 6 * a1)) * t * t + 3 * a1 * t

    def ease(x):
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if sample(x1, x2, mid) < x:
                lo = mid
            else:
                hi = mid
        return sample(y1, y2, (lo + hi) / 2)
    return ease


EASE_OUT = bezier(0.16, 1, 0.3, 1)     # "snap": fast then settles
EASE_IN_OUT = bezier(0.65, 0, 0.35, 1)
EASES = {'snap': EASE_OUT, 'smooth': EASE_IN_OUT}


def cam_at(keys, f):
    """Camera {s, x, y} at frame f from keyframes. A key's `ease` says how we ARRIVE at it:
    cut = jump at its frame, linear = glide from the previous key, snap/smooth = move during
    the `dur` frames that end at the key."""
    pi = 0
    for i, k in enumerate(keys):
        if k['f'] <= f:
            pi = i
    p = keys[pi]
    n = keys[pi + 1] if pi + 1 < len(keys) else None
    if n is None or f < p['f']:
        return p['s'], p['x'], p['y']
    k = 0.0
    if n['ease'] == 'linear':
        k = (f - p['f']) / max(1, n['f'] - p['f'])
    elif n['ease'] in EASES:
        d = n.get('dur', 12)
        r = min(1.0, max(0.0, (f - (n['f'] - d)) / d))
        k = EASES[n['ease']](r)
    return tuple(a + (b - a) * k for a, b in ((p['s'], n['s']), (p['x'], n['x']), (p['y'], n['y'])))


# ---------- helpers ----------

def focus_at(shots, t):
    for s in shots:
        if s['t0'] <= t < s['t1']:
            return s['focus']
    return shots[-1]['focus'] if shots else [W / 2, H * 0.4]


def beat_bounds(words, shots, splices, duration):
    """Times where the framing changes. Must-cuts: camera cuts and splices."""
    must = sorted({round(s['t0'], 3) for s in shots if s['t0'] > 0.3} | {s for s in splices if s > 0.3})
    soft = []
    for i, w in enumerate(words[:-1]):
        nxt = words[i + 1]
        if w['w'][-1:] in '.?!' or nxt['s'] - w['e'] > 0.45:
            soft.append(round(nxt['s'] - 0.03, 3))
    bounds, last = [], 0.0
    cands = sorted(set(must) | set(soft))
    for c in cands:
        if c >= duration - 1.2:
            break
        if c in must:
            if c - last >= 0.5:
                bounds.append(c)
                last = c
            continue
        nxt_must = next((m for m in must if m > c), duration)
        if c - last >= 2.8 and nxt_must - c >= 1.5:
            bounds.append(c)
            last = c
        elif c - last > 6.0:
            bounds.append(c)
            last = c
    # very long stretches with no sentence end: cut on a word start every ~4.5 s
    filled, prev = [], 0.0
    for b in bounds + [duration]:
        while b - prev > 7.0:
            target = prev + 4.5
            ws = min((w['s'] for w in words if w['s'] > prev + 3.0), key=lambda s: abs(s - target), default=None)
            if ws is None or ws >= b - 1.5:
                break
            filled.append(round(ws - 0.03, 3))
            prev = ws
        if b < duration:
            filled.append(b)
        prev = b
    return sorted(set(filled)), set(must)


def build_camera(meta):
    words, shots, dur = meta['words'], meta['shots'], meta['duration']
    fps = meta.get('fps', FPS)
    frames = int(round(dur * fps))
    bounds, must = beat_bounds(words, shots, meta.get('splices', []), dur)
    starts = [0.0] + bounds
    ends = bounds + [dur]
    keys, transitions = [], []
    wi = pi = 0
    snaps_left = 2 if dur > 25 else 1
    snap_after = dur * 0.3
    for b, (t0, t1) in enumerate(zip(starts, ends)):
        fx, fy = focus_at(shots, t0 + 0.05)
        f0, f1 = int(round(t0 * fps)), max(int(round(t0 * fps)) + 1, int(round(t1 * fps)) - 1)
        punched = b % 2 == 1
        if punched:
            s = PUNCH[pi % len(PUNCH)]
            pi += 1
        else:
            s = WIDE[wi % len(WIDE)]
            wi += 1
        if b == 0:
            keys.append({'f': 0, 's': OPEN_SCALE, 'x': fx, 'y': fy, 'ease': 'cut'})
            keys.append({'f': OPEN_FRAMES, 's': s, 'x': fx, 'y': fy, 'ease': 'snap', 'dur': OPEN_FRAMES})
            transitions.append({'t': 0.0, 'kind': 'open'})
        else:
            is_must = any(abs(t0 - m) < 0.05 for m in must)
            if punched and not is_must and snaps_left and t0 >= snap_after:
                keys.append({'f': f0 + SNAP_FRAMES, 's': s, 'x': fx, 'y': fy, 'ease': 'snap', 'dur': SNAP_FRAMES})
                transitions.append({'t': t0, 'kind': 'snap'})
                snaps_left -= 1
                snap_after = t0 + dur * 0.3
            else:
                keys.append({'f': f0, 's': s, 'x': fx, 'y': fy, 'ease': 'cut'})
                transitions.append({'t': t0, 'kind': 'cut'})
        keys.append({'f': f1, 's': round(s + DRIFT, 4), 'x': fx, 'y': fy, 'ease': 'linear'})
    cam = []
    for f in range(frames):
        s, x, y = cam_at(keys, f)
        cam.append([round(s, 4), round(x, 1), round(y, 1)])
    return keys, cam, transitions, list(zip(starts, ends))


# ---------- captions ----------

STRIP = re.compile(r'[.,;:]+$')


def display(word, upper):
    w = STRIP.sub('', word.strip())
    w = w.replace('"', '')
    return w.upper() if upper else w


def merge_tokens(words):
    """Whisper splits things like "$500" ",000" / "e" "-commerce" / "100" "%" into separate tokens."""
    out = []
    for w in words:
        t = w['w'].strip()
        if out and (t[:1] in ',%-&' or t.startswith('.0') or out[-1]['w'].endswith('-')) and len(t) > 0 \
                and w['s'] - out[-1]['e'] < 0.3:
            out[-1] = {**out[-1], 'w': (out[-1]['w'] + t).replace(' ', ''), 'e': w['e']}
        else:
            out.append(dict(w))
    return out


# words a caption group shouldn't end on: they lean on the next word ("OH MY | GOD", "FUCKING I | WAS")
WEAK_END = {'a', 'an', 'the', 'my', 'your', 'his', 'her', 'our', 'their', 'its', 'to', 'of', 'in', 'on',
            'at', 'for', 'with', 'from', 'by', 'and', 'or', 'but', 'so', 'i', "i'm", 'you', 'we', 'they',
            'is', 'are', 'was', 'be', 'can', 'will', 'gonna', 'about', 'some', 'this', 'that', 'if', 'oh'}


def _split_phrase(ph, max_words, max_chars, max_dur):
    """Split one phrase into caption groups: fewest groups that fit the limits, then the most even
    sizes, breaking after commas where possible and never leaving a weak word dangling at a group
    end or a lone word when it can be avoided (DP over break points)."""
    n = len(ph)

    def fits(i, j):            # words i..j-1 as one group
        if j - i == 1:
            return True
        chars = sum(len(c['text']) for c in ph[i:j]) + (j - i - 1)
        # span measured to the last word's onset: one drawn-out shouted word shouldn't split a group
        return j - i <= max_words and chars <= max_chars and ph[j - 1]['s'] - ph[i]['s'] <= max_dur

    def cost(i, j):
        c = 100 + (j - i - 2) ** 2                       # fewest groups first, then ~2-3 words each
        last = ph[j - 1]['raw']
        if j < n:                                        # inner break: prefer natural ones
            if last[-1:] in ',;:' or ph[j]['s'] - ph[j - 1]['e'] > 0.25:
                c -= 6                                   # comma or short pause
            elif last.lower().strip('.,!?"') in WEAK_END:
                c += 8
        if j - i == 1 and n > 1:
            c += 10                                      # lone word, unless it is the whole phrase
        return c
    best = [0.0] + [float('inf')] * n
    prev = [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(max(0, j - max_words), j):
            if best[i] < float('inf') and fits(i, j) and best[i] + cost(i, j) < best[j]:
                best[j], prev[j] = best[i] + cost(i, j), i
    cuts, j = [], n
    while j > 0:
        cuts.append((prev[j], j))
        j = prev[j]
    return [ph[i:j] for i, j in reversed(cuts)]


def _tok(t):
    return re.sub(r"[^\w']+", '', t.lower())


def apply_fixes(words, fixes):
    """Correct misheard words from clips.json "fix": [{"from": "give me bed", "to": "give me banned"}].
    The matched words keep their timing; a different word count is spread over the same span."""
    words = [dict(w) for w in words]
    for f in fixes or []:
        src = [_tok(t) for t in f['from'].split() if _tok(t)]
        dst = f['to'].split()
        if not src or not dst:
            continue
        i = 0
        while i + len(src) <= len(words):
            if [_tok(w['w']) for w in words[i:i + len(src)]] != src:
                i += 1
                continue
            old = words[i:i + len(src)]
            if len(dst) == len(old):
                new = [{**o, 'w': d} for o, d in zip(old, dst)]
            else:
                s, e = old[0]['s'], old[-1]['e']
                step = (e - s) / len(dst)
                new = [{'w': d, 's': round(s + k * step, 3), 'e': round(s + (k + 1) * step, 3), 'p': 1.0}
                       for k, d in enumerate(dst)]
            # keep the original punctuation on the last word ("bed," -> "banned,")
            tail = re.search(r'[.,!?]+$', old[-1]['w'])
            if tail and not re.search(r'[.,!?]$', new[-1]['w']):
                new[-1]['w'] += tail.group(0)
            words[i:i + len(src)] = new
            i += len(new)
    return words


def tag_groups(meta, upper=True, default_dur=1.2):
    """Expression tags from clips.json "tags": [{"at": 10.4, "text": "SCREAMS", "dur": 2.8}], with
    "at" in source seconds, as caption entries (*SCREAMS*) on the clip's timeline."""
    out, timeline = [], 0.0
    spans = []
    for a, b in meta.get('parts', []):
        spans.append((a, b, timeline))
        timeline += b - a
    for t in meta.get('tags', []) or []:
        at = float(t['at'])
        span = next(((a, b, off) for a, b, off in spans if a - 0.05 <= at < b), None)
        if not span:
            continue
        a, b, off = span
        s = at - a + off
        e = min(s + float(t.get('dur', default_dur)), b - a + off, meta['duration'])
        text = t['text'].strip('* ')
        text = f'*{text.upper() if upper else text.lower()}*'
        out.append({'start': round(s, 3), 'end': round(e, 3), 'tag': True,
                    'words': [{'text': text, 's': round(s, 3), 'e': round(e, 3), 'emph': False}]})
    return out


def fix_stretched(words, max_len=1.0):
    """Whisper sometimes stretches a word over the silence next to it ("put" lasting 4 s), which
    leaves it alone on screen for seconds. Cap words at max_len: a word that runs straight into
    the next one was said just before it (keep its end), otherwise keep its start."""
    out = []
    for k, w in enumerate(words):
        w = dict(w)
        if w['e'] - w['s'] > max_len:
            nxt = words[k + 1] if k + 1 < len(words) else None
            if nxt and nxt['s'] - w['e'] < 0.1:
                w['s'] = round(w['e'] - max_len / 2, 3)
            else:
                w['e'] = round(w['s'] + max_len, 3)
        out.append(w)
    return out


def caption_groups(meta, upper=True, max_words=3, max_chars=15, max_dur=1.6):
    words = fix_stretched(merge_tokens(apply_fixes(meta['words'], meta.get('fix'))))
    splices = meta.get('splices', [])
    emph = {e.lower().strip('.,!?') for e in meta.get('emphasis', [])}
    # 1) phrases: runs of words between hard breaks (sentence end, pause, splice)
    phrases, cur = [], []
    for w in words:
        text = display(w['w'], upper)
        if not text:
            continue
        if cur:
            gap = w['s'] - cur[-1]['e']
            crosses = any(cur[-1]['e'] <= s <= w['s'] + 0.01 for s in splices)
            if gap > 0.6 or crosses:
                phrases.append(cur)
                cur = []
        cur.append({'text': text, 's': w['s'], 'e': w['e'], 'raw': w['w'],
                    'emph': w['w'].lower().strip('.,!?"') in emph})
        if w['w'][-1:] in '.?!':
            phrases.append(cur)
            cur = []
    if cur:
        phrases.append(cur)
    # 2) split each phrase into the fewest groups that fit, balanced, at natural break points
    groups = []
    for ph in phrases:
        groups += _split_phrase(ph, max_words, max_chars, max_dur)
    for g in groups:
        for c in g:
            c.pop('raw', None)
    out = []
    for g, nxt in zip(groups, groups[1:] + [None]):
        end = g[-1]['e'] + 0.25
        if nxt is not None:
            end = nxt[0]['s'] if nxt[0]['s'] - g[-1]['e'] < 0.6 else min(end, nxt[0]['s'])
        end = min(end, meta['duration'])
        out.append({'start': round(g[0]['s'], 3), 'end': round(end, 3), 'words': g})
    return out


# ---------- sound effects ----------

def sfx_cues(transitions, beats, duration, mode='auto', fps=FPS, meta=None):
    if mode == 'none':
        return []
    cues = _sfx_cues(transitions, beats, duration, mode, fps)
    if mode == 'light' or not meta:
        return cues
    memes = meme_cues(meta, duration, cues)
    cues = [c for c in cues if not any(abs(c['at'] - m['at']) < 0.8 for m in memes)] + memes
    # minimal accents: a quiet pop on the first time each of up to 2 emphasis words is spoken,
    # at least 5 s apart and away from the other cues
    emph = {e.lower().strip('.,!?') for e in meta.get('emphasis', [])}
    used = {(c['file'], 0) for c in cues}
    for w in meta.get('words', []):
        if sum(c['role'] == 'click' for c in cues) >= 2:
            break
        word = w['w'].lower().strip('.,!?"\'')
        if word not in emph or w['s'] < 1.0 or any(abs(c['at'] - w['s']) < (5 if c['role'] != 'meme' else 2.5)
                                                    for c in cues):
            continue
        emph.discard(word)
        it = sfxlib.pick('click', 0.3, used)
        if not it:
            break
        used.add((it['file'], it['variant']))
        at = max(0.0, w['s'] - it['peak'])
        cues.append({'file': it['file'], 'role': 'click', 'offset': it['start'],
                     'length': round(min(it['length'], duration - at), 3), 'at': round(at, 3),
                     'gain_db': -22, 'norm_db': it.get('norm_db', 0)})
    return cues


MEME_LEVEL_DB = -21     # loudest 300 ms of a meme sound, in dBFS before the final mix: ~9 dB under speech
MEME_MAX_LEN, MEME_MAX = 2.5, 3


def _norm(t):
    return ''.join(ch for ch in t.lower() if ch.isalnum() or ch == "'")


def find_phrase(words, phrase):
    """Index of the first word of `phrase` in the clip's words, or None."""
    target = [_norm(t) for t in phrase.split() if _norm(t)]
    toks = [_norm(w['w']) for w in words]
    for i in range(len(toks) - len(target) + 1):
        if toks[i:i + len(target)] == target:
            return i
    return None


def meme_cues(meta, duration, existing):
    """Meme reaction sounds: the ones Claude picked in clips.json ("sfx"), each on its word.
    With no picks, one vine boom lands on the last emphasis word (the punchline)."""
    lib = sfxlib.memes()
    if not lib:
        return []
    words = meta.get('words', [])
    picks = meta.get('sfx')
    if picks is None:
        emph = {e.lower().strip('.,!?') for e in meta.get('emphasis', [])}
        last = next((w for w in reversed(words) if w['w'].lower().strip('.,!?"\'') in emph), None)
        picks = [{'word': last['w'], 'sound': 'vine-boom', '_at': last['s']}] if last else []
    cues = []
    for p in picks[:MEME_MAX]:
        it = lib.get(p.get('sound', ''))
        if not it:
            print(f"   ! sfx: unknown sound {p.get('sound')!r} (have: {', '.join(sorted(lib))})")
            continue
        at = p.get('_at')
        if at is None:
            i = find_phrase(words, p.get('word', ''))
            if i is None:
                print(f"   ! sfx: word {p.get('word')!r} not found in clip {meta.get('id')}")
                continue
            at = words[i]['e'] if p.get('on') == 'end' else words[i]['s']
        # short hits land their peak on the word; longer stings (anime wow, romance) start on it
        start = max(0.0, at - (it['peak'] if it['peak'] <= 0.5 else 0.0))
        if any(abs(c['at'] - start) < 1.5 for c in cues):
            continue
        length = round(min(it['length'], MEME_MAX_LEN, duration - start), 3)
        if length < min(it['length'], MEME_MAX_LEN, 0.6):
            print(f"   ! sfx: {p['sound']} on {p.get('word')!r} is cut off by the clip end, skipped")
            continue
        cues.append({'file': it['file'], 'role': 'meme', 'name': p['sound'], 'offset': it['start'],
                     'length': length, 'at': round(start, 3), 'gain_db': MEME_LEVEL_DB,
                     'norm_db': -it['loud_db'], 'fade': 0.35 if it['length'] > MEME_MAX_LEN else 0.06})
    return cues


def _sfx_cues(transitions, beats, duration, mode, fps):
    cues, used = [], set()

    def add(role, peak_at, gain_db, prefer_len=None):
        it = sfxlib.pick(role, prefer_len, used)
        if not it:
            return
        used.add((it['file'], it['variant']))
        offset, length = it['start'], it['length']
        at = peak_at - it['peak']
        if at < 0:
            offset, length, at = offset - at, length + at, 0.0
        length = min(length, duration - at)
        if length <= 0.05:
            return
        cues.append({'file': it['file'], 'role': role, 'offset': round(offset, 3), 'length': round(length, 3),
                     'at': round(at, 3), 'gain_db': gain_db, 'norm_db': it.get('norm_db', 0)})

    add('whoosh', OPEN_FRAMES / fps * 0.5, -18, prefer_len=0.5)          # opening zoom
    if mode == 'light':
        return cues
    snap = next((t for t in transitions if t['kind'] == 'snap'), None)
    if snap:                                                             # whoosh on the emphasis zoom
        add('whoosh', snap['t'] + SNAP_FRAMES / fps * 0.4, -20, prefer_len=0.5)
    return cues


# ---------- the plan ----------

CAPTION_Y = 1270


def place_on_seam(caps, meta, cam, fps):
    """Streamer layout: put each caption on the seam between the facecam (top) and the gameplay
    (below), following the zoom so it stays on the seam. Other shots keep the default height."""
    shots = meta.get('shots', [])
    for c in caps:
        mid = (c['start'] + c['end']) / 2
        shot = next((sh for sh in shots if sh['t0'] <= mid < sh['t1']), None)
        if not shot or shot.get('mode') != 'split':
            continue
        seam = shot.get('top_h') or shot['focus'][1] * 2
        f0, f1 = int(c['start'] * fps), max(int(c['start'] * fps) + 1, int(c['end'] * fps))
        ys = [y + (seam - y) * s for s, _, y in cam[min(f0, len(cam) - 1):min(f1, len(cam))]] or [seam]
        c['y'] = int(round(sum(ys) / len(ys)))

def build(meta, brand, opts):
    fps = meta.get('fps', FPS)
    keys, cam, transitions, beats = build_camera(meta) if opts.get('zoom', True) else \
        ([], [[1.0, W / 2, H / 2]] * int(round(meta['duration'] * fps)), [], [(0, meta['duration'])])
    upper = opts.get('uppercase', True)
    caps = caption_groups(meta, upper=upper) if opts.get('captions', True) else []
    if caps:
        caps = sorted(caps + tag_groups(meta, upper), key=lambda c: c['start'])
    if opts.get('caption_y') is None:
        place_on_seam(caps, meta, cam, fps)
    # a tag that overlaps spoken captions sits one line above them instead of on top of them
    size = opts.get('size', 96)
    for c in caps:
        if c.get('tag') and any(not o.get('tag') and o['start'] < c['end'] and c['start'] < o['end'] for o in caps):
            c['y'] = int((c.get('y') or opts.get('caption_y') or CAPTION_Y) - size * 1.2)
    return {
        'fps': fps, 'width': W, 'height': H,
        'duration': meta['duration'], 'frames': len(cam),
        'title': meta.get('title'),
        'camera_keys': keys, 'cam': cam,
        'captions': caps,
        'caption_style': {
            'font': opts.get('font', 'Montserrat ExtraBold'), 'font_file': opts.get('font_file', 'Montserrat-ExtraBold.ttf'),
            'size': opts.get('size', 96), 'y': opts.get('caption_y') or CAPTION_Y, 'outline': 8, 'uppercase': opts.get('uppercase', True),
        },
        'brand': brand,
        'sfx': sfx_cues(transitions, beats, meta['duration'], opts.get('sfx', 'auto'), fps, meta),
        'transitions': transitions,
    }

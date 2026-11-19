"""Hybrid highlight scan for long streams: find the moments worth transcribing without running
Whisper on the whole thing.

Two cheap signals, scored per second:
  * chat  - message rate (weighted by hype words / emotes) against the stream's normal rate,
            shifted back a few seconds because chat reacts after the moment happens.
  * voice - loudness in the voice band (300-3000 Hz) against the stream's normal level, so the
            streamer shouting or laughing counts more than constant game audio.
The best peaks become candidate windows (with context before and after) that `fetch` then
transcribes. Everything here is plain numpy over 1-second bins: minutes, not hours, for a 15 h VOD.
"""
import json
import re
from pathlib import Path

import numpy as np

SR = 16000
CHAT_LAG = 8          # chat reacts ~5-15 s after the moment; shift the chat signal back by this
PRE, POST = 75, 45    # context kept before / after a peak (s); a short needs the setup, not just the scream
MERGE_GAP = 20        # windows closer than this are merged into one
BASE_WIN = 600        # rolling "normal level" window (s)

HYPE = re.compile(
    r"\b(lul|lol|lmao|lmfao|kekw|omegalul|icant|pog|poggers|pogchamp|w+|l+|clip|clip it|holy|wtf|omg|"
    r"no way|bro|gg|ez|monkas|pepega|xd|haha+|ha+ha|aware|sadge|wow|insane|crazy)\b"
    r"|[\U0001F602\U0001F923\U0001F480\U0001F525\U0001F62D\U0001F631‼❗!?]{2,}"
    r"|[\U0001F602\U0001F923\U0001F480\U0001F525\U0001F62D\U0001F631]",
    re.IGNORECASE)


# ---------------- chat ----------------

def _yt_message(action):
    item = (action.get('addChatItemAction') or {}).get('item') or {}
    r = item.get('liveChatTextMessageRenderer') or item.get('liveChatPaidMessageRenderer')
    if not r:
        return None
    parts = []
    for run in (r.get('message') or {}).get('runs', []):
        if 'text' in run:
            parts.append(run['text'])
        elif 'emoji' in run:
            e = run['emoji']
            parts.append(e.get('emojiId') if len(e.get('emojiId', '')) <= 2 else (e.get('shortcuts') or [''])[0])
    return ' '.join(parts)


def load_chat(path):
    """[(seconds, text)] from a yt-dlp YouTube live_chat.json (JSON lines), a TwitchDownloader chat
    JSON ({"comments": [...]}) or a plain list of {"t": seconds, "text": ...}."""
    path = Path(path)
    raw = path.read_text(encoding='utf-8', errors='replace').strip()
    out = []
    if raw.startswith('{') and '"comments"' in raw[:2000]:
        for c in json.loads(raw).get('comments', []):
            t = c.get('content_offset_seconds')
            if t is not None:
                out.append((float(t), ((c.get('message') or {}).get('body') or '')))
    elif raw.startswith('['):
        for c in json.loads(raw):
            out.append((float(c.get('t', c.get('time', 0))), str(c.get('text', ''))))
    else:   # YouTube: one JSON object per line
        for line in raw.splitlines():
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            rep = obj.get('replayChatItemAction') or {}
            off = rep.get('videoOffsetTimeMsec')
            if off is None:
                continue
            for a in rep.get('actions', []):
                msg = _yt_message(a)
                if msg is not None:
                    out.append((int(off) / 1000.0, msg))
    return [(t, m) for t, m in out if t >= 0]


def chat_signal(msgs, n):
    """Per-second chat excitement (robust z-score), already shifted back by CHAT_LAG."""
    rate = np.zeros(n)
    for t, m in msgs:
        i = int(t) - CHAT_LAG
        if 0 <= i < n:
            rate[i] += 1.0 + (0.75 if HYPE.search(m or '') else 0.0)
    act = np.convolve(rate, np.ones(10), mode='same')          # messages in a 10 s window
    base = _rolling_median(act, BASE_WIN)
    spread = max(1.0, float(np.median(np.abs(act - base))) * 1.4826)
    return np.clip((act - base) / (spread + np.sqrt(np.maximum(base, 0))), 0, 8), act, base


# ---------------- audio ----------------

def voice_energy(wav):
    """Voice-band (300-3000 Hz) energy in dB for every second of a 16 kHz mono wav, read in blocks."""
    import soundfile as sf
    out = []
    n = -(-sf.info(str(wav)).frames // SR)  # real length in seconds (the last block is zero-padded)
    lo, hi = 300, 3000                      # 1 Hz per bin with 16000-sample frames
    for block in sf.blocks(str(wav), blocksize=SR * 60, dtype='float32', fill_value=0.0):
        frames = block.reshape(-1, SR)
        spec = np.abs(np.fft.rfft(frames * np.hanning(SR), axis=1)) ** 2
        out.append(10 * np.log10(spec[:, lo:hi].sum(axis=1) + 1e-6))
    return np.concatenate(out)[:n] if out else np.zeros(0)


def voice_signal(db):
    """Per-second voice excitement: loud moments (3 s peak) above the stream's rolling normal level."""
    peak = _rolling_max(db, 3)
    base = _rolling_median(db, BASE_WIN)
    diff = peak - base
    spread = max(1.5, float(np.median(np.abs(diff - np.median(diff)))) * 1.4826)
    return np.clip((diff - 3.0) / spread, 0, 8), diff


# ---------------- windows ----------------

def pick_windows(score, budget_s, min_score=1.5):
    """Greedy: take the highest peak, keep PRE/POST context, suppress around it, repeat until the
    transcription budget is spent or nothing exciting is left. Overlapping windows are merged.
    The bar adapts to streams that are loud all the time: never above the top 10% of the stream."""
    n = len(score)
    s = score.copy()
    if n:
        min_score = min(min_score, max(0.3, float(np.percentile(score, 90))))
    picks, used = [], 0.0
    while used < budget_s:
        i = int(np.argmax(s))
        if s[i] < min_score:
            break
        a, b = max(0, i - PRE), min(n, i + POST)
        picks.append((a, b, i))
        used += b - a
        s[max(0, a - PRE // 2): min(n, b + POST // 2)] = 0
    picks.sort()
    merged = []
    for a, b, i in picks:
        if merged and a - merged[-1]['b'] <= MERGE_GAP:
            merged[-1]['b'] = max(merged[-1]['b'], b)
            merged[-1]['peaks'].append(i)
        else:
            merged.append({'a': a, 'b': b, 'peaks': [i]})
    return merged


def scan(wav, duration, chat_path=None, budget_min=None):
    """Returns {'windows': [...], 'stats': {...}}; each window has a, b (s), peaks and a reason."""
    print('-> scanning voice loudness ...', flush=True)
    db = voice_energy(wav)
    n = len(db)
    v, vdiff = voice_signal(db)

    msgs = load_chat(chat_path) if chat_path and Path(chat_path).exists() else []
    if msgs:
        print(f'-> scanning chat: {len(msgs):,} messages', flush=True)
        c, act, cbase = chat_signal(msgs, n)
        score = 0.6 * c + 0.4 * v       # chat is the stronger signal for streams
    else:
        print('-> no chat replay: using voice loudness only (quiet highlights can be missed)', flush=True)
        c = act = cbase = None
        score = v.copy()
    score = np.convolve(score, np.ones(5) / 5, mode='same')     # ignore 1-second blips

    if budget_min is None:      # ~8% of the stream, between 20 and 90 minutes
        budget_min = min(90, max(20, (duration or n) * 0.08 / 60))
    wins = pick_windows(score, budget_min * 60)
    for w in wins:
        a, b = w['a'], w['b']
        bits = []
        if c is not None:
            k = int(np.argmax(act[a:b])) + a
            ratio = act[k] / max(1.0, cbase[k])
            if ratio >= 2.5:
                bits.append(f'chat x{ratio:.1f}')
        dv = float(vdiff[a:b].max())
        if dv >= 3:
            bits.append(f'voice +{dv:.0f} dB')
        w['score'] = round(float(score[a:b].max()), 2)
        w['reason'] = ', '.join(bits) or 'activity'
    total = sum(w['b'] - w['a'] for w in wins)
    return {'windows': wins, 'stats': {'seconds_scanned': n, 'seconds_selected': total,
                                       'chat_messages': len(msgs), 'budget_min': round(budget_min, 1)}}


# ---------------- helpers ----------------

def _rolling_median(x, win):
    """Rolling median, computed on 30 s block medians so it stays fast for 50k+ seconds."""
    if len(x) == 0:
        return x
    blk = 30
    nb = (len(x) + blk - 1) // blk
    pad = np.pad(x, (0, nb * blk - len(x)), mode='edge').reshape(nb, blk)
    meds = np.median(pad, axis=1)
    k = max(1, win // blk)
    roll = np.array([np.median(meds[max(0, j - k // 2): j + k // 2 + 1]) for j in range(nb)])
    return np.repeat(roll, blk)[:len(x)]


def _rolling_max(x, win):
    if len(x) == 0:
        return x
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(x, (win // 2, win - 1 - win // 2), mode='edge')
    return sliding_window_view(pad, win).max(axis=1)

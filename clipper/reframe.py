"""Shot-aware 16:9 -> 9:16 reframing: find camera cuts, find the speaker's face in each shot,
and build an ffmpeg filter that crops each shot around the face (static crop per shot, like a
human editor would, with no jittery tracking)."""
import re
import subprocess

import cv2
import numpy as np

from .common import ASSETS, FPS, H, W

# YuNet face detector (OpenCV Zoo, MIT licence): small, fast, and handles turned heads and small faces
YUNET = ASSETS / 'models' / 'face_detection_yunet_2023mar.onnx'
_DET = {}
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass


def _detector(w, h):
    if 'yunet' not in _DET:
        _DET['yunet'] = cv2.FaceDetectorYN.create(str(YUNET), '', (w, h), 0.6, 0.3, 5000)
    d = _DET['yunet']
    d.setInputSize((w, h))
    return d


def scene_cuts(video, start, dur, threshold=0.30, min_shot=0.8):
    """Return shot boundaries [(a, b), ...] in seconds relative to `start`."""
    cmd = ['ffmpeg', '-hide_banner', '-ss', f'{start:.3f}', '-t', f'{dur:.3f}', '-i', str(video),
           '-vf', f"scale=320:-2,select='gt(scene,{threshold})',showinfo", '-an', '-f', 'null', '-']
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    times = [float(m) for m in re.findall(r'pts_time:([0-9.]+)', r.stderr)]
    bounds = [0.0]
    for t in times:
        if t - bounds[-1] >= min_shot and dur - t >= min_shot:
            bounds.append(t)
    bounds.append(dur)
    return list(zip(bounds[:-1], bounds[1:]))


def _faces(img):
    """All face boxes (x, y, w, h) in a BGR image."""
    h, w = img.shape[:2]
    _, faces = _detector(w, h).detect(img)
    if faces is None:
        return []
    return [tuple(float(v) for v in f[:4]) for f in faces if f[2] >= 18]


def face_in_shot(cap, t0, t1, samples=5):
    """Median face centre (cx, cy as 0..1) and size (0..1 of frame height), or None."""
    pts = []
    for k in range(samples):
        t = t0 + (t1 - t0) * (k + 0.5) / samples
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        h0, w0 = frame.shape[:2]
        scale = 640 / w0
        small = cv2.resize(frame, (640, int(h0 * scale)))
        faces = _faces(small)
        if faces:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            sh = small.shape[0]
            pts.append(((x + w / 2) / 640, (y + h / 2) / sh, h / sh))
    if len(pts) < max(2, samples // 3):
        return None
    a = np.array(pts)
    return tuple(float(v) for v in np.median(a, axis=0))


def analyse_range(video, start, dur, src_w, src_h):
    """Shots for one source range: [{'a','b','face': (cx,cy,size)|None}] (times relative to start)."""
    shots = scene_cuts(video, start, dur)
    cap = cv2.VideoCapture(str(video))
    out = []
    for a, b in shots:
        out.append({'a': a, 'b': b, 'face': face_in_shot(cap, start + a, start + b)})
    cap.release()
    return out


# ---------------------------------------------------------------------------------------------
# Streamer mode: find the facecam overlay in every shot, so the streamer always sits in the top
# panel even when the overlay moves (menu vs gameplay scenes) or the stream cuts to full-cam.

def _border(score, lo, hi, thresh):
    """Strongest edge index in score[lo:hi] above thresh, or None."""
    lo, hi = max(lo, 0), min(hi, len(score))
    if hi - lo < 2:
        return None
    i = int(np.argmax(score[lo:hi])) + lo
    return i if score[i] >= thresh else None


def _cam_box(gray, face, thresh=10.0):
    """The overlay rectangle (x, y, w, h) around a small face, found from the straight edges of the
    overlay's border (the frame edge counts as a border). The top-left corner must show clear
    edges; the bottom/right edges often blend into the game, so when they are faint the box is
    completed as 16:9. Returns None if no box fits."""
    Hs, Ws = gray.shape
    fx, fy, fw, fh = face
    gy = np.abs(np.diff(gray, axis=0))           # horizontal edges, row i = between i and i+1
    gx = np.abs(np.diff(gray, axis=1))           # vertical edges
    x_lo, x_hi = int(max(0, fx - 1.5 * fw)), int(min(Ws, fx + 2.5 * fw))
    rows = np.percentile(gy[:, x_lo:x_hi], 25, axis=1)
    top = _border(rows, int(fy - 4 * fh), int(fy - 0.2 * fh), thresh)
    top = 0 if top is None and fy - 4 * fh <= 0 else (None if top is None else top + 1)
    if top is None:
        return None
    cols = np.percentile(gx[top:int(min(Hs, fy + 2 * fh)), :], 25, axis=0)
    left = _border(cols, int(fx - 6 * fw), int(fx - 0.2 * fw), thresh)
    left = 0 if left is None and fx - 6 * fw <= 0 else (None if left is None else left + 1)
    if left is None:
        return None
    # pick the width whose 16:9 box has the strongest right + bottom edges (frame edges count)
    rows_ = np.append(rows, 255.0)
    cols_ = np.append(cols, 255.0)
    cands = []
    for w in range(int(fx + 1.2 * fw - left), int(min(Ws - left, 0.55 * Ws)) + 1):
        h = int(round(w * 9 / 16))
        if top + h > Hs or top + h < fy + 1.1 * fh:
            continue
        r, b = left + w - 1, top + h - 1
        cands.append((cols_[max(r - 1, 0):r + 2].max() + rows_[max(b - 1, 0):b + 2].max(), w, h))
    if not cands:
        return None
    top_score = max(c[0] for c in cands)
    if top_score < thresh:
        return None
    # the overlay's own border is the nearest clear edge; stronger edges further out are gameplay
    _, w, h = next(c for c in cands if c[0] >= max(thresh, 0.6 * top_score))
    if w < 0.10 * Ws:
        return None
    return (left, top, w, h)


def streamer_cam(cap, t0, t1, samples=7, max_face=0.16):
    """Facecam overlay box in source px for this shot, or None. A small face counts as the
    streamer only if it sits inside a bordered 16:9-ish box that stays put across the shot
    (game characters move, the overlay doesn't)."""
    boxes = []
    for k in range(samples):
        cap.set(cv2.CAP_PROP_POS_MSEC, (t0 + (t1 - t0) * (k + 0.5) / samples) * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        h0, w0 = frame.shape[:2]
        scale = 960 / w0
        small = cv2.resize(frame, (960, int(h0 * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        for f in _faces(small):
            if f[3] / small.shape[0] > max_face:
                continue
            b = _cam_box(gray, f)
            if b:
                boxes.append(tuple(v / scale for v in b))
    if not boxes:
        return None
    # cluster by centre; the streamer box is the one seen in the most samples
    a = np.array(boxes)
    centres = a[:, :2] + a[:, 2:] / 2
    best = max(range(len(a)), key=lambda i: int((np.linalg.norm(centres - centres[i], axis=1) < 40).sum()))
    near = np.linalg.norm(centres - centres[best], axis=1) < 40
    if near.sum() < max(2, samples // 3):
        return None
    x, y, w, h = np.median(a[near], axis=0)
    return [int(v) // 2 * 2 for v in (x, y, w, h)]


def screen_box(src_w, src_h, top_h):
    """Centre crop of the source that fills the bottom panel exactly."""
    aspect = W / (H - top_h)
    w = min(src_w, int(src_h * aspect))
    h = min(src_h, int(w / aspect))
    return [(src_w - w) // 4 * 2, (src_h - h) // 4 * 2, w // 2 * 2, h // 2 * 2]


def analyse_streamer(video, start, dur, src_w, src_h, top_h, chunk=4.0):
    """Shots for a stream VOD: each shot is 'split' (facecam found: cam on top, gameplay below),
    else a normal face crop (full-cam shot), else 'fit'."""
    shots = scene_cuts(video, start, dur)
    cap = cv2.VideoCapture(str(video))
    out = []
    for a, b in shots:
        # check the cam every few seconds, so a facecam that moves without a scene cut is followed
        n = max(1, int(round((b - a) / chunk)))
        edges = [a + (b - a) * k / n for k in range(n + 1)]
        cams = [streamer_cam(cap, start + edges[k], start + edges[k + 1], samples=5) for k in range(n)]
        if not any(cams):
            out.append({'a': a, 'b': b, 'cam': None, 'face': face_in_shot(cap, start + a, start + b)})
            continue
        for k in range(n):          # a missed chunk inside a cam shot borrows the nearest found box
            if not cams[k]:
                cams[k] = next(cams[j] for j in sorted(range(n), key=lambda j: abs(j - k)) if cams[j])
        for k in range(n):          # merge neighbouring chunks that show the cam in the same place
            prev = out[-1] if out and out[-1]['cam'] and out[-1]['b'] == edges[k] else None
            if prev and max(abs(p - q) for p, q in zip(prev['cam'], cams[k])) < 40:
                prev['b'] = edges[k + 1]
            else:
                out.append({'a': edges[k], 'b': edges[k + 1], 'cam': cams[k], 'face': None})
    cap.release()
    # within one clip the overlay rarely changes size: a box that shares its corner with a box
    # seen for longer is the same overlay with a misread edge, so use the longer-seen box
    seen = {}
    for s in out:
        if s['cam']:
            key = tuple(s['cam'])
            seen[key] = seen.get(key, 0) + (s['b'] - s['a'])
    for s in out:
        if s['cam']:
            same_corner = [k for k in seen if abs(k[0] - s['cam'][0]) < 40 and abs(k[1] - s['cam'][1]) < 40]
            s['cam'] = list(max(same_corner, key=lambda k: seen[k]))
    merged = []
    for s in out:              # re-merge neighbours that now show the same box
        if merged and merged[-1]['cam'] and merged[-1]['cam'] == s['cam'] and merged[-1]['b'] == s['a']:
            merged[-1]['b'] = s['b']
        else:
            merged.append(s)
    out = merged
    # short shots with no cam (a missed detection mid-scene) reuse the neighbour's cam
    for i, s in enumerate(out):
        if not s['cam'] and s['b'] - s['a'] < 2.0:
            nb = [out[j]['cam'] for j in (i - 1, i + 1) if 0 <= j < len(out) and out[j]['cam']]
            if nb:
                s['cam'] = nb[0]
    plan = []
    for s in out:
        if s['cam']:
            plan.append({**s, 'mode': 'split', 'screen': screen_box(src_w, src_h, top_h),
                         'top_h': top_h, 'focus': [W // 2, top_h // 2]})
        else:
            plan += plan_shots([s], src_w, src_h)
    return plan


def crop_width(src_w, src_h):
    return min(src_w, int(src_h * 9 / 16) // 2 * 2)


def crop_x(cx, src_w, src_h):
    cw = crop_width(src_w, src_h)
    return int(min(max(cx * src_w - cw / 2, 0), src_w - cw)) // 2 * 2


def plan_shots(shots, src_w, src_h, min_fit=1.0):
    """Decide a crop (or 'fit' layout) for each shot. A shot with no face (quote card, slide, screen
    recording, wide b-roll) is shown whole over a blurred background, so no text gets cut off.
    Very short faceless shots (< min_fit s, usually a missed detection) reuse the nearest face."""
    is_vertical = src_w / src_h < 0.75
    known = [s['face'] for s in shots if s['face']]
    plan, last = [], (known[0] if known else None)
    for s in shots:
        face = s['face'] or (last if s['b'] - s['a'] < min_fit else None)
        if s['face']:
            last = s['face']
        if is_vertical:
            mode, x0, fx, fy = 'vertical', 0, (face[0] if face else 0.5) * W, (face[1] if face else 0.4) * H
        elif face is None:
            mode, x0, fx, fy = 'fit', 0, W / 2, H * 0.42
        else:
            cw = crop_width(src_w, src_h)
            x0 = crop_x(face[0], src_w, src_h)
            mode = 'crop'
            fx = (face[0] * src_w - x0) / cw * W
            fy = face[1] * H
        plan.append({**s, 'mode': mode, 'x0': x0, 'focus': [round(fx), round(fy)]})
    return plan


def shot_filter(inp, shot, src_w, src_h, label, fps=FPS):
    """ffmpeg filter chain turning one shot of input `inp` into 1080x1920 [label]."""
    trim = f"trim=start={shot['a']:.3f}:end={shot['b']:.3f},setpts=PTS-STARTPTS"
    tail = f'setsar=1,fps={fps},format=yuv420p'
    if shot['mode'] == 'vertical':
        return (f"[{inp}]{trim},scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={W}:{H},{tail}[{label}]")
    if shot['mode'] == 'fit':
        return (f"[{inp}]{trim},split[{label}b][{label}f];"
                f"[{label}b]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
                f"boxblur=30:3,eq=brightness=-0.08[{label}bb];"
                f"[{label}f]scale={W}:-2:flags=lanczos[{label}ff];"
                f"[{label}bb][{label}ff]overlay=0:(H-h)/2*0.85,{tail}[{label}]")
    if shot['mode'] == 'split':
        # streamer layout: facecam box on top, screen box below, stacked into one 1080x1920 frame
        cx, cy, cw_, ch_ = shot['cam']
        sx, sy, sw_, sh_ = shot['screen']
        top = shot['top_h']
        return (f"[{inp}]{trim},split[{label}c][{label}s];"
                f"[{label}c]crop={cw_}:{ch_}:{cx}:{cy},scale={W}:{top}:force_original_aspect_ratio=increase:"
                f"flags=lanczos,crop={W}:{top}[{label}ct];"
                f"[{label}s]crop={sw_}:{sh_}:{sx}:{sy},scale={W}:{H - top}:force_original_aspect_ratio=increase:"
                f"flags=lanczos,crop={W}:{H - top}[{label}st];"
                f"[{label}ct][{label}st]vstack,{tail}[{label}]")
    cw = crop_width(src_w, src_h)
    return (f"[{inp}]{trim},crop={cw}:{src_h}:{shot['x0']}:0,scale={W}:{H}:flags=lanczos,"
            f"{tail}[{label}]")

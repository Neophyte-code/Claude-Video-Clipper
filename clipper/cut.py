"""Step 3: cut the clips Claude picked (clips.json) into vertical 1080x1920 shorts.

    python run.py cut <project>               # all clips
    python run.py cut <project> --clips 1,3   # only some
    python run.py cut <project> --horizontal  # keep 16:9 (no reframe)

Downloads the video the first time (max 1080p). Snaps every range to whole words so nothing is
cut mid-word, reframes each camera shot around the speaker's face, and writes
clips/NN-title.mp4 plus clips/NN-title.json (timing, words, focus point) used by `edit`.

Streams are handled automatically: wherever a facecam overlay is on screen (found every few
seconds, so it is followed when it moves), the facecam goes on top and the gameplay below.
Full-cam shots and normal videos get the usual face crop. Turn it off with --no-streamer.
"""
import argparse
import sys
from pathlib import Path

from .common import die, out_fps, ffmpeg, fmt_ts, load_json, probe, project_dir, save_json, slugify
from .reframe import analyse_range, analyse_streamer, plan_shots, shot_filter

PAD_IN, PAD_OUT = 0.12, 0.30


def ensure_video(pdir, project):
    if project.get('video') and Path(project['video']).exists():
        return Path(project['video'])
    existing = [p for p in (pdir / 'source').glob('video.*') if p.suffix in ('.mp4', '.mkv', '.webm')]
    if existing:
        return existing[0]
    from .fetch import ytdlp
    print('-> downloading video (max 1080p, first time only) ...')
    fmt = ('bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b')
    ytdlp(['-f', fmt, '--merge-output-format', 'mp4', '-o', str(pdir / 'source' / 'video.%(ext)s'),
           project['source']], project.get('cookies_from_browser'))
    existing = list((pdir / 'source').glob('video.*'))
    if not existing:
        die('video download produced no file')
    return existing[0]


def snap_range(s, e, words):
    """Expand/shrink [s, e] to whole words and add small natural handles."""
    inside = [i for i, w in enumerate(words) if w['e'] > s + 0.05 and w['s'] < e - 0.05]
    if not inside:
        return None
    i0, i1 = inside[0], inside[-1]
    start = words[i0]['s'] - PAD_IN
    if i0 > 0:
        start = max(start, words[i0 - 1]['e'] + 0.02)
    end = words[i1]['e'] + PAD_OUT
    if i1 + 1 < len(words):
        end = min(end, max(words[i1]['e'] + 0.08, words[i1 + 1]['s'] - 0.04))
    return max(0.0, start), end, i0, i1


def boundary_warnings(ranges, words):
    """Flag clips that start or end mid-sentence (the most common clipping mistake)."""
    warns = []
    for s, e in ranges:
        snapped = snap_range(s, e, words)
        if not snapped:
            continue
        _, _, i0, i1 = snapped
        gap = max((words[i + 1]['s'] - words[i]['e'] for i in range(i0, i1)), default=0)
        if gap > 30:
            warns.append(f'has {gap:.0f} s with no transcribed speech inside it (with a hybrid-scan '
                         f'transcript this means it spans two highlight windows: keep it inside one)')
        prev, last = (words[i0 - 1] if i0 > 0 else None), words[i1]
        nxt = words[i1 + 1] if i1 + 1 < len(words) else None
        if prev and prev['w'][-1:] not in '.?!' and words[i0]['s'] - prev['e'] < 0.6:
            warns.append(f'starts mid-sentence, right after "...{" ".join(w["w"] for w in words[max(0, i0 - 5):i0])}"')
        if nxt and last['w'][-1:] not in '.?!' and nxt['s'] - last['e'] < 0.6:
            warns.append(f'ends mid-sentence: "...{" ".join(w["w"] for w in words[max(0, i1 - 5):i1 + 1])}" '
                         f'is followed by "{" ".join(w["w"] for w in words[i1 + 1:i1 + 7])}..."')
    return warns


def build_clip(video, info, clip, words, out_mp4, horizontal=False, split=None, streamer_top=None):
    src_w, src_h = info['width'], info['height']
    fps = out_fps(info.get('fps', 30))
    parts, inputs, chains, labels = [], [], [], []
    timeline = 0.0
    clip_words, shots_out, splices = [], [], []
    for k, (s, e) in enumerate(clip['ranges']):
        snapped = snap_range(s, e, words)
        if not snapped:
            print(f'   ! range {s}-{e} has no words, skipped')
            continue
        start, end, i0, i1 = snapped
        dur = end - start
        inputs += ['-ss', f'{start:.3f}', '-t', f'{dur:.3f}', '-i', str(video)]
        n = len(parts)
        if horizontal:
            chains.append(f'[{n}:v]fps={fps},setsar=1,format=yuv420p[v{n}]')
            shots = [{'a': 0, 'b': dur, 'mode': 'horizontal', 'focus': [src_w / 2, src_h * 0.4]}]
        elif split:
            shots = [{'a': 0, 'b': dur, 'mode': 'split', **split, 'focus': [540, split['top_h'] // 2]}]
            chains.append(shot_filter(f'{n}:v', shots[0], src_w, src_h, f'v{n}', fps))
        else:
            if streamer_top and src_w / src_h >= 0.75:
                shots = analyse_streamer(video, start, dur, src_w, src_h, streamer_top)
            else:
                shots = plan_shots(analyse_range(video, start, dur, src_w, src_h), src_w, src_h)
            if len(shots) == 1:
                chains.append(shot_filter(f'{n}:v', shots[0], src_w, src_h, f'v{n}', fps))
            else:
                sub = ''.join(f'[{n}s{j}]' for j in range(len(shots)))
                chains.append(f'[{n}:v]split={len(shots)}{sub}')
                for j, sh in enumerate(shots):
                    chains.append(shot_filter(f'{n}s{j}', sh, src_w, src_h, f'v{n}_{j}', fps))
                chains.append(''.join(f'[v{n}_{j}]' for j in range(len(shots))) +
                              f'concat=n={len(shots)}:v=1:a=0[v{n}]')
        fade = 0.12 if k == len(clip['ranges']) - 1 else 0.03   # softer tail on the clip's last word
        chains.append(f'[{n}:a]aresample=48000,afade=t=in:d=0.01,'
                      f'afade=t=out:st={max(0, dur - fade):.3f}:d={fade}[a{n}]')
        labels.append(f'[v{n}][a{n}]')
        for w in words[i0:i1 + 1]:
            clip_words.append({**w, 's': round(w['s'] - start + timeline, 3),
                               'e': round(w['e'] - start + timeline, 3)})
        for sh in shots:
            shots_out.append({'t0': round(sh['a'] + timeline, 3), 't1': round(sh['b'] + timeline, 3),
                              'mode': sh['mode'], 'focus': sh['focus'],
                              **({'cam': sh['cam']} if sh.get('cam') else {})})
        if timeline > 0:
            splices.append(round(timeline, 3))
        parts.append((start, end))
        timeline += dur
    if not parts:
        return None
    n = len(parts)
    chains.append(''.join(labels) + f'concat=n={n}:v=1:a=1[outv][outa]')
    tmp = out_mp4.with_suffix('.part.mp4')
    ffmpeg(*inputs, '-filter_complex', ';'.join(chains), '-map', '[outv]', '-map', '[outa]',
           '-c:v', 'libx264', '-preset', 'fast', '-crf', '17', '-pix_fmt', 'yuv420p', '-r', str(fps),
           '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-movflags', '+faststart', tmp)
    tmp.replace(out_mp4)
    return {'fps': fps, 'parts': [[round(a, 3), round(b, 3)] for a, b in parts], 'duration': round(timeline, 3),
            'shots': shots_out, 'splices': splices, 'words': clip_words}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('project')
    ap.add_argument('--clips', default=None, help='comma list of clip ids, e.g. 1,3')
    ap.add_argument('--horizontal', action='store_true', help='keep the original 16:9 framing')
    ap.add_argument('--check', action='store_true', help='only check clip boundaries, cut nothing')
    ap.add_argument('--split', action='store_true',
                    help='streamer layout: facecam box on top, screen box below (set --cam/--screen)')
    ap.add_argument('--no-streamer', dest='streamer', action='store_false',
                    help='turn off automatic facecam detection (on by default: wherever a facecam '
                         'overlay is found it goes on top, with the gameplay below)')
    ap.add_argument('--streamer', action='store_true', default=True, help=argparse.SUPPRESS)
    ap.add_argument('--cam', default='0,330,864,672', help='facecam box in source px: x,y,w,h')
    ap.add_argument('--screen', default='740,70,900,900', help='screen box in source px: x,y,w,h')
    ap.add_argument('--top-h', type=int, default=840, help='height of the facecam panel (of 1920)')
    a = ap.parse_args()
    box = lambda s: [int(v) // 2 * 2 for v in s.split(',')]
    split = {'cam': box(a.cam), 'screen': box(a.screen), 'top_h': a.top_h // 2 * 2} if a.split else None

    pdir = project_dir(a.project)
    project = load_json(pdir / 'project.json')
    picks = load_json(pdir / 'clips.json')
    if not picks or not picks.get('clips'):
        die(f'no clips.json in {pdir}. Claude writes it after reading transcript/transcript.txt')
    words = load_json(pdir / 'transcript' / 'words.json')
    want = {int(x) for x in a.clips.split(',')} if a.clips else None

    flagged = 0
    for i, clip in enumerate(picks['clips'], 1):
        cid = int(clip.get('id', i))
        if want and cid not in want:
            continue
        snapped = [snap_range(s, e, words) for s, e in clip['ranges']]
        total = sum(x[1] - x[0] for x in snapped if x)
        msgs = boundary_warnings(clip['ranges'], words)
        if total and not 15 <= total <= 90:
            msgs.append(f'length {total:.0f}s is outside 15-90 s')
        for m in msgs:
            print(f'   WARNING clip {cid} "{clip.get("title")}": {m}')
        flagged += bool(msgs)
    if a.check:
        print('check: all clip boundaries look clean' if not flagged else
              f'check: {flagged} clip(s) need a look. Fix the ranges with `run.py find`, '
              'unless the cut is intentional')
        return

    video = ensure_video(pdir, project)
    info = probe(video)
    print(f'-> source: {info["width"]}x{info["height"]} {info["fps"]:.2f}fps, {fmt_ts(info["duration"])}')
    out_dir = pdir / 'clips'
    out_dir.mkdir(exist_ok=True)

    done = []
    for i, clip in enumerate(picks['clips'], 1):
        cid = int(clip.get('id', i))
        if want and cid not in want:
            continue
        name = f'{cid:02d}-{slugify(clip.get("title", "clip"), 40)}'
        out = out_dir / f'{name}.mp4'
        print(f'-> clip {cid}: "{clip.get("title")}"  ranges={clip["ranges"]}')
        meta = build_clip(video, info, clip, words, out, a.horizontal, split,
                          a.top_h // 2 * 2 if a.streamer and not split else None)
        if not meta:
            print('   ! skipped (no words in its ranges)')
            continue
        meta.update({'id': cid, 'name': name, 'title': clip.get('title'), 'hook': clip.get('hook'),
                     'emphasis': clip.get('emphasis', []), 'file': out.name})
        save_json(out_dir / f'{name}.json', meta)
        modes = sorted({s['mode'] for s in meta['shots']})
        print(f'   ok {out.name}  {meta["duration"]:.1f}s, {len(meta["shots"])} shot(s), {"/".join(modes)}')
        done.append(out)

    print(f'\nDONE: {len(done)} clip(s) in {out_dir}')
    print('next (optional simple edit):  python run.py edit ' + pdir.name)


if __name__ == '__main__':
    main()

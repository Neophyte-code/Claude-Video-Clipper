# Sound effects

The edit keeps sound minimal and low under the voice: one soft whoosh on the opening, plus a
quiet pop on up to 2 emphasis words per clip (at least 5 s apart).

| File | Role | Used for |
|---|---|---|
| `mi-whoosh-clean.mp3`, `mi-whoosh-simple.mp3` | whoosh | opening (and emphasis zooms) |
| `mi-pop-finger.mp3`, `mi-pop-sfx.mp3` | click | pops on emphasis words |
| `mi-ding.mp3` | (none) | spare, not used by the edit |

The `mi-` sounds come from myinstants.com (user uploads, licence unclear). The original
sounds are kept in `bundled/` (not scanned); move them back here to use them again.

## Meme reaction sounds (`meme/`)

`meme/meme-<name>.mp3` are short reaction sounds: `vine-boom`, `fart`, `fart-button`,
`anime-wow`, `rizz`, `among-us`, `capcut-explosion`, `bone-crack`, `romance`, `punch`. Claude picks
up to 3 per clip in `clips.json` (`"sfx": [{"word": "bad guy", "sound": "vine-boom"}]`), each landing
on its word. Every sound is levelled by its loudest moment to about 9 dB under the speech, so they
never get louder than the voice. Long stings are capped at 2.5 s with a fade. A clip with no `sfx`
list gets one vine boom on its last emphasis word. Add your own: drop `meme-<name>.mp3` in `meme/`.
These also come from myinstants.com (user uploads, licence unclear).

**Use your own:** drop any `.mp3`/`.wav` in this folder with `whoosh`, `hit` or `click`/`pop`
in its name. `catalog.json` is rebuilt automatically when the folder changes.
Make sure you have the rights to any sound you add or publish.

# Song routines: validation results and library cleanup (2026-10-17)

Handoff for whoever owns the Supabase song library (`waldo-mirror/library.py`, uncommitted on Waldo).
Everything here was run on Waldo; nothing moved the robot.

## Where things are on Waldo

| What | Path |
|------|------|
| Full routines (also in Supabase) | `~/.waldo/choreo/<id>.npz` + `<id>.audio.*` |
| 90-second segments (not in Supabase) | `/tmp/claude-choreo/cuts/<id>_90s.npz` (same keys as a routine, plus `cut_start_s`; `env` is the full song's) |
| Raw check results, one file per routine | `/tmp/claude-choreo/res/<id>.jsonl` (`analyze`, `detect`, `sync`) |
| Checker | `/tmp/claude-choreo/check.py` (`analyze|detect|sync <id>`), `/tmp/claude-choreo/pairs.py <id>` (contacts per link pair) |

`/tmp` can be wiped on reboot: copy the cuts somewhere permanent if they're needed.

Run the checker from `~/HACKATHON2026/waldo-mirror` with `.venv/bin/python /tmp/claude-choreo/check.py analyze <id>`.

## Results (90-second segment of each routine)

All 8 routines: 0 comfort-range violations, max joint speed ≤ 1.0 rad/s, no frame with a hand across the midline,
torso/legs static (routines are 14 arm joints; playback holds the waist at 0). Contacts are mesh distance ≤ 0 on
the G1 URDF (`g1_body29_hand14`, finger links excluded), checked at 7.5 Hz. Torso vs shoulder-yaw contacts are
listed separately: that pair is only 2.9 cm apart in the robot's own rest pose, so they need a gantry check, not
a rejection.

| Rank | Song | Video id | Coach / both arms detected | Contacts (excl. torso–shoulder) | Torso–shoulder | Sync error, 3 clips (s) | Status |
|------|------|----------|---------------------------|---------------------------------|----------------|-------------------------|--------|
| 1 | Happy | `G74_o_43_RQ` | 91% / 82% | 0 | 5 | −0.03, **+185 (strength 0.077)**, −0.03 | Pass |
| 2 | Blinding Lights | `hnVwwzCVoqg` | 88% / 72% | 0 (right hand < 3 cm from hip, 30 frames) | 8 | −0.02, −0.02, −0.02 | Pass |
| 3 | September | `rsH3eSdpWKU` | 98% / 84% | 0 (hands come within 11.5 cm) | 18 | −0.01, −0.03, −0.97 (0.028) | Pass |
| 4 | Dynamite | `CM7ZSzCkx0Y` | 86% / 64% | 0 | 12 | −0.01, −0.03, +50.5 (0.024) | Pass, backup |
| 5 | Vogue | `mEVEC-slpJI` | 86% / 74% | 0 (head near right shoulder, 32 frames) | 20 | −0.02, −0.02, −0.02 | Pass, backup |
| – | I Gotta Feeling | `_8f3hZ23ZEs` | **62% / 56%** (crowd behind coach) | hand–hip ×4 | 27 | −0.02 ×3 | Reject |
| – | Can't Stop the Feeling | `ELJjp-lDfFg` | 82% / 62% | **arm–head ×15, hand–hip ×17** | 40 | +253 (0.051), −0.01, +4.2 | Reject |
| – | Y.M.C.A. | `YCDCwuGcEmA` | 92% / 68% | **hand–hand ×16** | 38 | −0.03, −0.01, −0.01 | Reject |
| – | Stayin' Alive | `_Gb8em52CYk` | | | | | Reject: Dance Central, not Just Dance |
| – | Can't Stop the Feeling | `bUSNcBaag88` | | | | | Reject: ALT version |
| – | We Will Rock You | none | | | | | No official Just Dance routine |
| ? | Levitating | `Xi5tDl4lDGY` | | | | | Not checked; library shows "tracked 0%" |

Sync clips are cut from the routine's own audio (ends at +30/+60/+90 s into the segment), so they test
`choreo.align()`, not real call audio: real-world sync is **inconclusive** for every song. Note that Happy's
+185 s miss (0.077) and Can't Stop the Feeling's +253 s miss (0.051) are above the worker's `min_strength` 0.05,
so a live call could jump to the wrong place. A prerecorded demo should start music and routine together.

Nothing is validated on hardware: run each pick once on the gantry at reduced speed with the operator before recording.

## Library cleanup (owner's call)

On Waldo, in `~/HACKATHON2026/waldo-mirror` (Supabase keys are in `~/.waldo-ask/env`):

```sh
./library.sh list
# remove rejected routines (also deletes the local cache copy)
.venv/bin/python -c "from library import remove
for i in ['_Gb8em52CYk','bUSNcBaag88','ELJjp-lDfFg','YCDCwuGcEmA','_8f3hZ23ZEs']: remove(i)"
# optional: publish a 90 s segment, one file at a time
./choreo.sh publish /tmp/claude-choreo/cuts/G74_o_43_RQ_90s.npz --query "happy pharrell williams"
```

- Never run `./choreo.sh publish` with no files: it publishes every cached routine, rejected ones included.
- Don't keep a full routine and its 90 s cut in the library together: they share the same audio envelope, so
  song recognition (`MIN_LEAD`) can't tell them apart. Replace the full one, or keep cuts out of Supabase.
- `prepare` in the current uncommitted `choreo.py` publishes by default; pass `--no-publish` to keep a test out of the library.

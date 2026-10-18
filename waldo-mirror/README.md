# waldo-mirror

Everything that moves the G1's arms beyond the built-in gestures: live mirroring from a phone camera,
and dancing (freestyle on the beat, or a Just Dance routine). Legs always stay with Unitree's own
balance controller.

| File | What it does |
|------|--------------|
| `pose.py` | MediaPipe body landmarks → each arm's reach in the person's torso frame → robot hand targets (mirror mode, workspace clamps) |
| `robot.py` | Unitree's arm IK (from `~/xr_teleoperate`) with comfort limits and a natural-posture preference; `ArmSdk`, a careful `rt/arm_sdk` controller (speed cap, 2 s ramped hand-over and release, slow waist yaw) |
| `mirror.py` | Live mirroring from the phone camera (`waldo-live` phone page) |
| `moves.py` | 7 dance moves (8 beats each) clipped to comfort ranges, energy fitted to the speed cap at the current tempo |
| `dance.py` | Dance engine: armed and idle until a dance call starts it; follows the beat, plays the voice agent's phrases or improvises, plays a synced Just Dance routine |
| `choreo.py` | Just Dance worker: YouTube search → video/audio tracks → MediaPipe at 15 fps → IK → speed-capped motion; syncs a routine to the call audio |
| `live.sh` | `check`, `start`/`stop` (mirroring), `dance`/`dance-dry`/`dance-off` (dance engine + choreo worker) |

## Setup (on Waldo)

```sh
./setup.sh      # venv on the robot SDK's Python (tv env) + MediaPipe + yt-dlp + the pose model; numpy pinned
.venv/bin/python -m pytest -q
```

## Mirroring

```sh
./run.sh                           # dry run: prints the robot's arm angles, sends nothing
./run.sh --debug-dir /tmp/dbg      # also saves an annotated frame per second
./live.sh start | stop             # live in tmux window waldo:mirror, 0.8 rad/s, starts once the phone sends
```

## Dancing

```sh
./live.sh dance-dry                # arm the engine without touching the robot (logs every move)
./live.sh dance                    # arm it live (1 rad/s); it moves only during a dance call
./live.sh dance-off                # disarm; arms released if it was dancing
./choreo.sh prepare "levitating dua lipa"   # learn a routine ahead of time (~100 s for a 5-minute song)
```

A dance starts when the operator texts "dance", Waldo calls, and it hears a steady beat; it says
"I can feel that song in my joints! Dancing in 3, 2, 1!" and dances until the song ends (8 s of
quiet), the operator says or texts "stop", or the call ends. Each phrase of 4 moves comes from the
voice agent listening to the music; between phrases the engine improvises from the loudness.
Texting the song name switches to the Just Dance routine once it's ready and synced.

## Safety

- Robot balancing (FSM 500), on the gantry, a spotter holding the controller (L1+A overrides everything).
- Comfort limits on every joint, speed caps (0.8 rad/s mirroring, 1 rad/s dancing), hands kept in front
  of the torso and on their own side, 10-minute cap on one dance.
- Every stop path glides the arms back to where they started and hands them to the stock controller.
- Downloading from YouTube is against its terms of service: fine for a lab demo, not for a product.

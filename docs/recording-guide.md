# Recording clips for the Jev watcher

Goal: labeled frames of the cup task, showing what goes right and what goes wrong, so we can measure how well
Jev tells them apart and tune the checks before the demo.

## Ground rules

1. **Use the room webcam (Logitech C920 on Waldo), not a phone.** Jev judges exactly what that camera sees.
   Don't move or re-aim it once recording starts. If it has to move, re-record everything.
2. **Keep the task setup the same as the demo:** same table, red cup, lighting, background.
3. **Make the arm move with the leader arm (teleop), not autonomously.** That way you can stage slips and misses on
   purpose. Someone keeps the follower's stop within reach at all times.
4. **"Person near the arm" shots: arm torque off or the arm parked, never moving.** Never put a hand near a
   powered, moving arm to get a shot.
5. Each take is 3–5 s of one situation. Between takes, change something: cup position, angle, distance, which hand.

## How to record (on Waldo)

```sh
cd ~/HACKATHON2026/waldo-live
./run.sh record --step grasp --label grip=slipping --label person=no --seconds 4
```

It counts down 3 s, saves 5 frames a second to `~/clips/<step>/`, and labels them. Set up the situation
*before* the countdown ends, and keep it going for the whole take. A wrong label is worse than no label:
if a take didn't show what you meant, tell us and we'll delete it.

## Shot list (about 8 takes each)

| Step | Labels | What it should look like |
|------|--------|--------------------------|
| `grasp` | `grip=secure person=no` | Cup firmly between the gripper fingers |
| `grasp` | `grip=slipping person=no` | Cup tilting or sliding out of the fingers |
| `grasp` | `grip=missed person=no` | Gripper closed beside or above the cup, cup not between the fingers |
| `lift` | `lifted=held person=no` | Cup off the table, still in the gripper |
| `lift` | `lifted=falling person=no` | Cup dropping or tipping out of the gripper |
| `lift` | `lifted=on_table person=no` | Arm up but cup still on the table, or knocked over |
| `approach` | `progress=yes person=no` | Gripper moving toward the cup |
| `approach` | `progress=no person=no` | Gripper moving away, stopped elsewhere, or heading to the wrong thing |
| `grasp` | `grip=secure person=yes` | **Arm parked or torque off.** A hand or arm within reach of the robot arm |
| `approach` | `progress=no person=yes` | **Arm parked or torque off.** Someone leaning over the bench |

**Also worth a few takes each:** another object next to the cup, the cup partly hidden by the arm, lights
dimmer or brighter, the cup at the edge of the frame.

That's about 80 takes, 30–45 minutes for one person plus one on the leader arm.

## After recording

```sh
./run.sh watch bench --task tasks/lift_cup.json --step grasp --frames ~/clips/grasp --labels ~/clips/grasp/labels.json
```

Repeat for `lift` and `approach`. This prints Jev's answer per frame, its speed, and its accuracy. When a check
does badly, rewrite its question or answer descriptions in `tasks/lift_cup.json` and rerun the bench.
The frames stay put, so tuning needs no re-recording.

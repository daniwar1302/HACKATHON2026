# HACKATHON2026

Work on **Waldo**, the robot, for Hackathon 2026.

## Connecting to Waldo

1. Install [Tailscale](https://tailscale.com/download), sign in, and accept the invite for the machine `waldo-waldos`.
2. Open an SSH session that also forwards the camera stream:

   ```sh
   ssh -L 8766:10.2.1.180:8766 waldo@100.120.182.66
   ```

   Ask the team for the password.
3. While that session is open, go to <http://localhost:8766/> to see the head camera and the room webcam.

Closing the SSH session also cuts off the camera stream.

If you can't connect, check that Tailscale says **Connected** and that `waldo-waldos` is in its machine list.

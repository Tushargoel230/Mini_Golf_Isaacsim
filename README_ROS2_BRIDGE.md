# Golf RL Policy — ROS2 Real Robot Bridge
**myCobot280 Pi ↔ Isaac Lab (SKRL PPO)**
*TU Dortmund · Tushar Goel · 2025*

---

## What This Does

Runs a trained RL golf policy in Isaac Sim and mirrors the stroke to the physical myCobot280 Pi arm over ROS2.
Transitions (home → init, post-stroke → home) are handled as single smooth `sync_send_angles` calls on the Pi; only the stroke phase streams real-time joint commands from the sim.

```
Isaac Sim (PC)                  File Bridge              ROS2                          Real Robot (Pi)
──────────────────              ────────────             ──────────────                ───────────────
golf_play_ros2_bridge.py ──→  /tmp/golf_joints.json  →  golf_joint_relay.py  →  /mycobot280_tushar/joint_commands  →  golf_robot_bridge.py  →  pymycobot  →  arm
golf_hole_selector.py    ──→  /tmp/golf_hole.json       (hole 1/2/3 selection)
```

**Why the file bridge?**
Isaac Lab uses Python 3.11 (conda). ROS2 Humble only has Python 3.10 bindings.
`rclpy` cannot be imported inside Isaac Lab. The relay runs with `/usr/bin/python3`
(system Python 3.10) and bridges the gap.

**Why 10 Hz?**
The relay publishes at 10 Hz (one command every 100 ms). This gives the real arm enough time
to partially execute each waypoint before the next arrives, producing smooth motion.
At 25–30 Hz the arm receives commands faster than it can move, causing jerky/no motion.

---

## Files

| File | Where it runs | What it does |
|------|--------------|--------------|
| `scripts/golf_play_ros2_bridge.py` | PC — Isaac Lab conda env | Runs RL policy, animates sim robot, writes joints + control signals to `/tmp/` |
| `scripts/golf_joint_relay.py` | PC — system Python 3.10 | Reads `/tmp/golf_joints.json`, publishes to ROS2 at 30 Hz |
| `scripts/golf_hole_selector.py` | PC — any Python | Terminal UI to select hole 1/2/3 |
| `scripts/golf_robot_bridge.py` | **Robot Pi** | Subscribes to joint commands, clamps to hardware limits, drives arm via pymycobot at 10 Hz |
| `scripts/golf_robot_controller.py` | Robot Pi (experimental) | Phase-aware controller using `sync_send_angles` for transitions — not yet in use |
| `scripts/golf_play_holes.py` | PC — Isaac Lab conda env | Sim-only version (no ROS2), for testing without real robot |

---

## Hardware / Software Setup

| | PC | Robot Pi |
|--|----|----|
| OS | Ubuntu 22.04 | Ubuntu (on Raspberry Pi inside arm) |
| Python (Isaac) | 3.11 — conda env `env_isaaclab` | — |
| Python (ROS2) | 3.10 — `/usr/bin/python3` | 3.10 — `/usr/bin/python3` |
| ROS2 | Humble | Humble |
| pymycobot | not needed | 4.0.3 |
| Robot serial port | — | `/dev/serial0` @ 1000000 baud |

**CRITICAL — ROS2 Domain ID:**
Both machines must use the same `ROS_DOMAIN_ID`.
- Pi is set to **44**
- PC default is 0 — export 44 on every PC terminal that uses ROS2

```bash
export ROS_DOMAIN_ID=44   # required on PC before any ros2 command
```

To make it permanent on PC:
```bash
echo 'export ROS_DOMAIN_ID=44' >> ~/.bashrc
```

---

## How to Run (up to 5 terminals)

### Terminal 1 — Isaac Sim + Policy (PC, conda env)
```bash
conda activate env_isaaclab
cd ~/code-goel/MyCobot_RL/Mini_G
python scripts/golf_play_ros2_bridge.py --task Mini-G-Golf-v0 --ros2 \
    --checkpoint /home/goel/code-goel/MyCobot_RL/Mini_G/logs/skrl/mini_g_golf/2026-04-02_17-51-05_ppo_torch_ppo/checkpoints/best_agent.pt
```
Sim robot goes to HOME pose and waits. The checkpoint path above is the default — omit `--checkpoint` and it is used automatically.

### Terminal 2 — ROS2 Relay (PC, system Python)
```bash
export ROS_DOMAIN_ID=44
source /opt/ros/humble/setup.bash
cd ~/code-goel/MyCobot_RL/Mini_G
/usr/bin/python3 scripts/golf_joint_relay.py
```
Reads `/tmp/golf_joints.json` and publishes to `/mycobot280_tushar/joint_commands` at 30 Hz.
Active only during the stroke phase — transitions are driven by the Pi controller directly.

### Terminal 3 — Hole Selector (PC, any terminal)
```bash
cd ~/code-goel/MyCobot_RL/Mini_G
python3 scripts/golf_hole_selector.py
# Type 1, 2, or 3 then Enter to trigger a stroke
```

### Terminal 4 — Robot Bridge (SSH into Pi)
```bash
ssh mycobot@129.217.130.32   # password: see lab notes (do not commit to git)
source /opt/ros/humble/setup.bash
cd ~/tushar/ros2_ws/src
python3 golf_robot_bridge.py --speed 80
```
Subscribes to `/mycobot280_tushar/joint_commands` and forwards every command to the arm via pymycobot.
The relay runs at **10 Hz** — one command every 100 ms — which gives the real arm enough time to
partially execute each step before the next arrives, producing smooth motion.

### Terminal 5 — Live Plot (PC, system Python) ← optional
```bash
export ROS_DOMAIN_ID=44
source /opt/ros/humble/setup.bash
cd ~/code-goel/MyCobot_RL/Mini_G
/usr/bin/python3 scripts/golf_plot_comparison.py [--save] [--outdir logs/tracking]
```
Shows 6 subplots: **blue = sim commanded**, **orange dashed = real measured**.
Each subplot title shows current per-joint error in degrees; figure title shows overall RMS.

`--save` records every data point to a timestamped CSV (`logs/tracking/golf_tracking_YYYYMMDD_HHMMSS.csv`)
saved automatically when the window is closed. Columns: `time_s, source (sim/real), j1_deg … j6_deg`.

Install matplotlib on system Python if missing:
```bash
/usr/bin/pip3 install matplotlib
```

---

## Hole Positions

| Input | Label | Angle | Direction |
|-------|-------|-------|-----------|
| `1` | Right hole | 85° | Slightly right of center |
| `2` | Center hole | 90° | Straight ahead (default) |
| `3` | Left hole | 95° | Slightly left of center |

All holes are 0.28 m from the ball position.

---

## Full Motion Sequence (per stroke)

```
Type hole number (1/2/3)
        ↓
[play script] write_control({"cmd": "init", "hole": N})
[Pi]          sync_send_angles → HOME → INIT pose   (smooth, one command)
[sim]         animate_joints   → HOME → INIT pose   (visual only, not streamed)
        ↓
[play script] env.reset() — Isaac initialises episode, hole marker moves
[play script] write_control({"cmd": "stroke"})
[Pi]          enables real-time _on_cmd forwarding
        ↓
Policy runs — EE swings through ball toward hole
Joint commands streamed: sim → /tmp/golf_joints.json → relay → ROS2 → Pi → pymycobot
        ↓
Episode ends (ball in hole ✓ or timeout)
        ↓
[play script] write_control({"cmd": "home"})
[Pi]          sync_send_angles → HOME pose   (smooth, one command)
[sim]         animate_joints   → HOME pose   (visual only, not streamed)
        ↓
Wait for next input
```

---

## Joint Hardware Limits

From `pymycobot/robot_info.py` — authoritative firmware limits applied in the controller:

| Joint | Min (°) | Max (°) |
|-------|---------|---------|
| J1 | −168 | +168 |
| J2 | −140 | +140 |
| J3 | −150 | +150 |
| J4 | −150 | +150 |
| J5 | −155 | +160 (asymmetric) |
| J6 | −180 | +180 |

All joint commands are clamped to these limits before `send_angles` / `sync_send_angles`.

---

## Verification Steps

**Check control file is being written:**
```bash
# On PC, after selecting a hole
cat /tmp/golf_control.json
# Should show: {"cmd": "stroke", "seq": 2}  (or whichever phase)
```

**Check joint stream during stroke:**
```bash
export ROS_DOMAIN_ID=44
source /opt/ros/humble/setup.bash
ros2 topic echo /mycobot280_tushar/joint_commands
# Should print arrays of 7 floats during stroke, silent during transitions
```

**Check Pi feedback:**
```bash
ros2 topic echo /mycobot280_tushar/joint_states
# Updates at 20 Hz continuously
```

**Angle sanity check:**
- HOME pose → `[0, 0, 0, 0, 0, 0]`
- INIT pose (hole 2) → `[0, −100, −38.8, 0, 90, 0]`

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'isaaclab'` | Not in conda env | `conda activate env_isaaclab` |
| `ModuleNotFoundError: rclpy._rclpy_pybind11` | rclpy inside conda Python 3.11 | Use `/usr/bin/python3` for relay, NOT conda python |
| Topics visible on Pi but no data | `ROS_DOMAIN_ID` mismatch | Set `export ROS_DOMAIN_ID=44` on PC |
| Topics not visible at all on Pi | DDS multicast / interface issue | `export CYCLONEDDS_URI='<CycloneDDS><Domain><General><NetworkInterfaceAddress>YOUR_IFACE</NetworkInterfaceAddress></General></Domain></CycloneDDS>'` |
| Pi ignores control file | Old `seq` number | Restart controller — it resets `_last_seq = -1` |
| Real arm doesn't move to init pose | Pi still in wrong state | Check Pi logs for state transitions; `cat /tmp/golf_control.json` |
| Arm moves during transitions (unexpected) | Old `golf_robot_bridge.py` still running | Kill it; only `golf_robot_controller.py` should run on Pi |
| Policy runs with random behaviour | Checkpoint not loaded | Check that `best_agent.pt` exists under `Mini_G/logs/` |
| send_angles errors at limit | Sim joint exceeded hardware range | Check `soft_joint_pos_limit_factor` in env cfg; currently 0.90 |

---

## Copy Bridge Script to Pi

Run from `Mini_G/` when you update `golf_robot_bridge.py`:
```bash
scp scripts/golf_robot_bridge.py mycobot@129.217.130.32:~/tushar/ros2_ws/src/
```

---

## What's Done / What's Next

- [x] RL policy trained — golf push task, SKRL PPO, 18-dim obs, Cartesian actions
- [x] Sim demo — hole selection 1/2/3, home↔init animation, state machine
- [x] ROS2 bridge — Isaac Sim → file → relay → Pi → pymycobot
- [x] ROS2 domain ID fix — Pi uses ID 44, PC must match
- [x] Correct joint limits from pymycobot `robot_info.py` (all 6 joints, including J5 asymmetric −155~+160)
- [x] `set_fresh_mode(1)` on Pi — firmware discards queued commands, always runs latest
- [x] Relay rate tuned to 10 Hz — gives real arm 100 ms per step, smooth motion confirmed on hardware
- [x] Stale JSON cleanup on relay + play script startup
- [x] Actuator params tuned to match real hardware (velocity_limit_sim=3.0, effort=3.0, stiffness=40, damping=5.6)
- [ ] Test first real stroke end-to-end with phase-aware controller
- [ ] Retrain policy with updated physics parameters for tighter sim-to-real tracking
- [ ] Camera input for real ball position (ROS2 → observation bridge)

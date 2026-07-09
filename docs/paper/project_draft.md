# Marble Maze CPS Project: Hardware Working Draft

*Simple overview of the hardware: what it is built from and how the parts
connect. Software (perception, control) is documented separately and is still
changing, so it is intentionally left out here.*

Group 10.

---

## 1. What the project does

We built a cyber-physical system that **solves a physical wooden labyrinth
automatically**. A steel marble sits on a wooden maze board that can tilt on two
axes. A camera looks down at the board, and two servo motors tilt the board to
roll the marble along a path from the start to the goal, with no human input
after the marble is placed.

At the hardware level the chain is:

```
camera  ->  computer  ->  Arduino  ->  PWM driver  ->  servos  ->  board tilt
```

---

## 2. Component list (what we bought)

| Component | Description | Qty |
|---|---|---|
| Arduino UNO R4 Minima | Microcontroller (USB-C) | 1 |
| PCA9685 | 16-channel PWM servo driver (I²C) | 1 |
| Digital metal-gear servo, 13 kg | Tilt actuators (2 used + 2 spare) | 4 |
| Servo horn set | RC metal/aluminium horns (MG996R/25T fit) | 1 set |
| M3 tie-rod ends | Ball-joint rod ends (linkage) | 1 set |
| M3 × 100 mm threaded rods | Push-rod stock | 1 set |
| 5 V 10 A power supply (50 W) | Dedicated servo power | 1 |
| USB2.0 UVC camera, OV9281 | Overhead global-shutter mono camera | 1 |
| K&F Magic Arm + Super Clamp | Overhead camera mount | 1 |
| Wooden labyrinth board | The maze itself (2×) | 2 |

Also used: jumper wires.

---

## 3. How the parts are connected

There are two separate electrical "sides" that share a common ground:

**Logic side (low power, from USB):**
- The controlling computer connects to the Arduino UNO R4 Minima over **USB**.
- Arduino → PCA9685 over **I²C**: `5V→VCC`, `GND→GND`, `SDA→SDA`, `SCL→SCL`.

**Servo power side (high power, separate rail):**
- 5 V / 10 A supply → PCA9685 **V+ / GND** screw terminal.
- The two servos plug into PCA9685 **channel 0 (yaw axis)** and **channel 1
  (pitch axis)**.

The grounds of both sides meet at the PCA9685, giving the PWM signal a shared
reference. Keeping servo power on its own rail stops motor current spikes from
disturbing the Arduino and camera.

**Mechanical connection (servo → board):**
Each servo drives one tilt axis of the board through a push-rod linkage
(one linkage per axis):

- The servo is mounted **horizontally underneath the board**.
- A **straight-arm (bar) horn** on the servo rotates in a plane parallel to the
  board. One axis uses a double-arm (two-sided) horn and the other a single-arm
  (one-sided) horn, but both have the **same arm length**, so the two axes get
  the same throw.
- A **screw at the tip of a horn arm** connects to a **tie-rod end**.
- That tie-rod end is joined to a **second tie-rod end by a threaded rod**
  (which also lets us adjust the link length).
- The far tie-rod end is **screwed into a piece of wood, which is itself screwed
  to the board**.
- As the horn rotates, the push-rod pushes/pulls that wood piece, tilting the
  board on that axis.

This linkage replaced an earlier direct-coupling design (see notes below).

**Camera:**
- USB camera mounted overhead on the Magic Arm + clamp, looking straight down at
  the board, referenced to the same base so calibration stays valid.

---

## 4. Command interface and on-board safety

The Arduino accepts simple text commands over USB and drives the servos through
the PCA9685:

- `SET <yaw> <pitch>`: tilt the two axes (values -1 to 1).
- `NEUTRAL`: return the board to level.
- `PING` / `PONG`: check the link is alive.

Safety runs on the Arduino itself, independent of the computer:

- **Travel limit**: won't tilt past a safe range (won't drive servos into the
  board's mechanical stops).
- **Smooth ramping**: no sudden jerks between commands.
- **Watchdog**: if commands stop arriving, the board returns to level within
  0.5 s, so a lost connection fails safe.

---

## 5. Notes / iterations (things we changed along the way)

- **Servo linkage:** the first design used a circular horn screwed directly to
  the board (direct coupling). It did not give enough range of motion. We
  replaced it with the push-rod linkage described above (horn → tie rod →
  threaded rod → tie rod → wooden lever), which gives a much larger tilt range.
- **Firmware travel limits and the wire failure:** to try to get more range, we
  removed the servo travel limits in the firmware. The servo then drove to its
  extreme and stalled, and the servo-power jumper wires overheated and burned.
  We restored the firmware travel limits (so the servo can no longer stall), and
  instead obtained the extra range mechanically from the push-rod linkage. We
  continue to use the same jumper wires; the fix was keeping the firmware limits,
  not heavier wiring.
- **Firmware speed:** raised the serial link and I²C clock and made the servo
  update run at 200 Hz, to reduce control latency.
- **Servos are RC (position) servos**, so they don't report their angle back,
  so the **camera** is used as the feedback sensor instead.

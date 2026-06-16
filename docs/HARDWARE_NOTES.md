# Hardware Notes

## Servos

Given vendor specs:

- Operating voltage: 4.8 V to 7.2 V.
- Recommended voltage: 5 V.
- Torque: 12 kg-cm at 6 V.
- Speed: 0.13 s/60 deg at 6 V, 0.17 s/60 deg at 4.8 V.
- Suggested current: 3 A per servo.

The vendor PWM description is contradictory. Bench-test the real mapping before installation.

Start with firmware limits around:

```text
min_us = 1200
neutral_us = 1500
max_us = 1800
```

Widen only after checking mechanical safety.

## Power

Use the 5 V 10 A supply for servo power. Do not power servos from the Arduino.

Required wiring:

- Servo supply 5 V -> PCA9685 V+.
- Servo supply GND -> PCA9685 GND.
- Arduino GND -> PCA9685 GND.
- Arduino SDA/SCL -> PCA9685 SDA/SCL.

Add a large capacitor across V+ and GND near the PCA9685.

## Camera

Reflective silver marble detection depends strongly on lighting. Use fixed lighting and lock camera exposure/gain where possible.

Verify actual modes with:

```bash
v4l2-ctl --list-formats-ext
```

USB2 may not support raw 720p at 120 FPS. Use the fastest stable mode that gives reliable tracking.

## Mechanical

Rigid mounts matter more than raw servo torque. The camera mount must not shift during a run or calibration becomes invalid.

Add mechanical stops where possible, and keep firmware limits conservative.


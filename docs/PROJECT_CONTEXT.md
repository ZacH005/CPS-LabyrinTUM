# Project Context

## Objective

Create an autonomous marble maze system that solves a physical wooden labyrinth without user input after the ball is placed at the start.

## Hardware

- USB2 UVC global-shutter mono camera, OV9281 class, 720p/120fps advertised.
- Arduino UNO R4 Minima.
- PCA9685 16-channel I2C PWM servo driver.
- Hobby digital metal-gear servos, 12 kg-cm at 6 V, 0.13 s/60 deg at 6 V.
- 5 V 10 A power supply.
- Wooden labyrinth board with a different layout from CyberRunner.
- M3 rods, ball joints, servo horns, and camera magic arm.

## Constraints

- Full-maze solve is required for the final demo.
- Manual ball reset is acceptable.
- Drilling/modifying the maze is allowed.
- RTX 4050 is available, but RL is not required.
- Ubuntu directly on the host machine is likely available.

## Strategy

Use a classical cyber-physical control stack:

1. Track ball position from overhead camera.
2. Convert image coordinates to board coordinates.
3. Follow a manually annotated maze path.
4. Drive yaw/pitch servos through Arduino/PCA9685.
5. Tune controller section by section.

## Main Risks

- Reflective silver marble can produce unstable detection under uncontrolled lighting.
- Hobby servos may have backlash and no position feedback.
- Camera bandwidth may not support raw 720p at 120 FPS over USB2.
- Mechanical flex in the camera mount can invalidate calibration.
- Servo PWM mapping from the vendor listing is contradictory and must be bench-tested.


#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

from cps_maze.config import load_config
from cps_maze.hardware.serial_link import ArduinoServoLink, ServoCommand


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--axis", choices=["yaw", "pitch"], required=True)
    parser.add_argument("--amplitude", type=float, default=0.15)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    args = parser.parse_args()

    if not 0.0 < args.amplitude <= 1.0:
        raise ValueError("--amplitude must be in (0, 1]")

    config = load_config(args.config)
    with ArduinoServoLink(
        port=config.serial["port"],
        baudrate=int(config.serial["baudrate"]),
        timeout_s=float(config.serial["timeout_s"]),
    ) as link:
        time.sleep(2.0)
        commands = [-args.amplitude, 0.0, args.amplitude, 0.0]
        for value in commands:
            yaw = value if args.axis == "yaw" else 0.0
            pitch = value if args.axis == "pitch" else 0.0
            print(f"Sending {args.axis}={value:.3f}")
            link.send(ServoCommand(yaw=yaw, pitch=pitch))
            time.sleep(args.hold_seconds)
        link.neutral()


if __name__ == "__main__":
    main()


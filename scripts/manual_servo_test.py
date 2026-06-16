#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

from cps_maze.config import load_config
from cps_maze.hardware.serial_link import ArduinoServoLink, ServoCommand


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--neutral", action="store_true")
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()

    config = load_config(args.config)
    serial_config = config.serial

    with ArduinoServoLink(
        port=serial_config["port"],
        baudrate=int(serial_config["baudrate"]),
        timeout_s=float(serial_config["timeout_s"]),
    ) as link:
        time.sleep(2.0)
        if args.neutral:
            link.neutral()
        else:
            link.send(ServoCommand(yaw=args.yaw, pitch=args.pitch))
        time.sleep(args.seconds)
        link.neutral()


if __name__ == "__main__":
    main()


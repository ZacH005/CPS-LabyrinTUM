from __future__ import annotations

from dataclasses import dataclass

import serial


@dataclass(frozen=True)
class ServoCommand:
    yaw: float
    pitch: float

    def clamped(self) -> "ServoCommand":
        return ServoCommand(
            yaw=max(-1.0, min(1.0, self.yaw)),
            pitch=max(-1.0, min(1.0, self.pitch)),
        )


class ArduinoServoLink:
    def __init__(self, port: str, baudrate: int, timeout_s: float):
        self.serial = serial.Serial(port=port, baudrate=baudrate, timeout=timeout_s)

    def send(self, command: ServoCommand) -> None:
        safe = command.clamped()
        line = f"SET {safe.yaw:.4f} {safe.pitch:.4f}\n"
        self.serial.write(line.encode("ascii"))

    def neutral(self) -> None:
        self.serial.write(b"NEUTRAL\n")

    def ping(self) -> str:
        self.serial.write(b"PING\n")
        return self.serial.readline().decode("ascii", errors="replace").strip()

    def close(self) -> None:
        self.serial.close()

    def __enter__(self) -> "ArduinoServoLink":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


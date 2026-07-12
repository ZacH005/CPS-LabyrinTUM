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


def apply_trim(command: ServoCommand, trim_yaw: float, trim_pitch: float) -> ServoCommand:
    """Offset a command by the neutral trim and clamp to the valid range.

    The trim is the measured command at which the board is actually LEVEL
    (the table/frame is slanted, so servo neutral is not level). Applying it
    here means command (0, 0) always means "level board" for every tool.
    """
    return ServoCommand(
        yaw=max(-1.0, min(1.0, command.yaw + trim_yaw)),
        pitch=max(-1.0, min(1.0, command.pitch + trim_pitch)),
    )


class ArduinoServoLink:
    def __init__(self, port: str, baudrate: int, timeout_s: float,
                 trim_yaw: float = 0.0, trim_pitch: float = 0.0):
        self.trim_yaw = float(trim_yaw)
        self.trim_pitch = float(trim_pitch)
        self.serial = serial.Serial(port=port, baudrate=baudrate, timeout=timeout_s)

    def set_trim(self, trim_yaw: float, trim_pitch: float) -> None:
        self.trim_yaw = float(trim_yaw)
        self.trim_pitch = float(trim_pitch)

    def send(self, command: ServoCommand) -> None:
        safe = apply_trim(command.clamped(), self.trim_yaw, self.trim_pitch)
        line = f"SET {safe.yaw:.4f} {safe.pitch:.4f}\n"
        self.serial.write(line.encode("ascii"))

    def neutral(self) -> None:
        """Go to LEVEL: the trimmed neutral when a trim is set.

        The firmware's own NEUTRAL (and its watchdog fallback) remain the raw
        servo center - that stays the crash-safe state; a trimmed neutral only
        holds while something keeps streaming."""
        if self.trim_yaw != 0.0 or self.trim_pitch != 0.0:
            self.send(ServoCommand(yaw=0.0, pitch=0.0))
        else:
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

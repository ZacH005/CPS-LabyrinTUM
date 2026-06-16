# Software Stack

## Recommended Base

- Ubuntu running directly on the host computer.
- Python 3.10 or newer.
- Arduino IDE or Arduino CLI for firmware upload.

## Python Dependencies

- `opencv-python` - camera capture and image processing.
- `numpy` - numeric computations.
- `pyserial` - PC to Arduino command link.
- `PyYAML` - runtime configuration.
- `pytest` - tests.
- `ruff` - linting.

## System Tools

- `v4l-utils` - inspect camera modes.
- `ffmpeg` - record and inspect video.
- `git` - version control.

Install useful Linux tools:

```bash
sudo apt update
sudo apt install -y v4l-utils ffmpeg
```

## Arduino Dependencies

- Arduino UNO R4 board support.
- `Adafruit_PWMServoDriver` library.
- `Wire` library.

## Optional Later

- ROS2, only if the team wants robotics middleware.
- PyTorch, only if adding learned perception or adaptive control.
- Jupyter, only for calibration/data analysis notebooks.


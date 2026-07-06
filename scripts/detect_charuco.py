from cps_maze.vision.aruco import CharucoDetector
from cps_maze.camera import CameraCapture
from cps_maze.config import load_config
import cv2
import argparse

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    charuco_detector = CharucoDetector()

    with CameraCapture(config.camera) as camera:
        while True:
            frame = camera.read()
            detection = charuco_detector.detect(frame.image)
            output = charuco_detector.draw_detection(frame.image, detection)
            cv2.imshow("CharUco Detection", output)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
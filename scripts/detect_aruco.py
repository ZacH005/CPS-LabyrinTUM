from cps_maze.vision.aruco import ArucoDetector
from cps_maze.camera import CameraCapture
from cps_maze.config import load_config
import cv2
import argparse

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    aruco_detector = ArucoDetector()

    with CameraCapture(config.camera) as camera:
        while True:
            frame = camera.read()
            detection = aruco_detector.detect(frame.image)
            output = aruco_detector.draw_detection(frame.image, detection)
            cv2.imshow("ArUco Detection", output)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()